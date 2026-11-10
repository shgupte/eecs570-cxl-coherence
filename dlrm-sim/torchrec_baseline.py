"""TorchRec-style row-wise pooled embedding communication.

This is the primary conventional baseline. It follows the shape of
``RwPooledEmbeddingSharding`` without running TorchRec:

* Sparse ids are sent to the GPU that owns each row. Duplicate ids in a bag
  are separate requests. Ids the sample's own GPU owns are not counted as
  network traffic.
* The owner looks up its rows and forms one partial sum per nonempty
  ``(sample, table, owner)``. Sum and mean pooling move the same number of
  vectors; mean only changes the scale of the values, which we do not store.
* TorchRec then reduce-scatters those partials. The collective is sized by
  the batch and the feature count, including empty partials, via
  ``input_splits = batch_size_per_rank``. We report that dense volume as
  ``reduce_scatter_bytes`` and, separately, the sparse nonempty count. The
  nonempty count is not the reduce-scatter measurement.

Backward is the reverse collective: an all-gather of pooled-output gradients,
one vector per sample per table per rank in the dense buffer. It is not one
gradient vector per embedding row. After the all-gather, each owner applies
the pooled gradient to the rows it contributed. Those per-row applications
are counted and are not network traffic. Each distinct accessed row is then
updated once, through the same ``apply_owner_updates`` path as the
row-transfer model.

A later coherent-cache model must not reuse this backward path. If the
requester pools locally, it holds full rows, and the gradient that returns
to the owner is per row, as in the row-transfer baseline.
"""

from __future__ import annotations

import numpy as np

from baseline import (
    StepResult,
    _matrix_to_lists,
    _pair_transfers,
    apply_owner_updates,
    collect_unique_accesses,
)
from config import SimConfig
from topology import Ownership
from workload import TrainingStep


def reduce_scatter_bytes(config: SimConfig, num_samples: int, num_tables: int) -> int:
    """Bytes sent by a dense pooled reduce-scatter.

    Each rank's buffer holds one partial per sample per table. A ring
    reduce-scatter sends ``(G - 1) / G`` of that buffer from every rank, so
    the job total is ``(G - 1) * samples * tables * embedding_bytes``.
    NCCL's algorithm and empty-bag compression are not modeled.
    """
    if config.num_gpus == 1:
        return 0
    return (config.num_gpus - 1) * num_samples * num_tables * config.embedding_bytes


def allgather_backward_bytes(config: SimConfig, num_samples: int, num_tables: int) -> int:
    """Bytes sent when pooled gradients are all-gathered back to every owner.

    Each rank contributes its own samples. The job total matches the forward
    reduce-scatter total above. This is the same style of approximation.
    """
    return reduce_scatter_bytes(config, num_samples, num_tables)


def _gpu_of_column(step: TrainingStep) -> np.ndarray:
    batch = step.batch
    gpu_of_column = np.empty(batch.batch_size, dtype=np.int64)
    first = int(batch.sample_ids[0])
    for sample_range in step.ranges:
        start = sample_range.sample_start - first
        end = sample_range.sample_end - first
        gpu_of_column[start:end] = sample_range.gpu_id
    return gpu_of_column


def account_pooled(step: TrainingStep, ownership: Ownership) -> dict:
    """Count id traffic and nonempty partials without deduplicating bags first."""
    num_gpus = ownership.num_gpus
    batch = step.batch
    gpu_of_column = _gpu_of_column(step)
    id_matrix = np.zeros((num_gpus, num_gpus), dtype=np.int64)
    partial_matrix = np.zeros((num_gpus, num_gpus), dtype=np.int64)
    owner_lookups = 0
    id_remote = 0
    partials = 0
    remote_partials = 0

    for table_index, table_id in enumerate(batch.table_ids):
        lengths = np.asarray(batch.lengths[table_index])
        rows = batch.table_window(table_index, 0, batch.batch_size)
        if int(rows.size) != int(lengths.sum()):
            raise RuntimeError(f"bag lengths do not cover table {table_id}")
        owner_lookups += int(rows.size)
        if rows.size == 0:
            continue
        row_owners = ownership.owners(int(table_id), rows)
        sample_of_row = np.repeat(np.arange(batch.batch_size, dtype=np.int64), lengths)
        sample_gpu = gpu_of_column[sample_of_row]
        remote = row_owners != sample_gpu
        id_remote += int(np.count_nonzero(remote))
        if np.any(remote):
            pair = sample_gpu[remote] * num_gpus + row_owners[remote]
            id_matrix += np.bincount(pair, minlength=num_gpus * num_gpus).reshape(num_gpus, num_gpus)

        keys = sample_of_row * num_gpus + row_owners
        unique_keys = np.unique(keys)
        partials += int(unique_keys.size)
        unique_sample = unique_keys // num_gpus
        unique_owner = unique_keys - unique_sample * num_gpus
        unique_gpu = gpu_of_column[unique_sample]
        remote_partial = unique_owner != unique_gpu
        remote_partials += int(np.count_nonzero(remote_partial))
        if np.any(remote_partial):
            pair = unique_owner[remote_partial] * num_gpus + unique_gpu[remote_partial]
            partial_matrix += np.bincount(pair, minlength=num_gpus * num_gpus).reshape(num_gpus, num_gpus)

    # Pooled gradients travel sample GPU -> contributing owner, the reverse of the partial.
    gradient_matrix = partial_matrix.T.copy()
    return {
        "owner_lookups": owner_lookups,
        "id_remote": id_remote,
        "partials": partials,
        "remote_partials": remote_partials,
        "id_matrix": id_matrix,
        "partial_matrix": partial_matrix,
        "gradient_matrix": gradient_matrix,
    }


def simulate_torchrec_step(
    step: TrainingStep,
    config: SimConfig,
    ownership: Ownership,
    versions: dict[tuple[int, int], int] | None,
) -> StepResult:
    """Pooled forward, pooled backward, then one owner update per distinct row."""
    if ownership.num_gpus != config.num_gpus:
        raise ValueError("ownership GPU count does not match the config")
    pooled = account_pooled(step, ownership)
    accesses = collect_unique_accesses(step.batch, step.ranges)
    # Per-row applications stay on the owner. They are not the backward messages.
    updates = apply_owner_updates(
        accesses,
        config.num_gpus,
        versions,
        gradients_aggregated=True,
        collect_rows=config.collect_rows,
    )
    id_bytes = int(pooled["id_remote"]) * config.request_bytes
    scatter = reduce_scatter_bytes(config, step.batch.batch_size, step.batch.num_tables)
    gather = allgather_backward_bytes(config, step.batch.batch_size, step.batch.num_tables)
    logical_partial_bytes = int(pooled["remote_partials"]) * config.embedding_bytes
    forward_bytes = id_bytes + scatter
    per_gpu = []
    for access in accesses.per_gpu:
        per_gpu.append(
            {
                "gpu_id": access.gpu_id,
                "sample_start": access.sample_start,
                "sample_end": access.sample_end,
                "num_samples": access.sample_end - access.sample_start,
                "raw_accesses": access.raw_accesses,
                "unique_accesses": int(sum(rows.size for rows in access.unique_rows.values())),
            }
        )
    return StepResult(
        model="torchrec",
        ownership_scheme=ownership.scheme,
        step_index=step.step_index,
        origin=step.origin,
        cross_step_locality=step.cross_step_locality,
        raw_accesses=accesses.raw_accesses,
        unique_gpu_row_accesses=int(sum(item["unique_accesses"] for item in per_gpu)),
        local_requests=0,
        remote_requests=0,
        forward_request_bytes=id_bytes,
        forward_response_bytes=scatter,
        forward_bytes=forward_bytes,
        backward_bytes=gather,
        total_bytes=forward_bytes + gather,
        local_gradient_contributions=0,
        remote_gradient_vectors=0,
        updated_rows=int(updates["updated_rows"]),
        cache_hits=0,
        logical_pair_transfers=_pair_transfers(
            pooled["id_matrix"], pooled["partial_matrix"], pooled["gradient_matrix"]
        ),
        sharing_histogram=updates["sharing_histogram"],
        mean_sharing=float(updates["mean_sharing"]),
        per_gpu=per_gpu,
        forward_request_vectors=_matrix_to_lists(pooled["id_matrix"]),
        forward_response_vectors=_matrix_to_lists(pooled["partial_matrix"]),
        backward_vectors=_matrix_to_lists(pooled["gradient_matrix"]),
        versions_touched=dict(versions) if versions is not None and config.collect_rows else None,
        updated_row_ids=updates["updated_row_ids"],
        contributors=updates["contributors"],
        owner_lookups=int(pooled["owner_lookups"]),
        id_request_count=int(pooled["id_remote"]),
        partial_pooled_vectors=int(pooled["partials"]),
        remote_partial_pooled_vectors=int(pooled["remote_partials"]),
        logical_remote_partial_bytes=logical_partial_bytes,
        reduce_scatter_bytes=scatter,
        pooled_gradient_vectors=int(pooled["partials"]),
        remote_pooled_gradient_vectors=int(pooled["remote_partials"]),
        logical_remote_pooled_gradient_bytes=logical_partial_bytes,
        allgather_backward_bytes=gather,
        per_row_remote_gradient_vectors=0,
        row_gradient_applications=int(pooled["owner_lookups"]),
    )


def simulate_torchrec(
    steps: tuple[TrainingStep, ...] | list[TrainingStep],
    config: SimConfig,
    ownership: Ownership,
) -> tuple[list[StepResult], dict[tuple[int, int], int] | None]:
    """Run the pooled baseline. The ownership map is the caller's, not inferred here."""
    versions: dict[tuple[int, int], int] | None = {} if config.track_versions else None
    results = [simulate_torchrec_step(step, config, ownership, versions) for step in steps]
    return results, versions

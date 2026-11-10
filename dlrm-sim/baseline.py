"""Conventional sharded embedding traffic.

There is no remote cache. A GPU that needs a row it does not own fetches that
row from the owner on every training step, including a row it fetched on the
previous step.

Forward responses are one full embedding vector per unique remote row. That
omits embedding-bag pooling. Optimized distributed DLRM implementations often
pool on the owner and send one vector per sample per table, so these byte
counts can overestimate forward traffic.

Requests are charged ``request_bytes`` per unique remote row, not per raw
index. Traffic between a GPU pair is one batched total per phase. The model
does not charge a network round trip per row.

Backward sends one aggregated gradient vector per unique remote row. Repeated
accesses inside one GPU's minibatch are one contribution. Two GPUs that touch
the same row send two contributions. The owner applies one logical update
after those contributions are aggregated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import SimConfig
from meta_trace import EmbeddingBatch
from topology import Ownership
from workload import SampleRange, TrainingStep, column_span


@dataclass
class GpuAccess:
    gpu_id: int
    sample_start: int
    sample_end: int
    raw_accesses: int = 0
    # table_id -> sorted unique row ids accessed by this GPU
    unique_rows: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class UniqueAccesses:
    """Deduplicated accesses for one step. Shared by forward, backward, and update."""

    raw_accesses: int
    per_gpu: tuple[GpuAccess, ...]


@dataclass
class StepResult:
    model: str
    ownership_scheme: str
    step_index: int
    origin: str
    cross_step_locality: str
    raw_accesses: int
    unique_gpu_row_accesses: int
    local_requests: int
    remote_requests: int
    forward_request_bytes: int
    forward_response_bytes: int
    forward_bytes: int
    backward_bytes: int
    total_bytes: int
    local_gradient_contributions: int
    remote_gradient_vectors: int
    updated_rows: int
    cache_hits: int
    logical_pair_transfers: int
    sharing_histogram: dict[int, int]
    mean_sharing: float
    per_gpu: list[dict]
    # Matrices are [source_gpu][destination_gpu], in vectors (not bytes).
    forward_request_vectors: list[list[int]]
    forward_response_vectors: list[list[int]]
    backward_vectors: list[list[int]]
    versions_touched: dict[tuple[int, int], int] | None = None
    updated_row_ids: set[tuple[int, int]] | None = None
    contributors: dict[tuple[int, int], tuple[int, ...]] | None = None
    # TorchRec pooled accounting. Zero on the row-transfer reference model.
    owner_lookups: int = 0
    id_request_count: int = 0
    partial_pooled_vectors: int = 0
    remote_partial_pooled_vectors: int = 0
    logical_remote_partial_bytes: int = 0
    reduce_scatter_bytes: int = 0
    pooled_gradient_vectors: int = 0
    remote_pooled_gradient_vectors: int = 0
    logical_remote_pooled_gradient_bytes: int = 0
    allgather_backward_bytes: int = 0
    per_row_remote_gradient_vectors: int = 0
    row_gradient_applications: int = 0


def collect_unique_accesses(
    batch: EmbeddingBatch,
    ranges: tuple[SampleRange, ...],
) -> UniqueAccesses:
    """Deduplicate ``(table_id, row_id)`` inside each GPU's sample range.

    Raw duplicates are counted before this dedup. Bags are not split apart:
    the window for a GPU is the concatenation of the bags of its samples.
    """
    per_gpu: list[GpuAccess] = []
    raw_accesses = 0
    for sample_range in ranges:
        column_start, column_end = column_span(batch, sample_range)
        access = GpuAccess(
            gpu_id=sample_range.gpu_id,
            sample_start=sample_range.sample_start,
            sample_end=sample_range.sample_end,
        )
        for table_index, table_id in enumerate(batch.table_ids):
            rows = batch.table_window(table_index, column_start, column_end)
            access.raw_accesses += int(rows.size)
            if rows.size:
                access.unique_rows[int(table_id)] = np.unique(rows)
        raw_accesses += access.raw_accesses
        per_gpu.append(access)
    return UniqueAccesses(raw_accesses=raw_accesses, per_gpu=tuple(per_gpu))


def _empty_matrix(num_gpus: int) -> np.ndarray:
    return np.zeros((num_gpus, num_gpus), dtype=np.int64)


def _matrix_to_lists(matrix: np.ndarray) -> list[list[int]]:
    return [[int(value) for value in row] for row in matrix.tolist()]


def route_conventional(accesses: UniqueAccesses, ownership: Ownership) -> dict[str, np.ndarray | int]:
    """Count local lookups and remote full-row fetches. No cache and no pooling."""
    num_gpus = ownership.num_gpus
    request = _empty_matrix(num_gpus)
    response = _empty_matrix(num_gpus)
    local_requests = 0
    remote_requests = 0
    unique_gpu_rows = 0
    per_gpu_local = []
    per_gpu_remote = []
    for access in accesses.per_gpu:
        gpu_local = 0
        gpu_remote = 0
        for table_id, rows in access.unique_rows.items():
            unique_gpu_rows += int(rows.size)
            row_owners = ownership.owners(table_id, rows)
            local_mask = row_owners == access.gpu_id
            gpu_local += int(np.count_nonzero(local_mask))
            remote_owners = row_owners[~local_mask]
            gpu_remote += int(remote_owners.size)
            if remote_owners.size:
                counts = np.bincount(remote_owners, minlength=num_gpus)
                request[access.gpu_id, :] += counts
                response[:, access.gpu_id] += counts
        per_gpu_local.append(gpu_local)
        per_gpu_remote.append(gpu_remote)
        local_requests += gpu_local
        remote_requests += gpu_remote
    return {
        "request": request,
        "response": response,
        "local_requests": local_requests,
        "remote_requests": remote_requests,
        "unique_gpu_row_accesses": unique_gpu_rows,
        "per_gpu_local": per_gpu_local,
        "per_gpu_remote": per_gpu_remote,
    }


def aggregate_gradients(routed: dict) -> dict[str, int]:
    """One local contribution per local unique row, one remote vector per remote row.

    Aggregation is per GPU before communication: the caller already deduplicated.
    This function does not change embedding versions.
    """
    return {
        "local_gradient_contributions": int(routed["local_requests"]),
        "remote_gradient_vectors": int(routed["remote_requests"]),
    }


def apply_owner_updates(
    accesses: UniqueAccesses,
    num_gpus: int,
    versions: dict[tuple[int, int], int] | None,
    *,
    gradients_aggregated: bool,
    collect_rows: bool,
) -> dict:
    """One owner update per distinct accessed row, after gradient aggregation."""
    if not gradients_aggregated:
        raise RuntimeError("owner updates run only after gradient contributions are aggregated")

    histogram = np.zeros(num_gpus + 1, dtype=np.int64)
    updated = 0
    updated_row_ids: set[tuple[int, int]] | None = set() if collect_rows else None
    contributors: dict[tuple[int, int], tuple[int, ...]] | None = {} if collect_rows else None

    table_ids = sorted({table_id for access in accesses.per_gpu for table_id in access.unique_rows})
    for table_id in table_ids:
        pieces_rows = []
        pieces_gpus = []
        for access in accesses.per_gpu:
            rows = access.unique_rows.get(table_id)
            if rows is None or rows.size == 0:
                continue
            pieces_rows.append(rows)
            pieces_gpus.append(np.full(rows.shape[0], access.gpu_id, dtype=np.int64))
        if not pieces_rows:
            continue
        stacked_rows = np.concatenate(pieces_rows)
        stacked_gpus = np.concatenate(pieces_gpus)
        order = np.argsort(stacked_rows, kind="mergesort")
        stacked_rows = stacked_rows[order]
        stacked_gpus = stacked_gpus[order]
        if stacked_rows.size == 0:
            continue
        change = np.flatnonzero(stacked_rows[1:] != stacked_rows[:-1]) + 1
        starts = np.empty(change.size + 1, dtype=np.int64)
        starts[0] = 0
        if change.size:
            starts[1:] = change
        ends = np.empty_like(starts)
        if change.size:
            ends[:-1] = change
        ends[-1] = stacked_rows.size
        degrees = ends - starts
        histogram += np.bincount(degrees, minlength=histogram.size)
        updated += int(degrees.size)

        if versions is not None or collect_rows:
            for start, end in zip(starts.tolist(), ends.tolist()):
                row_id = int(stacked_rows[start])
                key = (table_id, row_id)
                if versions is not None:
                    versions[key] = versions.get(key, 0) + 1
                if collect_rows:
                    gpu_ids = tuple(int(gpu) for gpu in stacked_gpus[start:end].tolist())
                    updated_row_ids.add(key)
                    contributors[key] = gpu_ids

    sharing = {degree: int(histogram[degree]) for degree in range(1, num_gpus + 1) if histogram[degree]}
    mean = 0.0
    if updated:
        mean = float(sum(degree * count for degree, count in sharing.items()) / updated)
    return {
        "updated_rows": updated,
        "sharing_histogram": sharing,
        "mean_sharing": mean,
        "updated_row_ids": updated_row_ids,
        "contributors": contributors,
    }


def _pair_transfers(*matrices: np.ndarray) -> int:
    total = 0
    for matrix in matrices:
        total += int(np.count_nonzero(matrix))
    return total


def simulate_step(
    step: TrainingStep,
    config: SimConfig,
    versions: dict[tuple[int, int], int] | None,
    ownership: Ownership | None = None,
) -> StepResult:
    """Full-vector forward, per-row backward, then one owner update per distinct row."""
    if ownership is None:
        ownership = Ownership.modulo(config.num_gpus)
    accesses = collect_unique_accesses(step.batch, step.ranges)
    routed = route_conventional(accesses, ownership)
    gradients = aggregate_gradients(routed)
    updates = apply_owner_updates(
        accesses,
        config.num_gpus,
        versions,
        gradients_aggregated=True,
        collect_rows=config.collect_rows,
    )

    request = routed["request"]
    response = routed["response"]
    # Gradient vectors follow the same requester -> owner pairs as requests.
    backward = request
    remote = int(routed["remote_requests"])
    forward_request_bytes = remote * config.request_bytes
    forward_response_bytes = remote * config.embedding_bytes
    backward_bytes = int(gradients["remote_gradient_vectors"]) * config.gradient_bytes
    forward_bytes = forward_request_bytes + forward_response_bytes

    per_gpu = []
    for index, access in enumerate(accesses.per_gpu):
        local = int(routed["per_gpu_local"][index])
        remote_gpu = int(routed["per_gpu_remote"][index])
        per_gpu.append(
            {
                "gpu_id": access.gpu_id,
                "sample_start": access.sample_start,
                "sample_end": access.sample_end,
                "num_samples": access.sample_end - access.sample_start,
                "raw_accesses": access.raw_accesses,
                "unique_accesses": local + remote_gpu,
                "local_requests": local,
                "remote_requests": remote_gpu,
                "local_gradient_contributions": local,
                "remote_gradient_vectors": remote_gpu,
                "forward_request_bytes": remote_gpu * config.request_bytes,
                "forward_response_bytes": remote_gpu * config.embedding_bytes,
                "backward_bytes": remote_gpu * config.gradient_bytes,
            }
        )

    return StepResult(
        model="row-transfer",
        ownership_scheme=ownership.scheme,
        step_index=step.step_index,
        origin=step.origin,
        cross_step_locality=step.cross_step_locality,
        raw_accesses=accesses.raw_accesses,
        unique_gpu_row_accesses=int(routed["unique_gpu_row_accesses"]),
        local_requests=int(routed["local_requests"]),
        remote_requests=remote,
        forward_request_bytes=forward_request_bytes,
        forward_response_bytes=forward_response_bytes,
        forward_bytes=forward_bytes,
        backward_bytes=backward_bytes,
        total_bytes=forward_bytes + backward_bytes,
        local_gradient_contributions=int(gradients["local_gradient_contributions"]),
        remote_gradient_vectors=int(gradients["remote_gradient_vectors"]),
        updated_rows=int(updates["updated_rows"]),
        cache_hits=0,
        logical_pair_transfers=_pair_transfers(request, response, backward),
        sharing_histogram=updates["sharing_histogram"],
        mean_sharing=float(updates["mean_sharing"]),
        per_gpu=per_gpu,
        forward_request_vectors=_matrix_to_lists(request),
        forward_response_vectors=_matrix_to_lists(response),
        backward_vectors=_matrix_to_lists(backward),
        versions_touched=dict(versions) if versions is not None and config.collect_rows else None,
        updated_row_ids=updates["updated_row_ids"],
        contributors=updates["contributors"],
        owner_lookups=int(routed["unique_gpu_row_accesses"]),
        per_row_remote_gradient_vectors=remote,
        row_gradient_applications=int(routed["unique_gpu_row_accesses"]),
    )


def simulate(
    steps: tuple[TrainingStep, ...] | list[TrainingStep],
    config: SimConfig,
    ownership: Ownership | None = None,
) -> tuple[list[StepResult], dict[tuple[int, int], int] | None]:
    """Run the row-transfer reference model. Versions persist when enabled."""
    versions: dict[tuple[int, int], int] | None = {} if config.track_versions else None
    results = [simulate_step(step, config, versions, ownership) for step in steps]
    return results, versions

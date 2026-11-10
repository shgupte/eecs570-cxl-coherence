"""Per-step and aggregate traffic metrics, as CSV and JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from baseline import StepResult
from config import SimConfig


STEP_FIELDS = (
    "step",
    "model",
    "ownership",
    "origin",
    "cross_step_locality",
    "raw_accesses",
    "unique_gpu_row_accesses",
    "local_requests",
    "remote_requests",
    "forward_request_bytes",
    "forward_response_bytes",
    "forward_bytes",
    "backward_bytes",
    "total_bytes",
    "local_gradient_contributions",
    "remote_gradient_vectors",
    "per_row_remote_gradient_vectors",
    "owner_lookups",
    "id_request_count",
    "partial_pooled_vectors",
    "remote_partial_pooled_vectors",
    "logical_remote_partial_bytes",
    "reduce_scatter_bytes",
    "pooled_gradient_vectors",
    "remote_pooled_gradient_vectors",
    "logical_remote_pooled_gradient_bytes",
    "allgather_backward_bytes",
    "row_gradient_applications",
    "updated_rows",
    "protocol",
    "cache_bytes",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "invalidations",
    "invalidation_bytes",
    "acknowledgement_bytes",
    "logical_pair_transfers",
    "mean_sharing",
)

_TEXT_FIELDS = {
    "step",
    "model",
    "ownership",
    "origin",
    "cross_step_locality",
    "protocol",
    "cache_bytes",
    "mean_sharing",
}


def _step_row(result: StepResult) -> dict:
    return {
        "step": result.step_index,
        "model": result.model,
        "ownership": result.ownership_scheme,
        "origin": result.origin,
        "cross_step_locality": result.cross_step_locality,
        "raw_accesses": result.raw_accesses,
        "unique_gpu_row_accesses": result.unique_gpu_row_accesses,
        "local_requests": result.local_requests,
        "remote_requests": result.remote_requests,
        "forward_request_bytes": result.forward_request_bytes,
        "forward_response_bytes": result.forward_response_bytes,
        "forward_bytes": result.forward_bytes,
        "backward_bytes": result.backward_bytes,
        "total_bytes": result.total_bytes,
        "local_gradient_contributions": result.local_gradient_contributions,
        "remote_gradient_vectors": result.remote_gradient_vectors,
        "per_row_remote_gradient_vectors": result.per_row_remote_gradient_vectors,
        "owner_lookups": result.owner_lookups,
        "id_request_count": result.id_request_count,
        "partial_pooled_vectors": result.partial_pooled_vectors,
        "remote_partial_pooled_vectors": result.remote_partial_pooled_vectors,
        "logical_remote_partial_bytes": result.logical_remote_partial_bytes,
        "reduce_scatter_bytes": result.reduce_scatter_bytes,
        "pooled_gradient_vectors": result.pooled_gradient_vectors,
        "remote_pooled_gradient_vectors": result.remote_pooled_gradient_vectors,
        "logical_remote_pooled_gradient_bytes": result.logical_remote_pooled_gradient_bytes,
        "allgather_backward_bytes": result.allgather_backward_bytes,
        "row_gradient_applications": result.row_gradient_applications,
        "updated_rows": result.updated_rows,
        "protocol": result.protocol,
        "cache_bytes": (
            ""
            if result.model != "coherent"
            else "unlimited"
            if result.cache_bytes is None
            else result.cache_bytes
        ),
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "cache_evictions": result.cache_evictions,
        "invalidations": result.invalidations,
        "invalidation_bytes": result.invalidation_bytes,
        "acknowledgement_bytes": result.acknowledgement_bytes,
        "logical_pair_transfers": result.logical_pair_transfers,
        "mean_sharing": result.mean_sharing,
    }


def aggregate(results: list[StepResult]) -> dict:
    if not results:
        raise ValueError("no steps to aggregate")
    totals = {key: 0 for key in STEP_FIELDS if key not in _TEXT_FIELDS}
    sharing: dict[int, int] = {}
    for result in results:
        row = _step_row(result)
        for key in totals:
            totals[key] += row[key]
        for degree, count in result.sharing_histogram.items():
            sharing[degree] = sharing.get(degree, 0) + count
    updated = totals["updated_rows"]
    mean = 0.0
    if updated:
        mean = sum(degree * count for degree, count in sharing.items()) / updated
    totals["mean_sharing"] = mean
    totals["sharing_histogram"] = {str(degree): count for degree, count in sorted(sharing.items())}
    totals["steps"] = len(results)
    return totals


def _result_dict(result: StepResult) -> dict:
    data = _step_row(result)
    data["sharing_histogram"] = {
        str(degree): count for degree, count in sorted(result.sharing_histogram.items())
    }
    data["per_gpu"] = result.per_gpu
    data["forward_request_vectors"] = result.forward_request_vectors
    data["forward_response_vectors"] = result.forward_response_vectors
    data["backward_vectors"] = result.backward_vectors
    return data


def export_json(
    path: Path,
    results: list[StepResult],
    config: SimConfig,
    *,
    workload: dict,
    assumptions: list[str],
) -> None:
    payload = {
        "model": results[0].model,
        "config": config.to_dict(),
        "workload": workload,
        "assumptions": assumptions,
        "steps": [_result_dict(result) for result in results],
        "aggregate": aggregate(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def export_csv(directory: Path, results: list[StepResult]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    steps_path = directory / "steps.csv"
    pairs_path = directory / "pairs.csv"
    totals = aggregate(results)
    with steps_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(_step_row(result))
        aggregate_row = {key: totals.get(key, "") for key in STEP_FIELDS}
        aggregate_row["step"] = "aggregate"
        aggregate_row["model"] = results[0].model
        aggregate_row["ownership"] = results[0].ownership_scheme
        aggregate_row["origin"] = ""
        aggregate_row["cross_step_locality"] = ""
        writer.writerow(aggregate_row)

    with pairs_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "phase", "src_gpu", "dst_gpu", "vectors"))
        for result in results:
            if result.model == "torchrec":
                phases = (
                    ("id_request", "forward_request_vectors"),
                    ("remote_partial", "forward_response_vectors"),
                    ("pooled_gradient", "backward_vectors"),
                )
            elif result.model == "coherent":
                phases = (
                    ("row_fetch", "forward_request_vectors"),
                    ("row_data", "forward_response_vectors"),
                    ("row_gradient", "backward_vectors"),
                )
            else:
                phases = (
                    ("forward_request", "forward_request_vectors"),
                    ("forward_response", "forward_response_vectors"),
                    ("backward_gradient", "backward_vectors"),
                )
            for phase, attr in phases:
                matrix = getattr(result, attr)
                for src, row in enumerate(matrix):
                    for dst, vectors in enumerate(row):
                        if vectors:
                            writer.writerow((result.step_index, phase, src, dst, vectors))
    return steps_path, pairs_path


ASSUMPTIONS = [
    "One Meta batch is one global training step. Samples are split into contiguous disjoint GPU ranges.",
    "Embedding identity is (table_id, row_id). Contiguous ownership splits each table into TorchRec-style blocks of ceil(num_rows / num_gpus). The trace has no table cardinality, so num_rows is max(observed row id) + 1 unless overridden. Modulo ownership remains available.",
    "The torchrec model keeps bag boundaries. It sends every remote index, including duplicates, then counts one partial pooled vector per nonempty (sample, table, owner).",
    "torchrec forward_bytes adds a dense reduce-scatter approximation, (num_gpus - 1) * samples * tables * embedding_bytes. That is not an exact NCCL trace. logical_remote_partial_bytes is the sparse payload and is reported separately.",
    "torchrec backward_bytes is the matching all-gather of pooled gradients. It does not send one gradient per embedding row. Per-row gradient applications stay on the owner.",
    "The row-transfer model is the earlier reference: one full vector and one gradient per unique remote row, with no pooling.",
    "Both models update each distinct accessed row once, after gradient aggregation, and they share the ownership map.",
    "A future coherent cache that pools on the requesting GPU must send per-row gradients. It does not reuse the pooled all-gather unchanged.",
    "There is no remote cache in the torchrec or row-transfer models.",
    "The coherent model is a valid/invalid cache of full rows. Readers pool locally and send one gradient per remote row. Every accessed row is updated, so those copies are invalidated before the next step. cache_bytes empty means unlimited. A repeated batch is a synthetic reuse test.",
    "Repeating a batch is a synthetic sequence. It is not measured production temporal locality.",
    "embedding_dim and bytes_per_element are configuration, not fields of the Meta access file. Sum and mean pooling move the same number of vectors.",
]

"""Valid/Invalid remote embedding caches.

Readers cache full rows fetched from the owner and pool locally. The owner is
the only writer. A cached copy is valid or absent. At the end of the step
every accessed row is updated, so every copy fetched this step is invalidated
before the next step. A repeated batch therefore misses again on those rows.
Hits are duplicate touches by the same GPU before that invalidation, and only
while the row is still in a finite cache.

This is not the TorchRec pooled backward path. Gradients are one vector per
remote row. ``protocol`` selects the coherence rule. Only ``vi`` is implemented.
``cache_bytes is None`` means the cache is unlimited.
"""

from __future__ import annotations

from collections import OrderedDict

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
from workload import TrainingStep, column_span


class VICache:
    """Per-GPU valid sets and a sharer list for each row."""

    def __init__(self, num_gpus: int, capacity_rows: int | None) -> None:
        if capacity_rows is not None and capacity_rows < 1:
            raise ValueError("cache capacity must be at least one row")
        self.capacity = capacity_rows
        self.caches: list[OrderedDict[tuple[int, int], None]] = [
            OrderedDict() for _ in range(num_gpus)
        ]
        self.sharers: dict[tuple[int, int], set[int]] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def access(self, gpu: int, table_id: int, row_id: int) -> bool:
        """Touch one remote row. Return True when the access is a miss."""
        key = (table_id, row_id)
        cache = self.caches[gpu]
        if key in cache:
            cache.move_to_end(key)
            self.hits += 1
            return False
        if self.capacity is not None and len(cache) >= self.capacity:
            victim, _ = cache.popitem(last=False)
            holders = self.sharers.get(victim)
            if holders is not None:
                holders.discard(gpu)
                if not holders:
                    del self.sharers[victim]
            self.evictions += 1
        cache[key] = None
        self.sharers.setdefault(key, set()).add(gpu)
        self.misses += 1
        return True

    def invalidate(self, updated: set[tuple[int, int]] | None) -> int:
        """Drop valid copies of updated rows. ``None`` drops every resident copy."""
        if updated is None:
            count = sum(len(cache) for cache in self.caches)
            for cache in self.caches:
                cache.clear()
            self.sharers.clear()
            return count
        count = 0
        for key in list(self.sharers):
            if key not in updated:
                continue
            for gpu in self.sharers.pop(key):
                self.caches[gpu].pop(key, None)
                count += 1
        return count

    @property
    def resident(self) -> int:
        return sum(len(cache) for cache in self.caches)


def _walk_finite(
    step: TrainingStep,
    ownership: Ownership,
    cache: VICache,
    fetch: np.ndarray,
) -> tuple[int, int]:
    """Replay remote accesses in sample and bag order. Return local and remote counts."""
    batch = step.batch
    local = 0
    remote = 0
    for sample_range in step.ranges:
        column_start, column_end = column_span(batch, sample_range)
        gpu = sample_range.gpu_id
        for table_index, table_id in enumerate(batch.table_ids):
            rows = batch.table_window(table_index, column_start, column_end)
            if rows.size == 0:
                continue
            owners = ownership.owners(int(table_id), rows)
            for row_id, owner in zip(rows.tolist(), owners.tolist()):
                if int(owner) == gpu:
                    local += 1
                    continue
                remote += 1
                if cache.access(gpu, int(table_id), int(row_id)):
                    fetch[gpu, int(owner)] += 1
    return local, remote


def _account_unlimited(
    step: TrainingStep,
    ownership: Ownership,
    fetch: np.ndarray,
    gradient: np.ndarray,
) -> dict[str, int]:
    """Unlimited VI. Every fetched copy stays until the step's invalidation."""
    batch = step.batch
    local = 0
    remote = 0
    hits = 0
    misses = 0
    for sample_range in step.ranges:
        column_start, column_end = column_span(batch, sample_range)
        gpu = sample_range.gpu_id
        for table_index, table_id in enumerate(batch.table_ids):
            rows = batch.table_window(table_index, column_start, column_end)
            if rows.size == 0:
                continue
            owners = ownership.owners(int(table_id), rows)
            local += int(np.count_nonzero(owners == gpu))
            remote_rows = rows[owners != gpu]
            remote += int(remote_rows.size)
            if remote_rows.size == 0:
                continue
            unique_rows = np.unique(remote_rows)
            misses += int(unique_rows.size)
            hits += int(remote_rows.size - unique_rows.size)
            unique_owners = ownership.owners(int(table_id), unique_rows)
            fetch[gpu] += np.bincount(unique_owners, minlength=ownership.num_gpus)
            gradient[gpu] += np.bincount(unique_owners, minlength=ownership.num_gpus)
    return {"local": local, "remote": remote, "hits": hits, "misses": misses}


def _gradient_matrix(step: TrainingStep, ownership: Ownership, gradient: np.ndarray) -> int:
    batch = step.batch
    vectors = 0
    for sample_range in step.ranges:
        column_start, column_end = column_span(batch, sample_range)
        gpu = sample_range.gpu_id
        for table_index, table_id in enumerate(batch.table_ids):
            rows = batch.table_window(table_index, column_start, column_end)
            if rows.size == 0:
                continue
            owners = ownership.owners(int(table_id), rows)
            remote_rows = rows[owners != gpu]
            if remote_rows.size == 0:
                continue
            unique_owners = ownership.owners(int(table_id), np.unique(remote_rows))
            gradient[gpu] += np.bincount(unique_owners, minlength=ownership.num_gpus)
            vectors += int(unique_owners.size)
    return vectors


def simulate_coherent_step(
    step: TrainingStep,
    config: SimConfig,
    ownership: Ownership,
    cache: VICache,
    versions: dict[tuple[int, int], int] | None,
) -> StepResult:
    if config.protocol != "vi":
        raise ValueError(f"unsupported coherence protocol {config.protocol}")
    if ownership.num_gpus != config.num_gpus:
        raise ValueError("ownership GPU count does not match the config")
    fetch = np.zeros((config.num_gpus, config.num_gpus), dtype=np.int64)
    gradient = np.zeros((config.num_gpus, config.num_gpus), dtype=np.int64)
    if config.cache_rows is None:
        counts = _account_unlimited(step, ownership, fetch, gradient)
        local = counts["local"]
        remote = counts["remote"]
        # The unlimited cache keeps every miss until invalidation.
        invalidations = counts["misses"]
        hits = counts["hits"]
        misses = counts["misses"]
        evictions = 0
        cache.invalidate(None)
    else:
        local, remote = _walk_finite(step, ownership, cache, fetch)
        _gradient_matrix(step, ownership, gradient)
        updated = None
        if config.collect_rows:
            accesses = collect_unique_accesses(step.batch, step.ranges)
            preview = apply_owner_updates(
                accesses,
                config.num_gpus,
                None,
                gradients_aggregated=True,
                collect_rows=True,
            )
            updated = preview["updated_row_ids"]
        invalidations = cache.invalidate(updated)
        hits = cache.hits
        misses = cache.misses
        evictions = cache.evictions
        cache.hits = cache.misses = cache.evictions = 0

    accesses = collect_unique_accesses(step.batch, step.ranges)
    updates = apply_owner_updates(
        accesses,
        config.num_gpus,
        versions,
        gradients_aggregated=True,
        collect_rows=config.collect_rows,
    )
    miss_count = int(fetch.sum())
    if config.cache_rows is None:
        miss_count = misses
    forward_request_bytes = miss_count * config.request_bytes
    forward_response_bytes = miss_count * config.embedding_bytes
    backward_bytes = int(gradient.sum()) * config.gradient_bytes
    inv_bytes = invalidations * config.invalidation_bytes
    ack_bytes = invalidations * config.acknowledgement_bytes
    forward_bytes = forward_request_bytes + forward_response_bytes
    per_gpu = []
    for sample_range in step.ranges:
        per_gpu.append(
            {
                "gpu_id": sample_range.gpu_id,
                "sample_start": sample_range.sample_start,
                "sample_end": sample_range.sample_end,
                "num_samples": sample_range.num_samples,
                "fetches": int(fetch[sample_range.gpu_id].sum()),
                "row_gradients": int(gradient[sample_range.gpu_id].sum()),
            }
        )
    return StepResult(
        model="coherent",
        ownership_scheme=ownership.scheme,
        step_index=step.step_index,
        origin=step.origin,
        cross_step_locality=step.cross_step_locality,
        raw_accesses=accesses.raw_accesses,
        unique_gpu_row_accesses=int(sum(int(rows.size) for item in accesses.per_gpu for rows in item.unique_rows.values())),
        local_requests=local,
        remote_requests=remote,
        forward_request_bytes=forward_request_bytes,
        forward_response_bytes=forward_response_bytes,
        forward_bytes=forward_bytes,
        backward_bytes=backward_bytes,
        total_bytes=forward_bytes + backward_bytes + inv_bytes + ack_bytes,
        local_gradient_contributions=local,
        remote_gradient_vectors=0,
        updated_rows=int(updates["updated_rows"]),
        cache_hits=hits,
        logical_pair_transfers=_pair_transfers(fetch, fetch.T, gradient),
        sharing_histogram=updates["sharing_histogram"],
        mean_sharing=float(updates["mean_sharing"]),
        per_gpu=per_gpu,
        forward_request_vectors=_matrix_to_lists(fetch),
        forward_response_vectors=_matrix_to_lists(fetch.T.copy()),
        backward_vectors=_matrix_to_lists(gradient),
        versions_touched=dict(versions) if versions is not None and config.collect_rows else None,
        updated_row_ids=updates["updated_row_ids"],
        contributors=updates["contributors"],
        owner_lookups=local + remote,
        per_row_remote_gradient_vectors=int(gradient.sum()),
        row_gradient_applications=local + int(gradient.sum()),
        protocol=config.protocol,
        cache_bytes=config.cache_bytes,
        cache_misses=misses,
        cache_evictions=evictions,
        invalidations=invalidations,
        invalidation_bytes=inv_bytes,
        acknowledgement_bytes=ack_bytes,
    )


def simulate_coherent(
    steps: tuple[TrainingStep, ...] | list[TrainingStep],
    config: SimConfig,
    ownership: Ownership,
) -> tuple[list[StepResult], dict[tuple[int, int], int] | None, VICache]:
    """Run VI caches across steps. The same cache is invalidated between steps."""
    cache = VICache(config.num_gpus, config.cache_rows)
    versions: dict[tuple[int, int], int] | None = {} if config.track_versions else None
    results = [simulate_coherent_step(step, config, ownership, cache, versions) for step in steps]
    return results, versions, cache

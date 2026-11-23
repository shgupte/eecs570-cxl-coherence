#!/usr/bin/env python3
"""Sharing-aware sample placement policies, backends, and cross-step affinity state.

Policies (every GPU receives exactly B // G samples):
  contig   samples in file order, B // G per GPU
  greedy   one pass; a sample goes to the GPU that already reads most of its
           rows this step (counting rows read 2..cmax times), minus lam * load
  refine   greedy, then `passes` extra passes seeded with the previous pass
  affinity refine + bonus `beta` for rows the GPU alone updated `lag` steps ago
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import ctypes
import subprocess
import time
import numpy as np

from batch_io import LookupBatch

MAX_GPUS = 8
VALID_POLICIES = ("contig", "greedy", "refine", "affinity")

BITS = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little"
).astype(np.float64)


def popcount8(mask: np.ndarray) -> np.ndarray:
    """Count set bits in each uint8 element of mask."""
    if len(mask) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unpackbits(mask[:, None], axis=1).sum(1)


def _validate_gpu_args(B: int, G: int, cap: int) -> None:
    if not (1 <= G <= MAX_GPUS):
        raise ValueError(f"gpus must be in 1..{MAX_GPUS} (uint8 mask & placer.c limit), got {G}")
    if cap <= 0 or G * cap < B:
        raise ValueError(f"capacity G*cap ({G}*{cap}) must cover batch size B ({B})")


def assign(
    rid: np.ndarray,
    valid: np.ndarray,
    G: int,
    cap: int,
    w: np.ndarray,
    lam: float,
    prior: np.ndarray | None = None,
    bonus: np.ndarray | None = None,
    beta: float = 0.0,
    owner: np.ndarray | None = None,
    wown: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy sample placement inner loop in pure NumPy.

    rid:   (B, Tc) compact row ids
    valid: (B, Tc) bool mask
    prior/bonus: optional per-row uint8 GPU bitmasks
    owner: optional per-row int8 static owner GPU id (0..G-1)
    """
    B = int(rid.shape[0])
    _validate_gpu_args(B, G, cap)
    nrow = len(w) if len(w) else (int(rid.max()) + 1 if rid.size else 0)
    cur = np.zeros(nrow, dtype=np.uint8)
    load = np.zeros(G, dtype=np.int64)
    out = np.empty(B, dtype=np.int8)
    bit = (1 << np.arange(G)).astype(np.uint8)
    w64 = np.asarray(w, dtype=np.float64)

    for s in range(B):
        r = rid[s][valid[s]]
        if len(r):
            wr = w64[r][:, None]
            sc = (BITS[cur[r]] * wr).sum(0)[:G]
            if prior is not None:
                sc = sc + 0.5 * (BITS[prior[r]] * wr).sum(0)[:G]
            if bonus is not None:
                sc = sc + beta * BITS[bonus[r]].sum(0)[:G]
            if owner is not None and wown != 0.0:
                np.add.at(sc, owner[r], wown)
        else:
            sc = np.zeros(G, dtype=np.float64)
        sc = sc - lam * load / cap
        sc[load >= cap] = -1e9
        g = int(sc.argmax())
        out[s] = g
        load[g] += 1
        if len(r):
            cur[r] |= bit[g]
    return out, cur


def load_c(src: Path | None = None, lib_path: Path | None = None):
    """Compile placer.c next to this file (once) and return its ctypes assign()."""
    if lib_path is None:
        src = Path(__file__).with_name("placer.c") if src is None else Path(src)
        lib_path = src.with_suffix(".so")
        if not lib_path.exists() or lib_path.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(
                ["gcc", "-O2", "-shared", "-fPIC", "-o", str(lib_path), str(src)],
                check=True,
            )
    f = ctypes.CDLL(str(lib_path)).assign
    P = ctypes.c_void_p
    f.argtypes = [
        ctypes.c_int, ctypes.c_int, P, ctypes.c_int, ctypes.c_int, P, ctypes.c_double,
        P, P, ctypes.c_double, P, ctypes.c_double, P, P,
    ]
    return f


def assign_c(
    f,
    rid: np.ndarray,
    valid: np.ndarray,
    G: int,
    cap: int,
    w: np.ndarray,
    lam: float,
    prior: np.ndarray | None = None,
    bonus: np.ndarray | None = None,
    beta: float = 0.0,
    owner: np.ndarray | None = None,
    wown: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run compiled placer.c assign(); returns (gpu, cur, elapsed_seconds)."""
    B, Tc = int(rid.shape[0]), int(rid.shape[1])
    _validate_gpu_args(B, G, cap)
    nrow = len(w) if len(w) else (int(rid.max()) + 1 if rid.size else 0)
    r32 = np.ascontiguousarray(np.where(valid, rid, -1), dtype=np.int32)
    cur = np.zeros(nrow, dtype=np.uint8)
    out = np.empty(B, dtype=np.int8)
    w32 = np.ascontiguousarray(w, dtype=np.float32)
    prior_c = None if prior is None else np.ascontiguousarray(prior, dtype=np.uint8)
    bonus_c = None if bonus is None else np.ascontiguousarray(bonus, dtype=np.uint8)
    owner_c = None if owner is None else np.ascontiguousarray(owner, dtype=np.int8)
    ptr = lambda x: None if x is None else x.ctypes.data

    t0 = time.perf_counter()
    f(
        B, Tc, r32.ctypes.data, G, cap, w32.ctypes.data, float(lam),
        ptr(prior_c), ptr(bonus_c), float(beta), ptr(owner_c), float(wown),
        cur.ctypes.data, out.ctypes.data,
    )
    return out, cur, time.perf_counter() - t0


class AssignBackend(Protocol):
    """Strategy interface for single-pass sample assignment."""

    reports_timing: bool

    def assign(
        self,
        rid: np.ndarray,
        valid: np.ndarray,
        G: int,
        cap: int,
        w: np.ndarray,
        lam: float,
        *,
        prior: np.ndarray | None = None,
        bonus: np.ndarray | None = None,
        beta: float = 0.0,
        owner: np.ndarray | None = None,
        wown: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        ...


class PythonBackend:
    """Pure-NumPy assignment backend."""

    reports_timing = False

    def assign(
        self,
        rid: np.ndarray,
        valid: np.ndarray,
        G: int,
        cap: int,
        w: np.ndarray,
        lam: float,
        *,
        prior: np.ndarray | None = None,
        bonus: np.ndarray | None = None,
        beta: float = 0.0,
        owner: np.ndarray | None = None,
        wown: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        gpu, cur = assign(
            rid, valid, G, cap, w, lam,
            prior=prior, bonus=bonus, beta=beta, owner=owner, wown=wown,
        )
        return gpu, cur, None


class CBackend:
    """Compiled placer.c assignment backend."""

    reports_timing = True

    def __init__(self, fn=None, *, src: Path | None = None, lib_path: Path | None = None) -> None:
        self._fn = load_c(src=src, lib_path=lib_path) if fn is None else fn

    def assign(
        self,
        rid: np.ndarray,
        valid: np.ndarray,
        G: int,
        cap: int,
        w: np.ndarray,
        lam: float,
        *,
        prior: np.ndarray | None = None,
        bonus: np.ndarray | None = None,
        beta: float = 0.0,
        owner: np.ndarray | None = None,
        wown: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float | None]:
        return assign_c(
            self._fn, rid, valid, G, cap, w, lam,
            prior=prior, bonus=bonus, beta=beta, owner=owner, wown=wown,
        )


class SoleUpdaterTracker:
    """Tracks which single GPU last updated each row, with configurable lag."""

    def __init__(self, lag: int = 1) -> None:
        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")
        self.lag = lag
        self.sole_k = np.array([], dtype=np.int64)
        self.sole_m = np.array([], dtype=np.uint8)
        self.hist: list[tuple[np.ndarray, np.ndarray]] = []

    def bonus_for(self, keys: np.ndarray) -> np.ndarray | None:
        """Return lagged uint8 sole-updater GPU bitmask aligned with keys, or None."""
        if len(self.hist) < self.lag:
            return None
        bk, bm = self.hist[-self.lag]
        if len(bk) == 0 or len(keys) == 0:
            return None
        ix = np.searchsorted(bk, keys)
        ix[ix >= len(bk)] = 0
        return np.where(bk[ix] == keys, bm[ix], 0).astype(np.uint8)

    def update(self, keys: np.ndarray, mask: np.ndarray) -> None:
        """Update sole-updater state with this step's touched keys and GPU masks."""
        if len(keys):
            n = popcount8(mask)
            keep = ~np.isin(self.sole_k, keys)
            nk = np.concatenate([self.sole_k[keep], keys])
            nm = np.concatenate([self.sole_m[keep], np.where(n == 1, mask, 0).astype(np.uint8)])
            order = np.argsort(nk, kind="stable")
            self.sole_k, self.sole_m = nk[order], nm[order]
        self.hist.append((self.sole_k, self.sole_m))
        self.hist = self.hist[-self.lag:]


@dataclass(frozen=True)
class PlacementConfig:
    """Configuration for SamplePlacer."""
    policy: str
    gpus: int = 8
    passes: int = 2
    beta: float = 2.0
    cmax: int = 64
    lam: float = 1.0
    lag: int = 1
    owner_weight: float = 0.0
    fast: bool = False

    def __post_init__(self) -> None:
        if self.policy not in VALID_POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; expected one of {VALID_POLICIES}")
        if not (1 <= self.gpus <= MAX_GPUS):
            raise ValueError(f"gpus must be in 1..{MAX_GPUS}, got {self.gpus}")
        if self.passes < 0:
            raise ValueError(f"passes must be >= 0, got {self.passes}")
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")


@dataclass(frozen=True)
class PlacementResult:
    """Per-step placement output."""
    keys: np.ndarray
    counts: np.ndarray
    mask: np.ndarray
    gpu: np.ndarray
    raw: int
    elapsed_sec: float | None = None

    @property
    def unique_rows(self) -> int:
        return len(self.keys)

    @property
    def pairs(self) -> int:
        return int(popcount8(self.mask).sum())

    @property
    def single_frac(self) -> float:
        if len(self.mask) == 0:
            return 0.0
        return float((popcount8(self.mask) == 1).mean())


def compact_batch(keys: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map valid keys to dense 0..U-1 row ids; returns (unique_keys, rid, counts)."""
    B, Tc = keys.shape
    if not valid.any():
        return np.zeros(0, dtype=np.int64), np.zeros((B, Tc), dtype=np.int64), np.zeros(0, dtype=np.int64)
    uniq, inv, counts = np.unique(keys[valid], return_inverse=True, return_counts=True)
    rid = np.zeros((B, Tc), dtype=np.int64)
    rid[valid] = inv
    return uniq, rid, counts


def compute_gpu_mask(rid: np.ndarray, valid: np.ndarray, gpu: np.ndarray, num_keys: int) -> np.ndarray:
    """Compute per-unique-key uint8 bitmask of GPUs reading each key."""
    mask = np.zeros(num_keys, dtype=np.uint8)
    if num_keys and valid.any():
        Tc = rid.shape[1]
        gpu_bits = (1 << gpu.astype(np.uint8))[:, None].repeat(Tc, 1)
        np.bitwise_or.at(mask, rid[valid], gpu_bits[valid])
    return mask


def compute_static_owners(
    keys: np.ndarray,
    tlen: np.ndarray,
    gpus: int,
    row_shift: int,
) -> np.ndarray:
    """Compute TorchRec-style row-wise block static owner GPU id for each packed key."""
    if len(keys) == 0:
        return np.zeros(0, dtype=np.int8)
    kt = keys >> row_shift
    kr = keys & ((1 << row_shift) - 1)
    lens = tlen[kt]
    if np.any(lens <= 0):
        raise ValueError("tlen contains non-positive table length for a referenced table")
    block = -(-lens // gpus)
    own = (kr // block).astype(np.int8)
    if int(own.min()) < 0 or int(own.max()) >= gpus:
        raise ValueError(
            f"computed owner ids out of range [0, {gpus - 1}]; check tlen against row ids"
        )
    return own


class SamplePlacer:
    """Stateful per-step sample placer implementing contig/greedy/refine/affinity."""

    def __init__(
        self,
        config: PlacementConfig,
        *,
        tlen: np.ndarray | None = None,
        backend: AssignBackend | None = None,
    ) -> None:
        if config.owner_weight != 0.0 and tlen is None:
            raise ValueError("tlen is required when owner_weight != 0")
        self.config = config
        self.tlen = None if tlen is None else np.asarray(tlen, dtype=np.int64)
        self.tracker = SoleUpdaterTracker(lag=config.lag)
        if backend is not None:
            self.backend: AssignBackend = backend
        elif config.fast:
            self.backend = CBackend()
        else:
            self.backend = PythonBackend()

    def place(self, batch: LookupBatch) -> PlacementResult:
        """Assign each sample in `batch` to a GPU and update cross-step affinity state."""
        cfg = self.config
        B = batch.batch_size
        G = cfg.gpus
        if B % G != 0:
            raise ValueError(f"batch size ({B}) must be divisible by gpus ({G})")
        cap = B // G

        keys, rid, counts = compact_batch(batch.keys, batch.valid)
        elapsed: float | None = None

        if cfg.policy == "contig" or len(keys) == 0:
            gpu = (np.arange(B) // cap).astype(np.int8)
            mask = compute_gpu_mask(rid, batch.valid, gpu, len(keys))
        else:
            bonus = self.tracker.bonus_for(keys) if cfg.policy == "affinity" else None
            w = ((counts >= 2) & (counts <= cfg.cmax)).astype(np.float32)
            own = (
                compute_static_owners(keys, self.tlen, G, batch.row_shift)
                if cfg.owner_weight != 0.0 and self.tlen is not None
                else None
            )
            gpu, cur, dt = self.backend.assign(
                rid, batch.valid, G, cap, w, cfg.lam,
                bonus=bonus, beta=cfg.beta, owner=own, wown=cfg.owner_weight,
            )
            extra_passes = cfg.passes if cfg.policy != "greedy" else 0
            for _ in range(extra_passes):
                gpu, cur, d2 = self.backend.assign(
                    rid, batch.valid, G, cap, w, cfg.lam,
                    prior=cur, bonus=bonus, beta=cfg.beta, owner=own, wown=cfg.owner_weight,
                )
                if dt is not None and d2 is not None:
                    dt += d2
            elapsed = dt
            mask = cur

        self.tracker.update(keys, mask)
        return PlacementResult(
            keys=keys,
            counts=counts,
            mask=mask,
            gpu=gpu,
            raw=batch.raw_lookups,
            elapsed_sec=elapsed,
        )

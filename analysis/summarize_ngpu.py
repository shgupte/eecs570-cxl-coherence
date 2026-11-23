#!/usr/bin/env python3
"""Summarize gem5 runs of gen_ngpu_traces.py traces (4 barriers per step).

usage: summarize_ngpu.py <results_root> <prefix> <log_dir> [--steps 16] [--first-step 4]

Run dirs: <prefix>-<trace tag>--<config tag>-<jobid>. Step/phase times (µs)
are averaged over steady-state steps first_step..steps-1; a warm-up step, if
present, is detected from the barrier count and skipped.
  read_us  : read phase (forward pass, critical path)
  write_us : owner update phase
  sync_us  : sync1 + sync2 (write-backs, lists, pushes)
  step_us  : whole step
"""
import argparse, re
from pathlib import Path

STATS = {
    "inv": r"L1Cache_Controller\.Inv::total\s+(\S+)",
    "ld_hit": r"RequestType\.LD\.hit_latency_hist_seqr::samples\s+(\S+)",
    "ld_miss": r"RequestType\.LD\.miss_latency_hist_seqr::samples\s+(\S+)",
    "mem_reads": r"mem_ctrl0\.readReqs\s+(\S+)",
}
NAME = re.compile(r"^(?P<prefix>[^-]+)-(?P<tag>.+)--(?P<cfg>.+)-(?P<job>\d+)$")


def phases(log, steps, first):
    rel ={int(b): int(t) for b, t in re.findall(r"generator0: barrier (\d+) release tick (\d+)", log)}
    n = len(rel)
    off = n - 4 * steps
    if off < 0 or off % 4:
        return None
    rel = {b - off: t for b, t in rel.items() if b >= off}
    end = lambda s: rel[4 * s + 3]
    ks = range(max(first, 1), steps)
    avg = lambda f: sum(f(s) for s in ks) / len(ks) / 1e6
    return dict(read_us=avg(lambda s: rel[4 * s] - end(s - 1)),
                write_us=avg(lambda s: rel[4 * s + 1] - rel[4 * s]),
                sync_us=avg(lambda s: rel[4 * s + 3] - rel[4 * s + 1]),
                step_us=avg(lambda s: end(s) - end(s - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("prefix")
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--first-step", type=int, default=4)
    a = ap.parse_args()
    keys = ["read_us", "write_us", "sync_us", "step_us"] + list(STATS)
    print("tag,config," + ",".join(keys))
    for d in sorted(a.root.glob(a.prefix + "-*")):
        m = NAME.match(d.name)
        if not m or m["prefix"] != a.prefix:
            continue
        log = a.logdir / f"cxl-trace-{m['job']}.log"
        if not (d / "stats.txt").exists() or not log.exists():
            continue
        r = phases(log.read_text(), a.steps, a.first_step)
        if r is None:
            continue
        text = (d / "stats.txt").read_text()
        for k, p in STATS.items():
            x = re.search(p, text, re.M)
            r[k] = float(x.group(1)) if x else 0.0
        print(f"{m['tag']},{m['cfg']}," + ",".join(
            f"{r[k]:.2f}" if k.endswith("_us") else f"{r[k]:.0f}" for k in keys))


if __name__ == "__main__":
    main()

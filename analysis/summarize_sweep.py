#!/usr/bin/env python3
"""Summarize gem5 runs of gen_multistep_traces.py traces.

usage: summarize_sweep.py <results_root> <name_prefix> <log_dir> [--first-step 4]

Run directories must be named <prefix>-<mode tag>-<l1>-l2lat<cycles>-<jobid>
(what run_coherence_sweep.sh produces); the job log
<log_dir>/cxl-trace-<jobid>.log holds the barrier release ticks.

Every step has 3 barriers (read, write, push). Runs made with --warm have one
extra warm-up step (barriers 0-2) that is skipped. Times are averaged over
steady-state steps first_step..last (default 4..15):
  read_us : read phase = forward-pass critical path
  step_us : whole step (read + write + push phases)
"""
import argparse, re
from pathlib import Path

STATS = {
    "inv": r"L1Cache_Controller\.Inv::total\s+(\S+)",
    "getx": r"L2Cache_Controller\.L1_GETX\s+(\S+)",
    "ld_hit": r"RequestType\.LD\.hit_latency_hist_seqr::samples\s+(\S+)",
    "ld_miss": r"RequestType\.LD\.miss_latency_hist_seqr::samples\s+(\S+)",
    "mem_reads": r"mem_ctrl0\.readReqs\s+(\S+)",
}
NAME = re.compile(r"-(?P<tag>(?:shared|uncached|swflush|sharded|push)(?:-s\d+)?(?:-a\d+)?(?:-w)?)"
                  r"-(?P<l1>\w+)-l2lat(?P<lat>\d+)-(?P<job>\d+)$")


def step_times(log, warm, first):
    off = 3 if warm else 0
    rel = {int(b) - off: int(t) for b, t in
           re.findall(r"generator0: barrier (\d+) release tick (\d+)", log) if int(b) >= off}
    ends = [rel[b] for b in sorted(rel) if b % 3 == 2]
    ks = range(max(first, 1), len(ends))
    if not ks:
        return float("nan"), float("nan")
    rd = [rel[3 * k] - ends[k - 1] for k in ks]
    st = [ends[k] - ends[k - 1] for k in ks]
    return sum(rd) / len(rd) / 1e6, sum(st) / len(st) / 1e6  # ticks are ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("prefix")
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--first-step", type=int, default=4)
    a = ap.parse_args()
    keys = ["read_us", "step_us"] + list(STATS)
    print("mode,l1,l2lat," + ",".join(keys))
    for d in sorted(a.root.glob(a.prefix + "-*")):
        m = NAME.search(d.name)
        if not m:
            continue
        log = a.logdir / f"cxl-trace-{m['job']}.log"
        if not (d / "stats.txt").exists() or not log.exists():
            continue
        text = (d / "stats.txt").read_text()
        r = {}
        r["read_us"], r["step_us"] = step_times(log.read_text(), m["tag"].endswith("-w"), a.first_step)
        for k, p in STATS.items():
            x = re.search(p, text, re.M)
            r[k] = float(x.group(1)) if x else 0.0
        print(f"{m['tag']},{m['l1']},{m['lat']},"
              + ",".join(f"{r[k]:.1f}" if k.endswith("_us") else f"{r[k]:.0f}" for k in keys))


if __name__ == "__main__":
    main()

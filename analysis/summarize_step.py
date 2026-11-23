#!/usr/bin/env python3
"""Summarize gem5 runs of gen_step_traces.py traces.

usage: summarize_step.py <results_root> <prefix> <log_dir> [--steps 12] [--first-step 2]

Phase times in µs, averaged over steps first_step..steps-1 (the 4 warm-up
barriers are skipped):
  rp_us  reads that lookahead could prefetch     rh_us  reads that must wait
  w1_us  gradient publish                        w2_us  combine + row update
  step_us = all four;  crit_us = rh + w1 + w2 (step time if rp is hidden
  behind GPU compute by lookahead prefetch)
"""
import argparse, re
from pathlib import Path

STATS = {
    "inv": r"L1Cache_Controller\.Inv::total\s+(\S+)",
    "ld_hit": r"RequestType\.LD\.hit_latency_hist_seqr::samples\s+(\S+)",
    "ld_miss": r"RequestType\.LD\.miss_latency_hist_seqr::samples\s+(\S+)",
    "st_hit": r"RequestType\.ST\.hit_latency_hist_seqr::samples\s+(\S+)",
    "st_miss": r"RequestType\.ST\.miss_latency_hist_seqr::samples\s+(\S+)",
    "fwd_getx": r"L1Cache_Controller\.Fwd_GETX::total\s+(\S+)",
    "fwd_gets": r"L1Cache_Controller\.Fwd_GETS::total\s+(\S+)",
    "mem_reads": r"mem_ctrl0\.readReqs\s+(\S+)",
}
NAME = re.compile(r"^(?P<prefix>[^-]+)-(?P<tag>.+)--(?P<cfg>.+)-(?P<job>\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("prefix")
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--first-step", type=int, default=2)
    a = ap.parse_args()
    ph = ["rp_us", "rh_us", "w1_us", "w2_us"]
    keys = ph + ["step_us", "crit_us"] + list(STATS)
    print("tag,config," + ",".join(keys))
    for d in sorted(a.root.glob(a.prefix + "-*")):
        m = NAME.match(d.name)
        log = a.logdir / f"cxl-trace-{m['job']}.log" if m else None
        if not m or m["prefix"] != a.prefix or not (d / "stats.txt").exists() or not log.exists():
            continue
        rel = {int(b): int(t) for b, t in re.findall(r"generator0: barrier (\d+) release tick (\d+)", log.read_text())}
        if len(rel) != 4 * (a.steps + 1):
            continue
        ks = range(a.first_step, a.steps)
        r = {p: sum(rel[4 * (s + 1) + i] - rel[4 * (s + 1) + i - 1] for s in ks) / len(ks) / 1e6
             for i, p in enumerate(ph)}
        r["step_us"] = sum(r[p] for p in ph)
        r["crit_us"] = r["step_us"] - r["rp_us"]
        text = (d / "stats.txt").read_text()
        for k, p in STATS.items():
            x = re.search(p, text, re.M)
            r[k] = float(x.group(1)) if x else 0.0
        print(f"{m['tag']},{m['cfg']}," + ",".join(f"{r[k]:.2f}" if k.endswith("_us") else f"{r[k]:.0f}" for k in keys))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize hot-row sweep runs: one row per run, plus shared-minus-sharded."""
import re, sys
from pathlib import Path

PAT = {
    "simTicks": r"^simTicks\s+(\S+)",
    "inv": r"L1Cache_Controller\.Inv::total\s+(\S+)",
    "upgrades": r"L2Cache_Controller\.L1_UPGRADE\s+(\S+)",
    "getx": r"L2Cache_Controller\.L1_GETX\s+(\S+)",
    "l1_repl": r"L1Cache_Controller\.L1_Replacement::total\s+(\S+)",
    "ld_miss": r"RequestType\.LD\.miss_latency_hist_seqr::samples\s+(\S+)",
    "ld_hit": r"RequestType\.LD\.hit_latency_hist_seqr::samples\s+(\S+)",
    "rd_lat0": r"cxl_trace_generator0\.avgReadLatency\s+(\S+)",
    "rd_lat1": r"cxl_trace_generator1\.avgReadLatency\s+(\S+)",
    "mem_reads": r"mem_ctrl0\.readReqs\s+(\S+)",
}


def grab(text):
    out = {}
    for k, p in PAT.items():
        m = re.search(p, text, re.M)
        out[k] = float(m.group(1)) if m else float("nan")
    # coherence message bytes: sum of all msg_byte counters on control vnets
    out["msg_bytes"] = sum(float(v) for v in re.findall(
        r"network\.msg_byte\.\S+\s+(\d+)", text))
    return out


root = Path(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else "hot"
logdir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
rows = {}
for d in sorted(root.glob(prefix + "-*")):
    m = re.match(prefix + r"-(shared|sharded)-(\w+)-(?:l2)?lat(\d+)-(\d+)", d.name)
    if not m or not (d / "stats.txt").exists():
        continue
    r = grab((d / "stats.txt").read_text())
    r["step_ns"] = r["total_ns"] = float("nan")
    if logdir:
        log = (logdir / f"cxl-trace-{m.group(4)}.log").read_text()
        rel = {int(b): int(t) for b, t in re.findall(
            r"generator0: barrier (\d+) release tick (\d+)", log)}
        ends = [rel[b] for b in sorted(rel) if b % 2 == 1]  # end of each step
        if len(ends) > 1:
            r["total_ns"] = ends[-1] / 1000
            r["step_ns"] = (ends[-1] - ends[0]) / (len(ends) - 1) / 1000
    rows[m.groups()[:3]] = r
keys = ["total_ns", "step_ns", "simTicks", "inv", "upgrades", "getx", "l1_repl", "ld_hit", "ld_miss",
        "rd_lat0", "rd_lat1", "mem_reads", "msg_bytes"]
print("mode,l1,lat," + ",".join(keys))
for (mode, l1, lat), r in sorted(rows.items()):
    print(f"{mode},{l1},{lat}," + ",".join(f"{r[k]:.0f}" for k in keys))
print("\nshared-minus-sharded")
print("l1,lat," + ",".join(keys))
for (mode, l1, lat), r in sorted(rows.items()):
    if mode == "shared" and ("sharded", l1, lat) in rows:
        s = rows[("sharded", l1, lat)]
        print(f"{l1},{lat}," + ",".join(f"{r[k]-s[k]:.0f}" for k in keys))

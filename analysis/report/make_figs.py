#!/usr/bin/env python3
"""Figures of REPORT.md. Numbers: results_gem5_port_{matrix,bulk,bulk2,fix}.csv
(gem5) and logs/linkmodel-*.log (analysis/linkmodel.py, full batch)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#a3a29a"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def fig_links():
    rows = [("NVLink (300 GB/s per GPU)", [("Sharded, deduplicated", 114), ("Sharded + placement", 91),
                                             ("Sharded + placement + row migration", 48)], BLUE),
            ("Network card (25 GB/s per GPU)", [("Sharded, deduplicated", 1367), ("Sharded + placement", 1088),
                                                  ("Sharded + placement + row migration", 578)], BLUE),
            ("CXL pool, 8 pool ports (63 GB/s per GPU)", [("CLAY-style baseline", 851), ("Software scheduled sharing", 464),
                                                           ("MESI + deltas", 381)], ORANGE),
            ("CXL pool, 1 pool port", [("CLAY-style baseline", 6556), ("Software scheduled sharing", 3312),
                                        ("MESI + deltas", 2644)], ORANGE)]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    y, ticks, labels = 0, [], []
    for group, items, col in rows:
        ax.text(30, y - 0.15, group, color=MUTED, fontsize=8.5, fontweight="bold", va="center")
        y += 1
        for name, v in items:
            ax.plot([30, v], [y, y], color=GRID, lw=1, zorder=1)
            ax.scatter([v], [y], s=42, color=col, zorder=3, edgecolor="white", linewidth=1)
            ax.text(v * 1.13, y, f"{v:,}", va="center", color=INK, fontsize=8.5)
            ticks.append(y); labels.append(name); y += 1
        y += 0.5
    ax.set_xscale("log"); ax.set_xlim(30, 14000); ax.set_ylim(y - 0.6, -0.8)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Embedding traffic time per step (µs, log scale; lower is better)")
    ax.scatter([], [], s=42, color=BLUE, label="Sharded all-to-all"); ax.scatter([], [], s=42, color=ORANGE, label="CXL pool")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig_links.png"); plt.close(fig)


def fig_gem5():
    names = ["Sharded, deduplicated", "Sharded + placement", "Sharded + placement + migration",
             "Pool 8 ports, CLAY-style baseline", "Pool 8 ports, software sharing", "Pool 8 ports, MESI + deltas",
             "Pool 1 port, MESI + deltas", "Pool, equal link budget, MESI + deltas"]
    w32 = [44.6, 33.7, 17.0, 75.8, 23.6, 23.7, 65.0, None]
    bulk = [27.6, 20.3, 12.2, 35.1, 11.1, 11.4, 63.7, 20.8]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    h = 0.36
    for i, (a, b) in enumerate(zip(w32, bulk)):
        if a is not None:
            ax.barh(i - h / 2 - 0.02, a, h, color=GRAY); ax.text(a + 1, i - h / 2 - 0.02, f"{a:.1f}", va="center", fontsize=8, color=INK)
        ax.barh(i + h / 2 + 0.02, b, h, color=VIOLET); ax.text(b + 1, i + h / 2 + 0.02, f"{b:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.set_xlim(0, 86)
    ax.set_xlabel("gem5 step time (µs, 1/64 row sample; lower is better)")
    ax.barh([0], [0], color=GRAY, label="32 requests outstanding"); ax.barh([0], [0], color=VIOLET, label="Bulk transfers")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig("fig_gem5.png"); plt.close(fig)


def fig_ports():
    p = [1, 2, 4, 8]
    series = [("CLAY-style baseline", [6556, 3282, 1652, 851], YELLOW), ("Software scheduled sharing", [3312, 1666, 853, 464], AQUA),
              ("MESI + deltas", [2644, 1333, 687, 381], ORANGE)]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for name, v, c in series:
        ax.plot(p, v, color=c, lw=2, marker="o", ms=6, mec="white", mew=1, label=name)
        ax.text(8.25, v[-1], f"{v[-1]}", va="center", fontsize=8.5, color=INK)
    for name, v in [("Sharded dedup, 25 GB/s network card: 1,367", 1367), ("Sharded dedup, NVLink: 114", 114)]:
        ax.axhline(v, color=BLUE, lw=1.2, ls=(0, (4, 3)))
        ax.text(1.0, v * 0.93, name, color=MUTED, fontsize=8.5, va="top")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(p); ax.set_xticklabels(p)
    ax.set_xlim(0.92, 9.6); ax.set_ylim(70, 9000); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xlabel("Pool ports (each 63 GB/s)"); ax.set_ylabel("Time per step (µs, log scale)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig("fig_ports.png"); plt.close(fig)


fig_links(); fig_gem5(); fig_ports()

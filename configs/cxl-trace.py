#!/usr/bin/env python3
"""Replay a CSV packet trace through SimCXL's device-side Ruby cache."""

import argparse
from pathlib import Path

from m5.objects import SimpleExtLink, SimpleIntLink, TrafficGen

from gem5.coherence_protocol import CoherenceProtocol
from gem5.components.boards.test_board import TestBoard
from gem5.components.cachehierarchies.ruby.cxl_mesi_two_level_cache_hierarchy import (
    CXLMESITwoLevelCacheHierarchy,
)
from gem5.components.memory.single_channel import DIMM_DDR5_4400
from gem5.components.processors.traffic_generator import TrafficGenerator
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-config", type=Path, required=True)
    parser.add_argument(
        "--device-config",
        type=Path,
        action="append",
        required=True,
        help="TrafficGen config for one accelerator. Repeat for more devices.",
    )
    parser.add_argument("--max-outstanding", type=int, default=32)
    parser.add_argument("--l1d-size", default="128KiB")
    parser.add_argument("--l2-size", default="512KiB")
    parser.add_argument(
        "--l2-latency-cycles",
        type=int,
        default=256,
        help="Cycles on each L2 request/response/to-L1 hop. SimCXL uses "
        "this hop for the CXL link; 256 is the original value.",
    )
    parser.add_argument(
        "--cxl-extra-latency-ns",
        type=int,
        default=0,
        help="Emulated CXL.mem link+controller latency added to every "
        "memory-controller access (embeddings live in this memory). "
        "0 keeps the original host-DDR5 baseline.",
    )
    parser.add_argument(
        "--link-bw",
        type=int,
        default=0,
        help="Bytes/cycle of every Ruby network link (SimpleNetwork "
        "bandwidth_factor, default 16 = 38.4 GB/s at 2.4 GHz). All pool "
        "traffic crosses the L2's single link, so this sets the pool port "
        "bandwidth. 0 keeps the default.",
    )
    parser.add_argument(
        "--pool-mediated",
        action="store_true",
        help="Ownership transfers go through the pool (L2) as in CXL "
        "back-invalidate, not cache to cache.",
    )
    args = parser.parse_args()
    for path in (args.cpu_config, *args.device_config):
        if not path.is_file():
            parser.error(f"not a file: {path}")
    return args


def main() -> None:
    args = parse_args()
    requires(coherence_protocol_required=CoherenceProtocol.CXL_MESI_TWO_LEVEL)

    if args.link_bw:
        SimpleExtLink.bandwidth_factor = args.link_bw
        SimpleIntLink.bandwidth_factor = args.link_bw
    cache_hierarchy = CXLMESITwoLevelCacheHierarchy(
        l1d_size=args.l1d_size,
        l1d_assoc=8,
        l1i_size="32KiB",
        l1i_assoc=8,
        l2_size=args.l2_size,
        l2_assoc=16,
        num_l2_banks=1,
        l2_latency=args.l2_latency_cycles,
        # passed only when set, so binaries built before the option still run
        **({"pool_mediated": True} if args.pool_mediated else {}),
    )
    memory = DIMM_DDR5_4400(size="3GiB")
    if args.cxl_extra_latency_ns:
        for ctrl in memory.get_memory_controllers():
            ctrl.static_frontend_latency = f"{10 + args.cxl_extra_latency_ns}ns"
    board = TestBoard(
        clk_freq="2.4GHz",
        generator=TrafficGenerator([str(args.cpu_config.resolve())]),
        memory=memory,
        cache_hierarchy=cache_hierarchy,
    )
    generators = []
    for index, config in enumerate(args.device_config):
        generator = TrafficGen(
            config_file=str(config.resolve()),
            max_outstanding_reqs=args.max_outstanding,
            progress_check="10ms",
        )
        setattr(board, f"cxl_trace_generator{index}", generator)
        generators.append(generator)
    # Bypass SimObject parenting; each generator is already a board child.
    object.__setattr__(board, "cxl_trace_generators", generators)

    Simulator(board=board).run()


if __name__ == "__m5_main__":
    main()
elif __name__ == "__main__":
    raise SystemExit("Run this script with gem5.opt, not plain Python.")

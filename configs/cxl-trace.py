#!/usr/bin/env python3
"""Replay a CSV packet trace through SimCXL's device-side Ruby cache."""

import argparse
from pathlib import Path

from m5.objects import (
    SimpleExtLink,
    SimpleIntLink,
    SimpleNetwork,
    Switch,
    TrafficGen,
)

import gem5.components.cachehierarchies.ruby.cxl_mesi_two_level_cache_hierarchy as hier_mod

from gem5.coherence_protocol import CoherenceProtocol
from gem5.components.boards.test_board import TestBoard
from gem5.components.cachehierarchies.ruby.cxl_mesi_two_level_cache_hierarchy import (
    CXLMESITwoLevelCacheHierarchy,
)
from gem5.components.memory.single_channel import DIMM_DDR5_4400
from gem5.components.processors.traffic_generator import TrafficGenerator
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires


FAST = 4096  # bytes/cycle of links that are not a modeled port
STAR = {}  # StarNet settings (a SimObject class cannot hold them)


class StarNet(SimpleNetwork):
    """Explicit port model (SimpleNetwork throttles only switch outputs, so
    in the default SimplePt2Pt every GPU has its own 16 B/cycle path FROM the
    pool and only traffic TO the pool shares one link).

    Every GPU has one port to a hub switch (gpu_bw bytes/cycle each way).
      pool    : the L2 (the pool) hangs off the hub through pool_ports ports
                (pool_ports * gpu_bw each way, shared by all GPUs).
      sharded : there are n_gpu L2s; L2 i is GPU i's shard and sits behind
                GPU i's port, so a remote row crosses the reader's and the
                owner's port, as in all-to-all over a switch.
    Everything else (host core, warm-up engines, memory) gets a fast link.
    Controller order from the hierarchy: host L1, GPU L1s, warm-up L1s, L2s,
    directories, DMA.
    """

    def __init__(self, ruby_system):
        super().__init__()
        self.netifs = []
        self.ruby_system = ruby_system

    def connectControllers(self, controllers):
        G, gpu_bw = STAR["n_gpu"], STAR["gpu_bw"]
        sharded, pool_ports = STAR["sharded"], STAR["pool_ports"]
        l2s = [c for c in controllers if type(c).__name__ == "L2Cache"]
        assert len(l2s) == (G if sharded else 1)
        hub = Switch(router_id=0)
        routers, ext, ints = [hub], [], []

        def router():
            routers.append(Switch(router_id=len(routers)))
            return routers[-1]

        def link(a, b, bw):
            for s, d in ((a, b), (b, a)):
                ints.append(
                    SimpleIntLink(
                        link_id=len(ints),
                        src_node=s,
                        dst_node=d,
                        bandwidth_factor=bw,
                    )
                )

        def attach(c, r):
            ext.append(
                SimpleExtLink(
                    link_id=len(ext),
                    ext_node=c,
                    int_node=r,
                    bandwidth_factor=FAST,
                )
            )

        gpu_r = [router() for _ in range(G)]
        for r in gpu_r:
            link(r, hub, gpu_bw)
        pool_r = None
        if not sharded:
            pool_r = router()
            link(pool_r, hub, gpu_bw * pool_ports)
        l2i = 0
        for i, c in enumerate(controllers):
            if 1 <= i <= G:
                attach(c, gpu_r[i - 1])
            elif c in l2s:
                attach(c, gpu_r[l2i] if sharded else pool_r)
                l2i += 1
            elif type(c).__name__ == "Directory" and not sharded:
                attach(c, pool_r)
            else:
                r = router()
                link(r, hub, FAST)
                attach(c, r)
        self.routers, self.ext_links, self.int_links = routers, ext, ints


class Hierarchy(CXLMESITwoLevelCacheHierarchy):
    """Commutative updates only for the GPUs (L1 controllers 1..n_gpu): the
    host core and the warm-up engines keep plain MESI stores."""

    def incorporate_cache(self, board):
        super().incorporate_cache(board)
        for i, c in enumerate(self._l1_controllers):
            if not 1 <= i <= STAR["n_gpu"]:
                c.commutative = False


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
    parser.add_argument(
        "--pool-mediated-gets",
        action="store_true",
        help="Reads of a line another GPU owns also go through the pool: the "
        "owner writes the data back and keeps a shared copy (CXL BISnpData).",
    )
    parser.add_argument(
        "--commutative",
        action="store_true",
        help="COUP-style: a store to a line the GPU does not own is sent to "
        "the pool as an update and merged there.",
    )
    parser.add_argument(
        "--topology",
        choices=["pt2pt", "pool", "sharded"],
        default="pt2pt",
        help="pt2pt: SimCXL default. pool / sharded: explicit ports (StarNet).",
    )
    parser.add_argument("--gpu-bw", type=int, default=16,
                        help="StarNet: bytes/cycle of each GPU port")
    parser.add_argument("--pool-ports", type=int, default=1,
                        help="StarNet pool: pool ports (each gpu-bw)")
    parser.add_argument("--gpus", type=int, default=8)
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
    if args.topology != "pt2pt":
        STAR.update(
            n_gpu=args.gpus,
            gpu_bw=args.gpu_bw,
            pool_ports=args.pool_ports,
            sharded=args.topology == "sharded",
        )
        hier_mod.SimplePt2Pt = StarNet
    STAR["n_gpu"] = args.gpus
    cache_hierarchy = Hierarchy(
        l1d_size=args.l1d_size,
        l1d_assoc=8,
        l1i_size="32KiB",
        l1i_assoc=8,
        l2_size=args.l2_size,
        l2_assoc=16,
        num_l2_banks=args.gpus if args.topology == "sharded" else 1,
        l2_latency=args.l2_latency_cycles,
        # passed only when set, so binaries built before the option still run
        **({"pool_mediated": True} if args.pool_mediated else {}),
        **({"pool_mediated_gets": True} if args.pool_mediated_gets else {}),
        **({"commutative": True} if args.commutative else {}),
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

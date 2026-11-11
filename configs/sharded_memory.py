"""Eight accelerators, each with a 2 GB DDR5 shard.

Set TRACE_DIR to a directory of accel0.cfg .. accel7.cfg produced by
scripts/generate_sharded_lookups.py. Without it, accelerator 0 replays the
two-read routing check and the others stay idle.
"""

import os

import m5

from m5.objects import (
    Bridge,
    NoncoherentXBar,
    AddrRange,
    DDR5_4400_4x8,
    MemCtrl,
    Root,
    SrcClockDomain,
    System,
    TrafficGen,
    VoltageDomain,
)

system = System() # system is the top level container which holds all the hardware
 
system.clk_domain = SrcClockDomain( # set the clk domain to be a global 2.4 GHz clock and all children will use this
    clock="2.4GHz",
    voltage_domain=VoltageDomain(),
)

system.mem_mode = "timing" #timing mode uses actual "cycle accuracy" rather than atomic which is instant
system.cache_line_size = 64 # set the cache line size

NUM_ACCELERATORS = 8 
SHARD_BYTES = 2 * 1024**3 

system.mem_ranges = [
    AddrRange(start=i * SHARD_BYTES, size=SHARD_BYTES) # spliots a 16 GB address space into 8 contiguous 2 GB chunks
    for i in range(NUM_ACCELERATORS) # gives one 2GB chunk to each accelerator
]

system.mem_ctrls = [ # pair each addr range with its own memory controller and DRAM 
    MemCtrl(
        dram=DDR5_4400_4x8(range=system.mem_ranges[i])
    )
    for i in range(NUM_ACCELERATORS)
]

system.local_xbars = [ # create 8 local crossbars for accelerator -> CPU talking with the following params
    NoncoherentXBar(
        width=32,
        frontend_latency=1,
        forward_latency=1,
        response_latency=1,
    )
    for _ in range(NUM_ACCELERATORS)
]

for i in range(NUM_ACCELERATORS): #requests flow from the cpu side ports to the memory side ports
    system.mem_ctrls[i].port = (
        system.local_xbars[i].mem_side_ports
    )

def _generator_config(index):
    trace_dir = os.environ.get("TRACE_DIR", "")
    if trace_dir:
        path = os.path.join(trace_dir, f"accel{index}.cfg")
        if os.path.isfile(path):
            return path
        return "configs/idle.cfg"
    if index == 0:
        return "configs/routing-smoke.cfg"
    return "configs/idle.cfg"


system.generators = [
    TrafficGen(
        config_file=_generator_config(i),
        max_outstanding_reqs=32,
        progress_check="10ms",
    )
    for i in range(NUM_ACCELERATORS)
]

for i in range(NUM_ACCELERATORS): # each generator plugs into the cpu side port of the xbar
    system.generators[i].port = (
        system.local_xbars[i].cpu_side_ports
    )


system.fabric = NoncoherentXBar( #setup the crossbar fabric
    width=32,
    frontend_latency=1,
    forward_latency=1,
    response_latency=1,
)

# each accelerator sends requests for other owners through this bridge
system.outgoing = [ 
    Bridge(
        delay="20ns",
        ranges=[
            system.mem_ranges[j]
            for j in range(NUM_ACCELERATORS)
            if j != i
        ],
    )
    for i in range(NUM_ACCELERATORS)
]

#the fabric sends requests to each owner through this bridge
system.incoming = [
    Bridge(
        delay="20ns",
        ranges=[system.mem_ranges[i]],
    )
    for i in range(NUM_ACCELERATORS)
]

for i in range(NUM_ACCELERATORS):
    #local router → outgoing bridge → fabric
    system.outgoing[i].cpu_side_port = (
        system.local_xbars[i].mem_side_ports
    )
    system.outgoing[i].mem_side_port = (
        system.fabric.cpu_side_ports
    )

    #fabric → incoming bridge → owners local router
    system.incoming[i].cpu_side_port = (
        system.fabric.mem_side_ports
    )
    system.incoming[i].mem_side_port = (
        system.local_xbars[i].cpu_side_ports
    )

root = Root(full_system=False, system=system)

m5.instantiate()

exit_event = m5.simulate()

print(
    f"Exited at tick {m5.curTick()}: "
    f"{exit_event.getCause()}"
)
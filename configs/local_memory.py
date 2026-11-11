"""One traffic generator connected to local DDR5 memory."""

import m5

from m5.objects import (
    AddrRange,
    DDR5_4400_4x8,
    MemCtrl,
    Root,
    SrcClockDomain,
    System,
    TrafficGen,
    VoltageDomain,
)

system = System() # initialize the system
 
system.clk_domain = SrcClockDomain( # set the clk domain
    clock="2.4GHz",
    voltage_domain=VoltageDomain(),
)

system.mem_mode = "timing" #set the memory cache line sizing
system.cache_line_size = 64

system.mem_ranges = [ # determine how much memory to use
    AddrRange(start=0, size="2GiB")
]

system.mem_ctrl = MemCtrl()

system.mem_ctrl.dram = DDR5_4400_4x8( # describe the dram type
    range=system.mem_ranges[0]
)

system.generator = TrafficGen( # define the config for the traffic generator
    config_file="configs/traffic.cfg",
    max_outstanding_reqs=32,
    progress_check="10ms",
)

system.generator.port = system.mem_ctrl.port # describe the response port for the generator

root = Root(full_system=False, system=system)

m5.instantiate()

exit_event = m5.simulate()

print(
    f"Exited at tick {m5.curTick()}: "
    f"{exit_event.getCause()}"
)
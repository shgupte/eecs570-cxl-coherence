import m5
from m5.objects import (
    AddrRange, EmbeddingEndpoint, Root, SimpleMemory,
    SrcClockDomain, System, VoltageDomain,
)

system = System()
system.clk_domain = SrcClockDomain(
    clock="1GHz",
    voltage_domain=VoltageDomain(),
)
system.mem_mode = "timing"
system.mem_ranges = [AddrRange("1MiB")]

system.memory = SimpleMemory(
    range=system.mem_ranges[0],
    latency="50ns",
)

system.endpoint = EmbeddingEndpoint()
system.endpoint.memory_port = system.memory.port

root = Root(full_system=False, system=system)

m5.instantiate()
m5.simulate(500_000)
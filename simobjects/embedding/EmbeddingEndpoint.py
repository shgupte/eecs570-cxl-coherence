from m5.params import Param, RequestPort
from m5.SimObject import SimObject
from m5.proxy import Parent


class EmbeddingEndpoint(SimObject):
    type = "EmbeddingEndpoint"
    cxx_header = "embedding/embedding_endpoint.hh"
    cxx_class = "gem5::EmbeddingEndpoint"

    update_latency = Param.Latency("100ns", "Fixed update overhead")
    cycle_time = Param.Latency("1ns", "Accelerator cycle time")

    embedding_dim = Param.Unsigned(128, "Values in one embedding row")
    gradient_count = Param.Unsigned(2, "Gradients for this row")
    ops_per_cycle = Param.Unsigned(64, "Scalar operations per cycle")

    memory_port = RequestPort("Connection to embedding memory")
    system = Param.System(Parent.any, "System containing this endpoint")
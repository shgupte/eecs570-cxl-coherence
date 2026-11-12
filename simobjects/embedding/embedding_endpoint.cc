#include "embedding_endpoint.hh"
#include "base/logging.hh"
#include <cstdint>
#include "sim/system.hh"
#include <memory>

namespace gem5
{

EmbeddingEndpoint::EmbeddingEndpoint(
    const EmbeddingEndpointParams &params
) : SimObject(params),
    updateLatency(params.update_latency),
    requestorId(params.system->getRequestorId(this)),
    updateEvent([this] { finishUpdate(); }, name()),
    readEvent([this] { issueRead(); }, name() + ".read"),
    memoryPort(name() + ".memory_port", *this)
{
    fatal_if ( // define the fatal conditions pretty much just if everything is 0
        params.gradient_count == 0 ||
        params.embedding_dim == 0 ||
        params.ops_per_cycle == 0,
        "Embedding dimensions, gradient count, and throughput must be positive"
    );
    
    const std::uint64_t aggregationOps =  
        std::uint64_t(params.gradient_count - 1) * params.embedding_dim; 

    const std::uint64_t sgdOps = 
        2ULL * params.embedding_dim;

    // prev 2 calculations just used to get the total number of ops which helps us determine the # of cycles
    const std::uint64_t totalOps = aggregationOps + sgdOps;

    // the number of cycles is determined by IPC
    const std::uint64_t cycles = 
        totalOps / params.ops_per_cycle +
        (totalOps % params.ops_per_cycle != 0);

    // log the total update latency
    updateLatency += cycles * params.cycle_time;
    
}

EmbeddingEndpoint::MemoryPort::MemoryPort( // construct the memory port
    const std::string &name,
    EmbeddingEndpoint &endpoint
) : RequestPort(name), endpoint(endpoint)
{
}

Port &EmbeddingEndpoint::getPort(const std::string &if_name, PortID idx) {
    if (if_name == "memory_port") { // return the memory port if its the one in the python desc
        return memoryPort;
    }

    return SimObject::getPort(if_name, idx); //else just let the sim object figure out which port it is
}

bool EmbeddingEndpoint::MemoryPort::recvTimingResp(PacketPtr pkt) {
    return endpoint.handleMemoryResponse(pkt); //does a timed memory response and returns whether it passed or failed
}

void EmbeddingEndpoint::MemoryPort::recvReqRetry() { //handle a retry if we recv a req to retry
    endpoint.handleMemoryRetry();
}

void EmbeddingEndpoint::sendMemoryRequest(PacketPtr pkt) {
    panic_if(retryPacket != nullptr, "Already waiting to retry"); // log if we try to retyr again

    if (!memoryPort.sendTimingReq(pkt)) { //if the timingrequest fails then set the retry packet to pkt
        retryPacket = pkt;
    }
}

void EmbeddingEndpoint::handleMemoryRetry() {
    panic_if(retryPacket == nullptr, "No packet to retry");

    PacketPtr pkt = retryPacket;
    retryPacket = nullptr;
    sendMemoryRequest(pkt);
}

void
EmbeddingEndpoint::issueRead()
{
    const Addr address = 0x1000;
    const unsigned size = 64;

    auto request = std::make_shared<Request>(
        address, size, Request::Flags{}, requestorId
    );

    PacketPtr pkt = new Packet(request, MemCmd::ReadReq);
    pkt->allocate();

    inform("Issuing embedding read\n");
    sendMemoryRequest(pkt);
}

bool
EmbeddingEndpoint::handleMemoryResponse(PacketPtr pkt)
{
    panic_if(pkt->isError(), "Memory request failed");

    const bool isRead = pkt->isRead();
    panic_if(!isRead && !pkt->isWrite(), "Unexpected response");

    delete pkt;

    if (isRead) {
        inform("Read response at tick %llu\n",
               static_cast<unsigned long long>(curTick()));
        schedule(updateEvent, curTick() + updateLatency);
    } else {
        inform("Write response at tick %llu\n",
               static_cast<unsigned long long>(curTick()));
    }

    return true;
}

void
EmbeddingEndpoint::issueWrite()
{
    const Addr address = 0x1000;
    const unsigned size = 64;

    auto request = std::make_shared<Request>(
        address, size, Request::Flags{}, requestorId
    );

    PacketPtr pkt = new Packet(request, MemCmd::WriteReq);
    pkt->dataDynamic(new std::uint8_t[size]());

    inform("Issuing embedding write\n");
    sendMemoryRequest(pkt);
}

void
EmbeddingEndpoint::startup()
{
    schedule(readEvent, curTick());
}

void
EmbeddingEndpoint::finishUpdate()
{
    inform("Processing completed at tick %llu\n",
           static_cast<unsigned long long>(curTick()));

    issueWrite();
}

} // namespace gem5
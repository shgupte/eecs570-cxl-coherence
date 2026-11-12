#ifndef __EMBEDDING_ENDPOINT_HH__
#define __EMBEDDING_ENDPOINT_HH__

#include "base/types.hh"
#include "params/EmbeddingEndpoint.hh"
#include "sim/eventq.hh"
#include "sim/sim_object.hh"
#include "mem/port.hh"
#include "mem/request.hh"

namespace gem5
{

class EmbeddingEndpoint : public SimObject
{
  private:
    Tick updateLatency; // tick is just a 64 representation of time
    RequestorID requestorId; // request id for the memory read
    EventFunctionWrapper updateEvent; //eventfunctionwrapper has a function to execute when event executes +  a name
    EventFunctionWrapper readEvent; // the read event thats being triggered
    
    bool handleMemoryResponse(PacketPtr pkt); // handle memory resps
    void handleMemoryRetry(); // if the memory read fails we need to retry

    void issueRead();
    void sendMemoryRequest(PacketPtr pkt);

    void issueWrite();
    
    void finishUpdate(); 

    PacketPtr retryPacket = nullptr;
    

    class MemoryPort : public RequestPort
    {
      private:
        EmbeddingEndpoint &endpoint;
    
      public:
        MemoryPort(const std::string &name, EmbeddingEndpoint &endpoint);
    
      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;
    };
    
    MemoryPort memoryPort;

  public:
    EmbeddingEndpoint(const EmbeddingEndpointParams &params);

    void startup() override;
    
    Port &getPort(
        const std::string &if_name,
        PortID idx = InvalidPortID
    ) override;
};



} // namespace gem5

#endif
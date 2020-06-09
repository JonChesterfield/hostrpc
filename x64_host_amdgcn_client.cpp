#include "x64_host_amdgcn_client.hpp"

#if defined(__AMDGCN__)
hostrpc::x64_amdgcn_client<hostrpc::x64_host_amdgcn_array_size>
    client_singleton;

void hostrpc::hostcall_client_async(uint64_t data[8])
{
  client_singleton.rpc_invoke<false>(static_cast<void*>(&data[0]));
}

#else

// #include "hsa.hpp" // probably can't use this directly
hostrpc::x64_amdgcn_server<hostrpc::x64_host_amdgcn_array_size>
    server_singleton;

void* hostcall_server_init(hsa_region_t fine, void* client_address)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      new hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>(fine);

  {
    size_t sz = res->client.serialize_size();
    uint64_t bytes[sz];
    res->client.serialize(bytes);
    memcpy(client_address, bytes, sz * sizeof(uint64_t));
  }
  {
    size_t sz = res->server.serialize_size();
    uint64_t bytes[sz];
    res->server.serialize(bytes);
    server_singleton.deserialize(bytes);
  }

  return static_cast<void*>(res);
}

void hostcall_server_dtor(void* arg)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      static_cast<
          hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>*>(arg);
  delete (res);
}

void hostcall_server_handle_one_packet()
{
  server_singleton.rpc_handle(nullptr);
}

#endif

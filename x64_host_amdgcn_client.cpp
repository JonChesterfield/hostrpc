#include "x64_host_amdgcn_client.hpp"

#if defined(__AMDGCN__)
__attribute__((visibility("default"))) hostrpc::x64_amdgcn_client<
    hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>>
    client_singleton;

void hostcall_client(uint64_t data[8])
{
  bool success = false;
  while (!success)
    {
      success =
          client_singleton.rpc_invoke<true>(static_cast<void *>(&data[0]));
    }
}

void hostcall_client_async(uint64_t data[8])
{
  bool success = false;
  while (!success)
    {
      client_singleton.rpc_invoke<false>(static_cast<void *>(&data[0]));
    }
}

#else

namespace hostrpc
{
thread_local unsigned my_id = 0;
}  // namespace hostrpc

const char* hostcall_client_symbol() { return "client_singleton"; }

hostrpc::x64_amdgcn_server<
    hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>>
    server_singleton;

void* hostcall_server_init(hsa_region_t fine, hsa_region_t gpu_coarse,
                           void* client_address)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      new hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>(
          fine, gpu_coarse);

  using ct = decltype(
      hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>::client);

  *reinterpret_cast<ct*>(client_address) = res->client;
  server_singleton = res->server;

  return static_cast<void*>(res);
}

void hostcall_server_dtor(void* arg)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      static_cast<
          hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>*>(arg);
  delete (res);
}

bool hostcall_server_handle_one_packet(void* arg)
{
  hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>* res =
      static_cast<
          hostrpc::x64_amdgcn_pair<hostrpc::x64_host_amdgcn_array_size>*>(arg);

  const bool verbose = false;
  if (verbose)
    {
      printf("Client\n");
      res->client.inbox.dump();
      res->client.outbox.dump();
      res->client.active.dump();

      printf("Server\n");
      res->server.inbox.dump();
      res->server.outbox.dump();
      res->server.active.dump();
    }

  bool r = server_singleton.rpc_handle(nullptr);

  if (verbose)
    {
      printf(" --------------\n");
    }

  return r;
}

#endif

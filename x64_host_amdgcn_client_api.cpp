#include "x64_host_amdgcn_client_api.hpp"
#include "base_types.hpp"
#include "interface.hpp"
#include "platform.hpp"

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;

#if defined(__AMDGCN__)
__attribute__((visibility("default")))
hostrpc::x64_amdgcn_t::client_t client_singleton;

void hostcall_client(uint64_t data[8])
{
  bool success = false;

  while (!success)
    {
      void * d = static_cast<void*>(&data[0]);
      success = client_singleton.invoke(d, d);
    }
}

void hostcall_client_async(uint64_t data[8])
{
  bool success = false;

  while (!success)
    {
      void * d = static_cast<void*>(&data[0]);
      success = client_singleton.invoke_async(d, d);
    }
}

#else

const char *hostcall_client_symbol() { return "client_singleton"; }

hostrpc::x64_amdgcn_t::server_t server_singleton;

void *hostcall_server_init(hsa_region_t fine, hsa_region_t gpu_coarse,
                           void *client_address)
{
  hostrpc::x64_amdgcn_t *res =
      new hostrpc::x64_amdgcn_t(fine.handle, gpu_coarse.handle);

  *static_cast<hostrpc::x64_amdgcn_t::client_t *>(client_address) =
      res->client();

  server_singleton = res->server();

  return static_cast<void *>(res);
}

void hostcall_server_dtor(void *arg)
{
  hostrpc::x64_amdgcn_t *res = static_cast<hostrpc::x64_amdgcn_t *>(arg);
  delete (res);
}

bool hostcall_server_handle_one_packet(void *arg)
{
  (void)arg;
  static thread_local uint64_t loc;
  return server_singleton.handle(nullptr, &loc);
}

#endif

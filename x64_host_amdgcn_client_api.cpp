#include "x64_host_amdgcn_client_api.hpp"
#include "base_types.hpp"
#include "interface.hpp"
#include "platform.hpp"

namespace hostrpc
{
namespace x64_host_amdgcn_client_api
{
#if defined(__AMDGCN__)
void fill(hostrpc::page_t *page, void *dv)
{
  uint64_t *d = static_cast<uint64_t *>(dv);
  if (0)
    {
      // Will want to set inactive lanes to nop here, once there are some
      if (platform::is_master_lane())
        {
          for (unsigned i = 0; i < 64; i++)
            {
              page->cacheline[i].element[0] = 0;
            }
        }
    }

  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      line->element[i] = d[i];
    }
}
void use(hostrpc::page_t *page, void *dv)
{
  uint64_t *d = static_cast<uint64_t *>(dv);
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      d[i] = line->element[i];
    }
}

#endif
#if defined(__x86_64__)
void operate(hostrpc::page_t *page, void *)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = 2 * (line.element[i] + 1);
        }
    }
}
#endif
}  // namespace x64_host_amdgcn_client_api
}  // namespace hostrpc

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;

#if defined(__AMDGCN__)
__attribute__((visibility("default")))
hostrpc::x64_amdgcn_t::client_t client_singleton;

void hostcall_client(uint64_t data[8])
{
  bool success = false;

  while (!success)
    {
      void *d = static_cast<void *>(&data[0]);
      success = client_singleton.invoke(d, d);
    }
}

void hostcall_client_async(uint64_t data[8])
{
  bool success = false;

  while (!success)
    {
      void *d = static_cast<void *>(&data[0]);
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

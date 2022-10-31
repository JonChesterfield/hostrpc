#include "hostrpc_printf.h"
#include "hostrpc_printf_enable.hpp"

#include "x64_gcn_type.hpp"
#undef printf

#include "cxa_atexit.hpp"

#include "hostrpc_printf_client.hpp"
#include "hostrpc_printf_server.hpp"

#if (HOSTRPC_AMDGCN)

__attribute__((visibility("default")))
hostrpc::x64_gcn_type<hostrpc::size_runtime<uint32_t>>::client_type
    hostrpc_x64_gcn_debug_client[1];

// Makes function instantiation simpler
HOSTRPC_PRINTF_INSTANTIATE_CLIENT(hostrpc::x64_gcn_type<hostrpc::size_runtime<uint32_t>>::client_type,
                                  &hostrpc_x64_gcn_debug_client[0])

#endif

#if (HOSTRPC_HOST)

#include "hsa.hpp"

#include "incprintf.hpp"

#include <algorithm>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using SZ = hostrpc::size_runtime<uint32_t>;

struct global
{
  static uint32_t size_p(hsa_agent_t kernel_agent)
  {
    uint32_t cus = hsa::agent_get_info_compute_unit_count(kernel_agent);
    uint32_t waves = hsa::agent_get_info_max_waves_per_cu(kernel_agent);
    uint32_t nonblocking_size = cus * waves;
    return hostrpc::round64(nonblocking_size);
  }

  static std::unique_ptr<hostrpc::x64_gcn_type<SZ>> alloc_p(
      hsa_agent_t kernel_agent)
  {
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);

    uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
    if (fine_grained_region.handle == fail ||
        coarse_grained_region.handle == fail)
      {
        fprintf(stderr, "Failed to find allocation region on kernel agent\n");
        exit(1);
      }

    SZ N = size_p(kernel_agent);

    return std::make_unique<hostrpc::x64_gcn_type<SZ>>(
        N, hostrpc::arch::x64{},
        hostrpc::arch::gcn{fine_grained_region.handle,
                           coarse_grained_region.handle});
  }

  using wrap_state = wrap_server_state<hostrpc::x64_gcn_type<SZ>>;

  hsa::init hsa_instance;
  std::vector<std::unique_ptr<wrap_state>> state;
  static pthread_mutex_t mutex;

  global(const global &) = delete;
  global(global &&) = delete;

  global() : hsa_instance{} {}

  hostrpc::x64_gcn_type<SZ> *spawn(hsa_agent_t kernel_agent)
  {
    lock l(&mutex);
    state.emplace_back(std::unique_ptr<wrap_state>(
        new wrap_state(alloc_p(kernel_agent), size_p(kernel_agent))));
    hostrpc::x64_gcn_type<SZ> *r = state.back().get()->p.get();
    return r;
  }

 private:
  struct lock
  {
    lock(pthread_mutex_t *m) : m(m) { pthread_mutex_lock(m); }
    ~lock() { pthread_mutex_unlock(m); }
    pthread_mutex_t *m;
  };
} global_instance;
pthread_mutex_t global::mutex = PTHREAD_MUTEX_INITIALIZER;
}  // namespace

int hostrpc_print_enable_on_hsa_agent(hsa_executable_t ex,
                                      hsa_agent_t kernel_agent)
{
  const bool verbose = true;

  if (verbose) fprintf(stderr, "called print enable\n");
  const char *gpu_local_ptr = "hostrpc_x64_gcn_debug_client";

  hsa_executable_symbol_t symbol;
  {
    hsa_status_t rc = hsa_executable_get_symbol_by_name(ex, gpu_local_ptr,
                                                        &kernel_agent, &symbol);
    if (rc != HSA_STATUS_SUCCESS)
      {
        if (verbose) fprintf(stderr, "can't find symbol %s\n", gpu_local_ptr);
        return 1;
      }

    hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
    if (kind != HSA_SYMBOL_KIND_VARIABLE)
      {
        if (verbose) fprintf(stderr, "symbol not a variable\n");
        return 1;
      }
  }

  void *addr =
      reinterpret_cast<void *>(hsa::symbol_get_info_variable_address(symbol));

  hostrpc::x64_gcn_type<SZ> *p = global_instance.spawn(kernel_agent);

  for (uint32_t i = 0; i < p->server.size(); i++)
    {
      hostrpc::page_t *page = &p->server.shared_buffer[i];
      hostrpc::port_t port = static_cast<hostrpc::port_t>(i);
      clear()(port, page);
    }

  {
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    auto c = hsa::allocate(fine_grained_region, sizeof(p->client));
    void *vc = c.get();
    if (verbose)
      if (!vc) fprintf(stderr, "Alloc failed\n");
    memcpy(vc, &p->client, sizeof(p->client));
    int rc = hsa::copy_host_to_gpu(kernel_agent, addr, vc, sizeof(p->client));
    if (rc != 0)
      {
        if (verbose) fprintf(stderr, "Copy host to gpu failed\n");
        return 1;
      }
  }

  if (verbose) fprintf(stderr, "Returning success (0) from print_enable\n");
  return 0;
}

#endif

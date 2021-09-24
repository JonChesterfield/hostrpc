#include "hostcall.hpp"
#include "base_types.hpp"
#include "hostcall_hsa.hpp"
#include "queue_to_index.hpp"

#include <stddef.h>
#include <stdint.h>

#include "allocator.hpp"
#include "detail/client_impl.hpp"
#include "platform/detect.hpp"
#include "detail/server_impl.hpp"
#include "host_client.hpp"
#include "x64_gcn_type.hpp"

#if HOSTRPC_HOST
#include <new>
#endif

// trying to get something running on gfx8
#if HOSTRPC_HOST
#include "hsa.hpp"
#include <array>
#include <cassert>
#include <thread>
#include <vector>
#endif

#include "platform.hpp"  // assert

// a 'per queue' structure, one per gpu, is basically a global variable
// could be factored as such

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;

#if defined(__AMDGCN__)

namespace hostrpc
{
namespace x64_host_amdgcn_client
{
struct fill
{
  uint64_t *d;
  fill(uint64_t *d) : d(d) {}
  void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    hostcall_ops::pass_arguments(page, d);
  };
};

struct use
{
  uint64_t *d;
  use(uint64_t *d) : d(d) {}
  void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    hostcall_ops::use_result(page, d);
  };
};
}  // namespace x64_host_amdgcn_client
}  // namespace hostrpc

// Accessing this, sometimes, raises a page not present fault on gfx8
// drawback of embedding in image is that multiple shared libraries will all
// need their own copy, whereas it really should be one per gpu

// Doesn't need to be initialized, though zeroing might help debugging
__attribute__((visibility("default")))
hostrpc::x64_gcn_type<SZ>::client_type *client_singleton;

template <bool C>
static void hostcall_impl(uint64_t data[8])
{
  hostrpc::x64_host_amdgcn_client::fill f(&data[0]);
  hostrpc::x64_host_amdgcn_client::use u(&data[0]);

  auto *c = &client_singleton[get_queue_index()];

  bool success = false;
  while (!success)
    {
      if (C)
        {
          success = c->rpc_invoke(f, u);
        }
      else
        {
          success = c->rpc_invoke(f);
        }
    }
}

void hostcall_client(uint64_t data[8]) { return hostcall_impl<true>(data); }

void hostcall_client_async(uint64_t data[8])
{
  return hostcall_impl<false>(data);
}

#endif

#if HOSTRPC_HOST

namespace hostrpc
{
namespace x64_host_amdgcn_client
{
struct operate
{
  void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    hostcall_ops::operate(page);
  }
};

struct clear
{
  void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    hostcall_ops::clear(page);
  }
};
}  // namespace x64_host_amdgcn_client
}  // namespace hostrpc

class hostcall_impl
{
 public:
  hostcall_impl(hsa_agent_t kernel_agent);

  hostcall_impl(hostcall_impl &&o) = delete;
  hostcall_impl(const hostcall_impl &) = delete;

  static uint64_t find_symbol_address(hsa_executable_t &ex,
                                      hsa_agent_t kernel_agent,
                                      const char *sym);

  int enable_executable(hsa_executable_t ex)
  {
    const char *gpu_local_ptr = "client_singleton";
    hsa_executable_symbol_t symbol;
    {
      hsa_status_t rc = hsa_executable_get_symbol_by_name(
          ex, gpu_local_ptr, &kernel_agent, &symbol);
      if (rc != HSA_STATUS_SUCCESS)
        {
          return 1;
        }

      hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
      if (kind != HSA_SYMBOL_KIND_VARIABLE)
        {
          return 1;
        }
    }

    void *addr =
        reinterpret_cast<void *>(hsa::symbol_get_info_variable_address(symbol));

#if 0
    fprintf(stderr, "addr 0x%lx, passing to copy 0x%lx <- 0x%lx via 0x%lx\n",
            (uint64_t)addr,
            (uint64_t)&addr,
            (uint64_t)&hsa_coarse_clients,
            (uint64_t)hsa_fine_scratch);
#endif

    // need 8 bytes of scratch, ignores the type here
    memcpy(hsa_fine_scratch, &hsa_coarse_clients, sizeof(void *));

    int rc = hsa::copy_host_to_gpu(kernel_agent, (void *)addr,
                                   (void *)hsa_fine_scratch, sizeof(void *));

    if (rc != 0)
      {
        return 1;
      }

    return 0;
  }

  int enable_queue(hsa_queue_t *queue)
  {
    uint16_t queue_id = queue_to_index(queue);
    if (stored_pairs[queue_id] != 0)
      {
        // already enabled
        return 0;
      }

    // TODO: Avoid this heap alloc?
    auto res = std::unique_ptr<hostrpc::x64_gcn_type<SZ>>(
        new (std::nothrow) hostrpc::x64_gcn_type<SZ>(
            SZ{}, fine_grained_region.handle, coarse_grained_region.handle));
    if (!res)
      {
        return 1;
      }

    // This is fairly ugly. May want to unconditionally call the
    // amdgpu clear() function instead, which happens to do the same
    // thing (if called with all lanes active)
    uint64_t hostrpc_nop = UINT64_MAX;
    for (uint64_t i = 0; i < res->server.size(); i++)
      {
        hostrpc::page_t *page = &res->server.shared_buffer[i];
        for (uint64_t c = 0; c < 64; c++)
          {
            hostrpc::cacheline_t *line = &page->cacheline[c];
            for (uint64_t e = 0; e < 8; e++)
              {
                line->element[e] = hostrpc_nop;
              }
          }
      }

    // clients is on the gpu and res->client is not
    if (0)
      {
        // fails on gfx8, might need a barrier on gfx9
        hsa_coarse_clients[queue_id] = res->client;
      }
    else
      {
        // should work on gfx8, possibly slowly. Route via fine grain memory.
        *hsa_fine_scratch = res->client;
        int rc = hsa::copy_host_to_gpu(
            kernel_agent, hsa_coarse_clients + queue_id, hsa_fine_scratch);
        *hsa_fine_scratch = {};

        if (rc != 0)
          {
            return 1;
          }
      }

    stored_pairs[queue_id] = std::move(res);

    return 0;
  }

  int spawn_worker(hsa_queue_t *queue)
  {
    uint16_t queue_id = queue_to_index(queue);
    if (stored_pairs[queue_id] == 0)
      {
        return 1;
      }
    return spawn_worker(queue_id);
  }

  ~hostcall_impl();

 private:
  int spawn_worker(uint16_t queue_id)
  {
    HOSTRPC_ATOMIC(uint32_t) *control = &thread_killer;
    auto server = stored_pairs[queue_id]->server;

    // TODO. Can't actually use std::thread because the constructor throws.
    threads.emplace_back([control, server]() mutable {
      uint32_t ql = 0;
      for (;;)
        {
          hostrpc::x64_host_amdgcn_client::operate op;
          hostrpc::x64_host_amdgcn_client::clear cl;
          while (server.rpc_handle(op, cl, &ql))
            {
            }

          if (*control != 0)
            {
              return;
            }

          platform::sleep_briefly();
        }
    });
    return 0;  // can't detect errors from std::thread
  }

  // pointer to gpu array, allocated in coarse (per-gpu)
  hostrpc::x64_gcn_type<SZ>::client_type *hsa_coarse_clients;

  hostrpc::x64_gcn_type<SZ>::client_type *hsa_fine_scratch;

  std::array<std::unique_ptr<hostrpc::x64_gcn_type<SZ>>, MAX_NUM_DOORBELLS>
      stored_pairs;

  HOSTRPC_ATOMIC(uint32_t) thread_killer = 0;
  std::vector<std::thread> threads;

  hsa_agent_t kernel_agent;
  hsa_region_t fine_grained_region;
  hsa_region_t coarse_grained_region;
};

// todo: port to hsa.h api
// todo: constructor function instead of constructor for error return
hostcall_impl::hostcall_impl(hsa_agent_t kernel_agent)
    : kernel_agent(kernel_agent)
{
  using Ty = hostrpc::x64_gcn_type<SZ>::client_type;

  // todo: error checks here
  fine_grained_region = hsa::region_fine_grained(kernel_agent);

  coarse_grained_region = hsa::region_coarse_grained(kernel_agent);

  {
    void *ptr;
    hsa_status_t r = hsa_memory_allocate(fine_grained_region, sizeof(Ty), &ptr);
    hsa_fine_scratch = (r == HSA_STATUS_SUCCESS)
                           ? new (reinterpret_cast<Ty *>(ptr)) Ty
                           : nullptr;
  }

  {
    void *ptr;
    hsa_status_t r = hsa_memory_allocate(coarse_grained_region,
                                         sizeof(Ty) * MAX_NUM_DOORBELLS, &ptr);
    if (r == HSA_STATUS_SUCCESS)
      {
        Ty *cast = reinterpret_cast<Ty *>(ptr);
        for (size_t i = 0; i < MAX_NUM_DOORBELLS; i++)
          {
            new (cast + i) Ty;
          }
        hsa_coarse_clients = __builtin_launder(cast);
      }
    else
      {
        hsa_coarse_clients = nullptr;
      }
  }
}

hostcall_impl::~hostcall_impl()
{
  thread_killer = 1;
  for (size_t i = 0; i < threads.size(); i++)
    {
      threads[i].join();
    }
  using Ty = hostrpc::x64_gcn_type<SZ>::client_type;
  hsa_fine_scratch->~Ty();
  hsa_memory_free(hsa_fine_scratch);

  for (size_t i = 0; i < MAX_NUM_DOORBELLS; i++)
    {
      hsa_coarse_clients[i].~Ty();
    }
  hsa_memory_free(hsa_coarse_clients);
}

uint64_t hostcall_impl::find_symbol_address(hsa_executable_t &ex,
                                            hsa_agent_t kernel_agent,
                                            const char *sym)
{
  // TODO: This was copied from the loader, sort out the error handling
  hsa_executable_symbol_t symbol;
  {
    hsa_status_t rc =
        hsa_executable_get_symbol_by_name(ex, sym, &kernel_agent, &symbol);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "HSA failed to find symbol %s\n", sym);
        exit(1);
      }
  }

  hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
  if (kind != HSA_SYMBOL_KIND_VARIABLE)
    {
      fprintf(stderr, "Symbol %s is not a variable\n", sym);
      exit(1);
    }

  return hsa::symbol_get_info_variable_address(symbol);
}

template <size_t expect, size_t actual>
static void assert_size_t_equal()
{
  static_assert(expect == actual, "");
}

hostcall::hostcall(hsa_agent_t kernel_agent)
    : state_(std::unique_ptr<hostcall_impl>(new (std::nothrow)
                                                hostcall_impl(kernel_agent)))
{
}

hostcall::~hostcall() {}

int hostcall::enable_executable(hsa_executable_t ex)
{
  return state_->enable_executable(ex);
}
int hostcall::enable_queue(hsa_queue_t *queue)
{
  return state_->enable_queue(queue);
}
int hostcall::spawn_worker(hsa_queue_t *queue)
{
  return state_->spawn_worker(queue);
}

#endif

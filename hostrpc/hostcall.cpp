#include "hostcall.hpp"
#include "base_types.hpp"
#include "queue_to_index.hpp"
#include "x64_host_gcn_client.hpp"

#include <stddef.h>
#include <stdint.h>

namespace hostrpc
{
namespace x64_host_amdgcn_client
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__AMDGCN__)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::pass_arguments(page, d);
#else
    (void)page;
    (void)dv;
#endif
  };
};

struct use
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__AMDGCN__)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::use_result(page, d);
#else
    (void)page;
    (void)dv;
#endif
  };
};

struct operate
{
  static void call(hostrpc::page_t *page, void *)
  {
#if defined(__x86_64__)
    hostcall_ops::operate(page);
#else
    (void)page;
#endif
  }
};

struct clear
{
  static void call(hostrpc::page_t *page, void *)
  {
#if defined(__x86_64__)
    hostcall_ops::clear(page);
#else
    (void)page;
#endif
  }
};
}  // namespace x64_host_amdgcn_client

using SZ = size_compiletime<hostrpc::x64_host_amdgcn_array_size>;
using x64_amdgcn_pair = hostrpc::x64_gcn_pair_T<
    SZ, x64_host_amdgcn_client::fill, x64_host_amdgcn_client::use,
    x64_host_amdgcn_client::operate, x64_host_amdgcn_client::clear,
    counters::client_nop, counters::server_nop>;

}  // namespace hostrpc

// trying to get something running on gfx8
#if defined(__x86_64__)
#include "../impl/data.h"
#include "hostcall.h"
#include "hsa.hpp"
#include <cassert>
#include <thread>
#include <vector>
#endif

#include "detail/platform.hpp"  // assert

// a 'per queue' structure, one per gpu, is basically a global variable
// could be factored as such

#if defined(__AMDGCN__)

// Accessing this, sometimes, raises a page not present fault on gfx8
// drawback of embedding in image is that multiple shared libraries will all
// need their own copy, whereas it really should be one per gpu

__attribute__((visibility("default")))
hostrpc::x64_amdgcn_pair::client_type client_singleton[MAX_NUM_DOORBELLS];

template <bool C>
static void hostcall_impl(uint64_t data[8])
{
  auto *c = &client_singleton[get_queue_index()];

  bool success = false;
  while (!success)
    {
      void *d = static_cast<void *>(&data[0]);
      success = c->rpc_invoke<C>(d, d);
    }
}

void hostcall_client(uint64_t data[8]) { return hostcall_impl<true>(data); }

void hostcall_client_async(uint64_t data[8])
{
  return hostcall_impl<false>(data);
}

#endif

#if defined(__x86_64__)

// Get the start of the array
const char *hostcall_client_symbol() { return "client_singleton"; }

class hostcall_impl
{
  using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;

 public:
  hostcall_impl(void *client_symbol_address, hsa_agent_t kernel_agent);
  hostcall_impl(hsa_executable_t executable, hsa_agent_t kernel_agent);

  hostcall_impl(hostcall_impl &&o)
      : clients(std::move(o.clients)),
        stored_pairs(std::move(o.stored_pairs)),
        threads(std::move(o.threads)),
        fine_grained_region(std::move(o.fine_grained_region)),
        coarse_grained_region(std::move(o.coarse_grained_region))
  {
    clients = nullptr;
    stored_pairs = {};
    // threads = {};
  }

  hostcall_impl(const hostcall_impl &) = delete;

  static uint64_t find_symbol_address(hsa_executable_t &ex,
                                      hsa_agent_t kernel_agent,
                                      const char *sym);

  int enable_queue(hsa_agent_t kernel_agent, hsa_queue_t *queue)
  {
    uint16_t queue_id = queue_to_index(queue);
    if (stored_pairs[queue_id] != 0)
      {
        // already enabled
        return 0;
      }

    // TODO: Avoid this heap alloc
    hostrpc::x64_amdgcn_pair *res = new hostrpc::x64_amdgcn_pair(
        SZ{}, fine_grained_region.handle, coarse_grained_region.handle);
    if (!res)
      {
        return 1;
      }

    // clients is on the gpu and res->client is not

    if (0)
      {
        clients[queue_id] = res->client;  // fails on gfx8
      }
    else
      {
        // should work on gfx8, possibly slowly
        hsa_region_t fine = hsa::region_fine_grained(kernel_agent);
        using Ty = hostrpc::x64_amdgcn_pair::client_type;
        auto c = hsa::allocate(fine, sizeof(Ty));
        auto *l = new (reinterpret_cast<Ty *>(c.get())) Ty(res->client);

        int rc = hsa::copy_host_to_gpu(
            kernel_agent, reinterpret_cast<void *>(&clients[queue_id]),
            reinterpret_cast<const void *>(l), sizeof(Ty));

        // Can't destroy it here - the destructor (rightly) cleans up the
        // memory, but the memcpy above has put those onto the gpu l->~Ty(); //
        // this may be a bad move

        if (rc != 0)
          {
            return 1;
          }
      }

    stored_pairs[queue_id] = res;

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

  ~hostcall_impl()
  {
    thread_killer = 1;
    for (size_t i = 0; i < threads.size(); i++)
      {
        threads[i].join();
      }
    for (size_t i = 0; i < MAX_NUM_DOORBELLS; i++)
      {
        delete stored_pairs[i];
      }
  }

 private:
  int spawn_worker(uint16_t queue_id)
  {
    _Atomic(uint32_t) *control = &thread_killer;
    auto server = stored_pairs[queue_id]->server;

    // TODO. Can't actually use std::thread because the constructor throws.
    threads.emplace_back([control, server]() mutable {
      uint32_t ql = 0;
      for (;;)
        {
          while (server.rpc_handle(nullptr, &ql))
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

  // Going to need these to be opaque
  hostrpc::x64_amdgcn_pair::client_type *clients;  // static, on gpu

  std::vector<hostrpc::x64_amdgcn_pair *> stored_pairs;

  _Atomic(uint32_t) thread_killer = 0;
  std::vector<std::thread> threads;
  hsa_region_t fine_grained_region;
  hsa_region_t coarse_grained_region;
};

// todo: port to hsa.h api

hostcall_impl::hostcall_impl(void *client_addr, hsa_agent_t kernel_agent)
{
  // The client_t array is per-gpu-image. Find it.
  clients =
      reinterpret_cast<hostrpc::x64_amdgcn_pair::client_type *>(client_addr);

  // todo: error checks here
  fine_grained_region = hsa::region_fine_grained(kernel_agent);

  coarse_grained_region = hsa::region_coarse_grained(kernel_agent);

  // probably can't use vector for exception-safety reasons
  stored_pairs.resize(MAX_NUM_DOORBELLS);
}

hostcall_impl::hostcall_impl(hsa_executable_t executable,
                             hsa_agent_t kernel_agent)

    : hostcall_impl(reinterpret_cast<void *>(find_symbol_address(
                        executable, kernel_agent, hostcall_client_symbol())),
                    kernel_agent)
{
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

hostcall::hostcall(hsa_executable_t executable, hsa_agent_t kernel_agent)
{
  assert_size_t_equal<hostcall::state_t::align(), alignof(hostcall_impl)>();
  assert_size_t_equal<hostcall::state_t::size(), sizeof(hostcall_impl)>();
  new (reinterpret_cast<hostcall_impl *>(state.data))
      hostcall_impl(hostcall_impl(executable, kernel_agent));
}

hostcall::hostcall(void *client_symbol_address, hsa_agent_t kernel_agent)
{
  assert_size_t_equal<hostcall::state_t::align(), alignof(hostcall_impl)>();
  assert_size_t_equal<hostcall::state_t::size(), sizeof(hostcall_impl)>();
  new (reinterpret_cast<hostcall_impl *>(state.data))
      hostcall_impl(hostcall_impl(client_symbol_address, kernel_agent));
}

hostcall::~hostcall() { state.destroy<hostcall_impl>(); }

bool hostcall::valid() { return true; }

int hostcall::enable_queue(hsa_agent_t kernel_agent, hsa_queue_t *queue)
{
  return state.open<hostcall_impl>()->enable_queue(kernel_agent, queue);
}
int hostcall::spawn_worker(hsa_queue_t *queue)
{
  return state.open<hostcall_impl>()->spawn_worker(queue);
}

static std::vector<std::unique_ptr<hostcall>> state;

void spawn_hostcall_for_queue(uint32_t device_id, hsa_agent_t agent,
                              hsa_queue_t *queue, void *client_symbol_address)
{
  // printf("Setting up hostcall on id %u, queue %lx\n", device_id,
  // (uint64_t)queue); uint64_t * w = (uint64_t*)queue; printf("Host words at q:
  // 0x%lx 0x%lx 0x%lx 0x%lx\n", w[0],w[1],w[2],w[3]);
  if (device_id >= state.size())
    {
      state.resize(device_id + 1);
    }

  // Only create a single instance backed by a single thread per queue at
  // present
  if (state[device_id] != nullptr)
    {
      return;
    }

  // make an instance for this device if there isn't already one
  if (state[device_id] == nullptr)
    {
      std::unique_ptr<hostcall> r(new hostcall(client_symbol_address, agent));
      if (r && r->valid())
        {
          state[device_id] = std::move(r);
          assert(state[device_id] != nullptr);
        }
      else
        {
          fprintf(stderr, "Failed to construct a hostcall, going to assert\n");
        }
    }

  assert(state[device_id] != nullptr);
  // enabling it for a queue repeatedly is a no-op

  if (state[device_id]->enable_queue(agent, queue) == 0)
    {
      // spawn an additional thread
      if (state[device_id]->spawn_worker(queue) == 0)
        {
          // printf("Success for setup on id %u, queue %lx, ptr %lx\n",
          // device_id,
          //      (uint64_t)queue, (uint64_t)client_symbol_address);
          // all good
        }
    }

  // TODO: Indicate failure
}

void free_hostcall_state() { state.clear(); }

#endif

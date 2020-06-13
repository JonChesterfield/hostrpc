#include "client_impl.hpp"
#include "interface.hpp"
#include "platform.hpp"
#include "server_impl.hpp"
#include "x64_host_amdgcn_client_api.hpp"

// hsa uses freestanding C headers, unlike hsa.hpp
#if !defined(__AMDGCN__)
#include "hsa.h"
#include <new>
#include <string.h>
#endif

namespace hostrpc
{
namespace x64_host_amdgcn_client
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
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
  };
};

struct use
{
  static void call(hostrpc::page_t *page, void *dv)
  {
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        d[i] = line->element[i];
      }
  };
};

struct operate
{
  static void call(hostrpc::page_t *page, void *)
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
};
}  // namespace x64_host_amdgcn_client

template <typename SZ>
using x64_amdgcn_client =
    hostrpc::client_impl<SZ, hostrpc::copy_functor_given_alias,
                         x64_host_amdgcn_client::fill,
                         x64_host_amdgcn_client::use, hostrpc::nop_stepper>;

template <typename SZ>
using x64_amdgcn_server =
    hostrpc::server_impl<SZ, hostrpc::copy_functor_given_alias,
                         x64_host_amdgcn_client::operate, hostrpc::nop_stepper>;

// needs to scale with CUs
static const constexpr size_t x64_host_amdgcn_array_size = 2048;

#if !defined(__AMDGCN__)
namespace
{
inline _Atomic uint64_t *hsa_allocate_slot_bitmap_data_alloc(
    hsa_region_t region, size_t size)
{
  const size_t align = 64;
  void *memory = hostrpc::hsa::allocate(region.handle, align, size);
  return reinterpret_cast<_Atomic uint64_t *>(memory);
}

inline void hsa_allocate_slot_bitmap_data_free(_Atomic uint64_t *d)
{
  hostrpc::hsa::deallocate(static_cast<void *>(d));
}

inline void *alloc_from_region(hsa_region_t region, size_t size)
{
  return hostrpc::hsa::allocate(region.handle, 8, size);
}
}  // namespace

template <typename SZ>
struct x64_amdgcn_pair
{
  hostrpc::x64_amdgcn_client<SZ> client;
  hostrpc::x64_amdgcn_server<SZ> server;
  SZ sz;

  x64_amdgcn_pair(SZ sz, hsa_region_t fine, hsa_region_t gpu_coarse) : sz(sz)
  {
    size_t N = sz.N();
    // todo: alignment on the page_t, works at present because allocate has high
    // granularity
    hostrpc::page_t *client_buffer =
        reinterpret_cast<page_t *>(alloc_from_region(fine, N * sizeof(page_t)));
    hostrpc::page_t *server_buffer = client_buffer;

    auto *send_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);
    auto *recv_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);
    auto *client_active_data =
        hsa_allocate_slot_bitmap_data_alloc(gpu_coarse, N);
    auto *server_active_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);

    const size_t size = N;
    slot_bitmap_all_svm send = {size, send_data};
    slot_bitmap_all_svm recv = {size, recv_data};
    slot_bitmap_device client_active = {size, client_active_data};
    slot_bitmap_device server_active = {size, server_active_data};

    client = {sz, recv, send, client_active, server_buffer, client_buffer};

    server = {sz, send, recv, server_active, client_buffer, server_buffer};
  }

  ~x64_amdgcn_pair()
  {
    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hsa_allocate_slot_bitmap_data_free(client.inbox.data());
    hsa_allocate_slot_bitmap_data_free(client.outbox.data());
    hsa_allocate_slot_bitmap_data_free(client.active.data());
    hsa_allocate_slot_bitmap_data_free(server.active.data());

    assert(client.local_buffer == server.remote_buffer);
    assert(client.remote_buffer == server.local_buffer);

    if (client.local_buffer == client.remote_buffer)
      {
        hsa_memory_free(client.local_buffer);
      }
    else
      {
        hsa_memory_free(client.local_buffer);
        hsa_memory_free(server.local_buffer);
      }
  }
};

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;
using ty = x64_amdgcn_pair<SZ>;

x64_amdgcn_t::x64_amdgcn_t(uint64_t hsa_region_t_fine_handle,
                           uint64_t hsa_region_t_coarse_handle)
{
  SZ sz;
  hsa_region_t fine = {.handle = hsa_region_t_fine_handle};
  hsa_region_t coarse = {.handle = hsa_region_t_coarse_handle};

  ty *s = new (std::nothrow) ty(sz, fine, coarse);
  state = static_cast<void *>(s);
}

x64_amdgcn_t::~x64_amdgcn_t()
{
  ty *s = static_cast<ty *>(state);
  if (s)
    {
      delete s;
    }
}

bool x64_amdgcn_t::valid() { return state != nullptr; }

static decltype(ty::client) *open_client(uint64_t *state)
{
  return reinterpret_cast<decltype(ty::client) *>(state);
}
static decltype(ty::server) *open_server(uint64_t *state)
{
  return reinterpret_cast<decltype(ty::server) *>(state);
}

x64_amdgcn_t::client_t x64_amdgcn_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  client_t res;
  auto *cl = reinterpret_cast<decltype(ty::client) *>(&res.state[0]);
  *cl = s->client;
  return res;
}

__attribute__((used)) x64_amdgcn_t::server_t x64_amdgcn_t::server()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  server_t res;
  auto *cl = reinterpret_cast<decltype(ty::server) *>(&res.state[0]);
  *cl = s->server;
  return res;
}

bool x64_amdgcn_t::client_t::invoke_impl(void *application_state)
{
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<true>(application_state);
}

bool x64_amdgcn_t::client_t::invoke_async_impl(void *application_state)
{
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<false>(application_state);
}

bool x64_amdgcn_t::server_t::handle_impl(void *application_state, uint64_t *l)
{
  auto *se = open_server(&state[0]);
  return se->rpc_handle(application_state, l);
}

#endif

}  // namespace hostrpc

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

const char *hostcall_client_symbol() { return "client_singleton"; }

hostrpc::x64_amdgcn_server<
    hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>>
    server_singleton;

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;
using rt = hostrpc::x64_amdgcn_pair<SZ>;

void *hostcall_server_init(hsa_region_t fine, hsa_region_t gpu_coarse,
                           void *client_address)
{
  rt *res = new rt(SZ{}, fine, gpu_coarse);

  using ct = decltype(rt::client);

  *reinterpret_cast<ct *>(client_address) = res->client;
  server_singleton = res->server;

  return static_cast<void *>(res);
}

void hostcall_server_dtor(void *arg)
{
  rt *res = static_cast<rt *>(arg);
  delete (res);
}

bool hostcall_server_handle_one_packet(void *arg)
{
  rt *res = static_cast<rt *>(arg);

  const bool verbose = false;

  const size_t size = hostrpc::x64_host_amdgcn_array_size;
  if (verbose)
    {
      printf("Client\n");
      res->client.inbox.dump(size);
      res->client.outbox.dump(size);
      res->client.active.dump(size);

      printf("Server\n");
      res->server.inbox.dump(size);
      res->server.outbox.dump(size);
      res->server.active.dump(size);
    }

  bool r = server_singleton.rpc_handle(nullptr);

  if (verbose)
    {
      printf(" --------------\n");
    }

  return r;
}

#endif

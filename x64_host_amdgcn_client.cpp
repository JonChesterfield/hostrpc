#include "client_impl.hpp"
#include "interface.hpp"
#include "platform.hpp"
#include "server_impl.hpp"
#include "x64_host_amdgcn_client_api.hpp"

// hsa uses freestanding C headers, unlike hsa.hpp
#if defined(__x86_64__)
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

#if defined(__x86_64__)
namespace
{
inline _Atomic uint64_t *hsa_allocate_slot_bitmap_data_alloc(
    hsa_region_t region, size_t size)
{
  const size_t align = 64;
  void *memory = hostrpc::hsa::allocate(region.handle, align, size);
  return hostrpc::careful_array_cast<_Atomic uint64_t>(memory, size);
}

inline void hsa_allocate_slot_bitmap_data_free(_Atomic uint64_t *d)
{
  hostrpc::hsa::deallocate(static_cast<void *>(d));
}

}  // namespace

#endif

template <typename SZ>
struct x64_amdgcn_pair
{
  hostrpc::x64_amdgcn_client<SZ> client;
  hostrpc::x64_amdgcn_server<SZ> server;
  SZ sz;

  x64_amdgcn_pair(SZ sz, uint64_t fine_handle, uint64_t coarse_handle) : sz(sz)
  {
#if defined(__x86_64__)
    size_t N = sz.N();
    hsa_region_t fine = {.handle = fine_handle};
    hsa_region_t coarse = {.handle = coarse_handle};

    hostrpc::page_t *client_buffer = hostrpc::careful_array_cast<page_t>(
        hostrpc::hsa::allocate(fine_handle, alignof(page_t),
                               N * sizeof(page_t)),
        N);

    hostrpc::page_t *server_buffer = client_buffer;

    auto *send_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);
    auto *recv_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);
    auto *client_active_data = hsa_allocate_slot_bitmap_data_alloc(coarse, N);
    auto *server_active_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);

    slot_bitmap_all_svm send = {N, send_data};
    slot_bitmap_all_svm recv = {N, recv_data};
    slot_bitmap_device client_active = {N, client_active_data};
    slot_bitmap_device server_active = {N, server_active_data};

    client = {sz, recv, send, client_active, server_buffer, client_buffer};

    server = {sz, send, recv, server_active, client_buffer, server_buffer};
#else
    (void)fine_handle;
    (void)coarse_handle;
#endif
  }

  ~x64_amdgcn_pair()
  {
#if defined(__x86_64__)
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
#endif
  }
};

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;
using ty = x64_amdgcn_pair<SZ>;

x64_amdgcn_t::x64_amdgcn_t(uint64_t hsa_region_t_fine_handle,
                           uint64_t hsa_region_t_coarse_handle)
{
  state = nullptr;
#if defined(__x86_64__)
  SZ sz;
  ty *s = new (std::nothrow)
      ty(sz, hsa_region_t_fine_handle, hsa_region_t_coarse_handle);
  state = static_cast<void *>(s);
#else
  (void)hsa_region_t_fine_handle;
  (void)hsa_region_t_coarse_handle;
#endif
}

x64_amdgcn_t::~x64_amdgcn_t()
{
#if defined(__x86_64__)
  ty *s = static_cast<ty *>(state);
  if (s)
    {
      delete s;
    }
#endif
}

bool x64_amdgcn_t::valid() { return state != nullptr; }

#if defined(__AMDGCN__)
static decltype(ty::client) *open_client(unsigned char *state)
{
  return __builtin_launder(reinterpret_cast<decltype(ty::client) *>(state));
}
#endif

#if defined(__x86_64__)
static decltype(ty::server) *open_server(unsigned char *state)
{
  return __builtin_launder(reinterpret_cast<decltype(ty::server) *>(state));
}
#endif

x64_amdgcn_t::client_t x64_amdgcn_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  x64_amdgcn_t::client_t res;
  auto *p = new (reinterpret_cast<decltype(ty::client) *>(res.state)) decltype(
      s->client);
  *p = s->client;

  storage<40, 8> ex;
  using ext = decltype(ty::client);
  auto r = ex.construct<ext>(s->client);
  assert(r == ex.open<ext>());
  ex.destroy<ext>();

  return res;
}

x64_amdgcn_t::server_t x64_amdgcn_t::server()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  x64_amdgcn_t::server_t res;
  auto *p = new (reinterpret_cast<decltype(ty::server) *>(res.state)) decltype(
      s->server);
  *p = s->server;
  return res;
}

bool x64_amdgcn_t::client_t::invoke_impl(void *f, void *u)
{
#if defined(__AMDGCN__)
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<true>(f, u);
#else
  (void)f;
  (void)u;
  return false;
#endif
}

bool x64_amdgcn_t::client_t::invoke_async_impl(void *f, void *u)
{
#if defined(__AMDGCN__)
  auto *cl = open_client(&state[0]);
  return cl->rpc_invoke<false>(f, u);
#else
  (void)f;
  (void)u;
  return false;
#endif
}

bool x64_amdgcn_t::server_t::handle_impl(void *application_state, uint64_t *l)
{
#if defined(__x86_64__)
  auto *se = open_server(&state[0]);
  return se->rpc_handle(application_state, l);
#else
  (void)application_state;
  (void)l;
  return false;
#endif
}

}  // namespace hostrpc

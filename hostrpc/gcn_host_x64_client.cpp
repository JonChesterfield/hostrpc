#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "test_common.hpp"

#if defined(__x86_64__)
#include "hsa.h"
static void copy_page(hostrpc::page_t *dst, hostrpc::page_t *src)
{
  __builtin_memcpy(dst, src, sizeof(hostrpc::page_t));
}
#endif

struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__x86_64__)
    hostrpc::page_t *d = static_cast<hostrpc::page_t *>(dv);
    copy_page(page, d);
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
#if defined(__x86_64__)
    hostrpc::page_t *d = static_cast<hostrpc::page_t *>(dv);
    copy_page(d, page);
#else
    (void)page;
    (void)dv;
#endif
  };
};

#if defined(__AMDGCN__)
static void gcn_server_callback(hostrpc::cacheline_t *)
{
  // not yet implemented, maybe take a function pointer out of [0]
}
#endif

struct operate
{
  static void call(hostrpc::page_t *page, void *)
  {
#if defined(__AMDGCN__)
    // Call through to a specific handler, one cache line per lane
    hostrpc::cacheline_t *l = &page->cacheline[platform::get_lane_id()];
    gcn_server_callback(l);
#else
    (void)page;
#endif
  };
};

struct clear
{
  static void call(hostrpc::page_t *, void *) {}
};

namespace hostrpc
{
template <typename SZ>
using gcn_x64_client =
    hostrpc::client_impl<SZ, hostrpc::copy_functor_given_alias, fill, use,
                         hostrpc::nop_stepper>;

template <typename SZ>
using gcn_x64_server =
    hostrpc::server_impl<SZ, hostrpc::copy_functor_given_alias, operate, clear,
                         hostrpc::nop_stepper>;

template <typename SZ>
struct gcn_x64_pair
{
  using client_type = hostrpc::gcn_x64_client<SZ>;
  using server_type = hostrpc::gcn_x64_server<SZ>;
  client_type client;
  server_type server;
  SZ sz;

  gcn_x64_pair(SZ sz, uint64_t fine_handle, uint64_t coarse_handle) : sz(sz)
  {
    // TODO: This is very similar to x64_host_gcn_client
    // Should be able to abstract over the allocation location

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

    // could be malloc here, gpu can't see the client locks
    auto *client_active_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);
    auto *client_outbox_staging_data =
        hsa_allocate_slot_bitmap_data_alloc(fine, N);

    auto *server_active_data = hsa_allocate_slot_bitmap_data_alloc(coarse, N);
    auto *server_outbox_staging_data =
        hsa_allocate_slot_bitmap_data_alloc(coarse, N);

    message_bitmap send = {send_data};
    message_bitmap recv = {recv_data};
    lock_bitmap client_active = {client_active_data};
    slot_bitmap_coarse client_outbox_staging = {client_outbox_staging_data};
    lock_bitmap server_active = {server_active_data};
    slot_bitmap_coarse server_outbox_staging = {server_outbox_staging_data};

    client = {sz,
              recv,
              send,
              client_active,
              client_outbox_staging,
              server_buffer,
              client_buffer};
    server = {sz,
              send,
              recv,
              server_active,
              server_outbox_staging,
              client_buffer,
              server_buffer};
#else
    (void)fine_handle;
    (void)coarse_handle;
#endif
  }

  ~gcn_x64_pair()
  {
#if defined(__x86_64__)
    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hsa_allocate_slot_bitmap_data_free(client.inbox.data());
    hsa_allocate_slot_bitmap_data_free(client.outbox.data());
    hsa_allocate_slot_bitmap_data_free(client.active.data());
    hsa_allocate_slot_bitmap_data_free(client.outbox_staging.data());
    hsa_allocate_slot_bitmap_data_free(server.active.data());

    // precondition of structure
    assert(client.local_buffer == server.remote_buffer);
    assert(client.remote_buffer == server.local_buffer);

    // postcondition of this instance
    assert(client.local_buffer == client.remote_buffer);
    hsa_memory_free(client.local_buffer);
#endif
  }
};

using ty = gcn_x64_pair<hostrpc::size_runtime>;

gcn_x64_t::gcn_x64_t(size_t N, uint64_t hsa_region_t_fine_handle,
                     uint64_t hsa_region_t_coarse_handle)

{
  // for gfx906, probably want N = 2048
  N = hostrpc::round(N);

  state = nullptr;
#if defined(__x86_64__)
  hostrpc::size_runtime sz(N);
  ty *s = new (std::nothrow)
      ty(sz, hsa_region_t_fine_handle, hsa_region_t_coarse_handle);
  state = static_cast<void *>(s);
#else
  (void)hsa_region_t_fine_handle;
  (void)hsa_region_t_coarse_handle;
#endif
}

gcn_x64_t::~gcn_x64_t()
{
#if defined(__x86_64__)
  ty *s = static_cast<ty *>(state);
  if (s)
    {
      // Should probably call the destructors on client/server state here
      delete s;
    }
#endif
}

bool gcn_x64_t::valid() { return state != nullptr; }

gcn_x64_t::client_t gcn_x64_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::client_type ct = s->client;
  return {ct};
}

gcn_x64_t::server_t gcn_x64_t::server()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::server_type st = s->server;
  return {st};
}

void gcn_x64_t::client_t::invoke(hostrpc::page_t *page)
{
  void *vp = static_cast<void *>(page);
  bool r = false;
  do
    {
      r = state.open<ty::client_type>()->rpc_invoke<true>(vp, vp);
    }
  while (r == false);
}

void gcn_x64_t::client_t::invoke_async(hostrpc::page_t *page)
{
  void *vp = static_cast<void *>(page);
  bool r = false;
  do
    {
      r = state.open<ty::client_type>()->rpc_invoke<false>(vp, vp);
    }
  while (r == false);
}

bool gcn_x64_t::server_t::handle(uint64_t *loc)
{
  return state.open<ty::server_type>()->rpc_handle(loc);
}

}  // namespace hostrpc

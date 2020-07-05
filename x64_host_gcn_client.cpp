#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "test_common.hpp"

#if defined(__x86_64__)
#include "hsa.h"
#endif

#if defined(__AMDGCN__)
static void copy_page(hostrpc::page_t *dst, hostrpc::page_t *src)
{
  if (true)
    {
      __builtin_memcpy(dst, src, sizeof(hostrpc::page_t));
    }
  else
    {
      for (unsigned i = 0; i < 64; i++)
        {
          for (unsigned e = 0; e < 8; e++)
            {
              dst->cacheline[i].element[e] = src->cacheline[i].element[e];
            }
        }
    }
}
#endif

struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if defined(__AMDGCN__)
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
#if defined(__AMDGCN__)
    hostrpc::page_t *d = static_cast<hostrpc::page_t *>(dv);
    copy_page(d, page);
#else
    (void)page;
    (void)dv;
#endif
  };
};

namespace hostrpc
{
template <typename SZ>
using x64_gcn_client =
    hostrpc::client_impl<SZ, hostrpc::copy_functor_given_alias, fill, use,
                         hostrpc::nop_stepper>;

template <typename SZ>
using x64_gcn_server =
    hostrpc::server_indirect_impl<SZ, hostrpc::copy_functor_given_alias,
                                  hostrpc::nop_stepper>;

template <typename SZ>
struct x64_gcn_pair
{
  using client_type = hostrpc::x64_gcn_client<SZ>;
  using server_type = hostrpc::x64_gcn_server<SZ>;
  client_type client;
  server_type server;
  SZ sz;

  x64_gcn_pair(SZ sz, uint64_t fine_handle, uint64_t coarse_handle) : sz(sz)
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
    // allocating in coarse is probably not sufficient, likely to need to mark
    // the pointer with an address space
    auto *client_active_data = hsa_allocate_slot_bitmap_data_alloc(coarse, N);
    auto *client_outbox_staging_data =
        hsa_allocate_slot_bitmap_data_alloc(coarse, N);

    // server_active could be 'malloc', gcn can't access it
    auto *server_active_data = hsa_allocate_slot_bitmap_data_alloc(fine, N);

    slot_bitmap_all_svm send = {N, send_data};
    slot_bitmap_all_svm recv = {N, recv_data};
    slot_bitmap_device client_active = {N, client_active_data};
    slot_bitmap_device client_outbox_staging = {N, client_outbox_staging_data};
    slot_bitmap_device server_active = {N, server_active_data};

    client = {sz,
              recv,
              send,
              client_active,
              client_outbox_staging,
              server_buffer,
              client_buffer};

    server = {sz, send, recv, server_active, client_buffer, server_buffer};
#else
    (void)fine_handle;
    (void)coarse_handle;
#endif
  }

  ~x64_gcn_pair()
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

using ty = x64_gcn_pair<hostrpc::size_runtime>;

x64_gcn_t::x64_gcn_t(size_t N, uint64_t hsa_region_t_fine_handle,
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

x64_gcn_t::~x64_gcn_t()
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

bool x64_gcn_t::valid() { return state != nullptr; }

x64_gcn_t::client_t x64_gcn_t::client()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::client_type ct = s->client;
  return {ct};
}

x64_gcn_t::server_t x64_gcn_t::server()
{
  ty *s = static_cast<ty *>(state);
  assert(s);
  ty::server_type st = s->server;
  return {st};
}

// The boolean is uniform, but seem to be seeing some control flow problems
// in the caller. Forcing to a scalar with broadcast_master is ineffective.
// Simplifying by doing the loop-until-available here
void x64_gcn_t::client_t::invoke(hostrpc::page_t *page)
{
  void *vp = static_cast<void *>(page);
  bool r = false;
  do
    {
      r = state.open<ty::client_type>()->rpc_invoke<true>(vp, vp);
    }
  while (r == false);
}

void x64_gcn_t::client_t::invoke_async(hostrpc::page_t *page)
{
  void *vp = static_cast<void *>(page);
  bool r = false;
  do
    {
      r = state.open<ty::client_type>()->rpc_invoke<false>(vp, vp);
    }
  while (r == false);
}

bool x64_gcn_t::server_t::handle(hostrpc::closure_func_t func,
                                 void *application_state, uint64_t *l)
{
  return handle<ty::server_type>(func, application_state, l);
}

}  // namespace hostrpc

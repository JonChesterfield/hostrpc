#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"

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
  using client_type = hostrpc::x64_gcn_client<SZ>;
  using server_type = hostrpc::x64_gcn_server<SZ>;
  client_type client;
  server_type server;
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
    // allocating in coarse is probably not sufficient, likely to need to mark
    // the pointer with an address space
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

static constexpr size_t round(size_t x) { return 64u * ((x + 63u) / 64u); }

using ty = x64_amdgcn_pair<hostrpc::size_runtime>;

x64_gcn_t::x64_gcn_t(size_t N, uint64_t hsa_region_t_fine_handle,
                     uint64_t hsa_region_t_coarse_handle)
{
  // for gfx906, probably want N = 2048
  N = round(N);

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

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "test_common.hpp"

#include "gcn_host_x64_client.hpp"

namespace hostrpc
{
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

using ty = gcn_x64_pair_T<hostrpc::size_runtime, fill, use, operate, clear>;

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

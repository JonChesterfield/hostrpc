#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "interface.hpp"
#include "memory.hpp"
#include "test_common.hpp"
#include "x64_host_gcn_client.hpp"

#if defined(__x86_64__)
#include "hsa.h"
#endif

#if defined(__AMDGCN__)
static void copy_page(hostrpc::page_t *dst, hostrpc::page_t *src)
{
  unsigned id = platform::get_lane_id();
  hostrpc::cacheline_t *dline = &dst->cacheline[id];
  hostrpc::cacheline_t *sline = &src->cacheline[id];
  for (unsigned e = 0; e < 8; e++)
    {
      dline->element[e] = sline->element[e];
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

using ty = x64_gcn_pair_T<hostrpc::size_runtime, fill ,use, indirect::operate, indirect::clear>;
                                    

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

hostrpc::client_counters x64_gcn_t::client_t::get_counters()
{
  return state.open<ty::client_type>()->get_counters();
}

bool x64_gcn_t::server_t::handle(hostrpc::closure_func_t func,
                                 void *application_state, uint64_t *l)
{
  return handle<ty::server_type>(func, application_state, l);
}

}  // namespace hostrpc

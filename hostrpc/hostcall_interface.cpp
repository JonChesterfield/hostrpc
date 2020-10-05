#include "hostcall_interface.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform.hpp"
#include "detail/server_impl.hpp"
#include "hostcall.hpp"  // hostcall_ops prototypes
#include "memory.hpp"
#include "test_common.hpp"

#include "x64_host_gcn_client.hpp"

// Glue the opaque hostcall_interface class onto the freestanding implementation

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

template <typename SZ>
using x64_amdgcn_pair = hostrpc::x64_gcn_pair_T<
    SZ, x64_host_amdgcn_client::fill, x64_host_amdgcn_client::use,
    x64_host_amdgcn_client::operate, x64_host_amdgcn_client::clear,
    counters::client_nop, counters::server_nop>;

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;
using ty = x64_amdgcn_pair<SZ>;

hostcall_interface_t::hostcall_interface_t(uint64_t hsa_region_t_fine_handle,
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

hostcall_interface_t::~hostcall_interface_t()
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

bool hostcall_interface_t::valid() { return state != nullptr; }

hostcall_interface_t::client_t hostcall_interface_t::client()
{
  using res_t = hostcall_interface_t::client_t;
  static_equal<res_t::state_t::size(), sizeof(ty::client_type)>();
  static_assert(res_t::state_t::size() == sizeof(ty::client_type), "");
  static_assert(res_t::state_t::align() == alignof(ty::client_type), "");

  ty *s = static_cast<ty *>(state);
  assert(s);
  res_t res;
  auto *cl = res.state.construct<ty::client_type>(s->client);
  (void)cl;
  assert(cl == res.state.open<ty::client_type>());
  return res;
}

hostcall_interface_t::server_t hostcall_interface_t::server()
{
  // Construct an opaque server_t into the aligned state field
  using res_t = hostcall_interface_t::server_t;
  static_equal<res_t::state_t::size(), sizeof(ty::server_type)>();
  static_assert(res_t::state_t::align() == alignof(ty::server_type), "");

  ty *s = static_cast<ty *>(state);
  assert(s);
  res_t res;
  auto *sv = res.state.construct<ty::server_type>(s->server);
  (void)sv;
  assert(sv == res.state.open<ty::server_type>());
  return res;
}

void hostcall_interface_t::client_t::dump()
{
  auto *cl = state.open<ty::client_type>();
  cl->dump();
}

bool hostcall_interface_t::client_t::invoke_impl(void *f, void *u)
{
#if defined(__AMDGCN__)
  auto *cl = state.open<ty::client_type>();
  return cl->rpc_invoke<true>(f, u);
#else
  (void)f;
  (void)u;
  return false;
#endif
}

bool hostcall_interface_t::client_t::invoke_async_impl(void *f, void *u)
{
#if defined(__AMDGCN__)
  auto *cl = state.open<ty::client_type>();
  return cl->rpc_invoke<true>(f, u);
#else
  (void)f;
  (void)u;
  return false;
#endif
}

bool hostcall_interface_t::server_t::handle_impl(void *application_state,
                                                 uint32_t *l)
{
#if defined(__x86_64__)
  auto *se = state.open<ty::server_type>();
  return se->rpc_handle(application_state, l);
#else
  (void)application_state;
  (void)l;
  return false;
#endif
}

}  // namespace hostrpc

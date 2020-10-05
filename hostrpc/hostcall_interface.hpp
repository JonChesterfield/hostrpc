#ifndef HOSTCALL_INTERFACE_HPP_INCLUDED
#define HOSTCALL_INTERFACE_HPP_INCLUDED

#include "base_types.hpp"
#include "hostcall.hpp"  // hostcall_ops prototype
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

template <typename SZ>
using x64_amdgcn_pair = hostrpc::x64_gcn_pair_T<
    SZ, x64_host_amdgcn_client::fill, x64_host_amdgcn_client::use,
    x64_host_amdgcn_client::operate, x64_host_amdgcn_client::clear,
    counters::client_nop, counters::server_nop>;

struct hostcall_interface_t
{
  x64_amdgcn_pair<size_compiletime<hostrpc::x64_host_amdgcn_array_size>>
      instance;

  hostcall_interface_t(uint64_t hsa_region_t_fine_handle,
                       uint64_t hsa_region_t_coarse_handle)
      : instance(size_compiletime<hostrpc::x64_host_amdgcn_array_size>{},
                 hsa_region_t_fine_handle, hsa_region_t_coarse_handle)
  {
  }

  using client_type = decltype(instance)::client_type;
  using server_type = decltype(instance)::server_type;

  ~hostcall_interface_t() {}
  hostcall_interface_t(const hostcall_interface_t &) = delete;
  bool valid() { return true; }

  template <bool have_continuation>
  bool rpc_invoke(void *fill, void *use) noexcept
  {
    return instance.client.rpc_invoke<have_continuation>(fill, use);
  }

  bool rpc_handle(void *operate_state, void *clear_state,
                  uint32_t *location_arg) noexcept
  {
    return instance.server.rpc_handle(operate_state, clear_state, location_arg);
  }

  client_counters client_counters() { return instance.client.get_counters(); }
  server_counters server_counters() { return instance.server.get_counters(); }
};

}  // namespace hostrpc

#endif

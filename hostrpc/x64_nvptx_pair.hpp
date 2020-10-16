#ifndef X64_NVPTX_PAIR_HPP_INCLUDED
#define X64_NVPTX_PAIR_HPP_INCLUDED

#include "base_types.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform_detect.h"
#include "detail/server_impl.hpp"
#include "hostcall.hpp"
#include "x64_host_ptx_client.hpp"

namespace hostrpc
{
namespace x64_host_nvptx_client
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if (HOSTRPC_NVPTX)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::pass_arguments(page, d);
#endif
#if (HOSTRPC_HOST)
    (void)page;
    (void)dv;
#endif
  };
};

struct use
{
  static void call(hostrpc::page_t *page, void *dv)
  {
#if (HOSTRPC_NVPTX)
    uint64_t *d = static_cast<uint64_t *>(dv);
    hostcall_ops::use_result(page, d);
#endif
#if (HOSTRPC_HOST)
    (void)page;
    (void)dv;
#endif
  };
};

struct operate
{
  static void call(hostrpc::page_t *page, void *)
  {
#if (HOSTRPC_HOST)
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
#if (HOSTRPC_HOST)
    hostcall_ops::clear(page);
#else
    (void)page;
#endif
  }
};
}  // namespace x64_host_nvptx_client

using x64_nvptx_pair = hostrpc::x64_ptx_pair_T<
    hostrpc::size_runtime, x64_host_nvptx_client::fill,
    x64_host_nvptx_client::use, x64_host_nvptx_client::operate,
    x64_host_nvptx_client::clear, counters::client_nop, counters::server_nop>;

}  // namespace hostrpc

#endif

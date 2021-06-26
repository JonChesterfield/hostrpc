#ifndef HOSTRPC_X64_GCN_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_GCN_TYPE_HPP_INCLUDED

#include "base_types.hpp"

#include "allocator.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform_detect.hpp"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
template <typename SZ>
using x64_gcn_type_base =
    client_server_pair_t<SZ, uint64_t, allocator::hsa<alignof(page_t)>,
                         allocator::hsa<64>, allocator::host_libc<64>,
                         allocator::hsa<64>>;

template <typename SZ>
struct x64_gcn_type : public x64_gcn_type_base<SZ>
{
  using base = x64_gcn_type_base<SZ>;
  HOSTRPC_ANNOTATE x64_gcn_type(SZ sz, uint64_t fine_handle,
                                uint64_t coarse_handle)
      : base(sz, typename base::AllocBuffer(fine_handle),
             typename base::AllocInboxOutbox(fine_handle),
             typename base::AllocLocal(),
             typename base::AllocRemote(coarse_handle))
  {
  }
  x64_gcn_type() = default;
};

}  // namespace hostrpc

#endif

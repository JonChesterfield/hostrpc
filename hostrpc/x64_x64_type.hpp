#ifndef HOSTRPC_X64_X64_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_X64_TYPE_HPP_INCLUDED

#include "allocator.hpp"
#include "base_types.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "platform/detect.hpp"

namespace hostrpc
{
template <typename SZ>
using x64_x64_type_base = client_server_pair_t<
    SZ, uint64_t, hostrpc::allocator::host_libc<alignof(page_t)>,
    hostrpc::allocator::host_libc<64>, hostrpc::allocator::host_libc<64>,
    hostrpc::allocator::host_libc<64> >;

template <typename SZ>
struct x64_x64_type : public x64_x64_type_base<SZ>
{
  using base = x64_x64_type_base<SZ>;
  HOSTRPC_ANNOTATE x64_x64_type(SZ sz)
      : base(sz, typename base::AllocBuffer(),
             typename base::AllocInboxOutbox(), typename base::AllocLocal(),
             typename base::AllocRemote())
  {
  }
};

}  // namespace hostrpc

#endif

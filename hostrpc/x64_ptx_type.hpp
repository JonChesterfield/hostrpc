#ifndef HOSTRPC_X64_PTX_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_PTX_TYPE_HPP_INCLUDED

#include "base_types.hpp"

#include "allocator.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"
#include "platform/detect.hpp"

namespace hostrpc
{
template <typename SZ>
using x64_ptx_type_base = client_server_pair_t<
    SZ, uint32_t, hostrpc::allocator::cuda_shared<alignof(page_t)>,
    hostrpc::allocator::cuda_shared<64>, hostrpc::allocator::host_libc<64>,
    hostrpc::allocator::cuda_gpu<64>>;

template <typename SZ>
struct x64_ptx_type : public x64_ptx_type_base<SZ>
{
  using base = x64_ptx_type_base<SZ>;
  HOSTRPC_ANNOTATE x64_ptx_type(SZ sz)
      : base(sz, typename base::AllocBuffer(),
             typename base::AllocInboxOutbox(), typename base::AllocLocal(),
             typename base::AllocRemote())
  {
  }
};

}  // namespace hostrpc

#endif

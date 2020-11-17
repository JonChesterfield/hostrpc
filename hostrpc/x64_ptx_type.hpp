#ifndef HOSTRPC_X64_PTX_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_PTX_TYPE_HPP_INCLUDED

#include "base_types.hpp"

#include "allocator.hpp"
#include "client_server_pair.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform_detect.h"
#include "detail/server_impl.hpp"
#include "host_client.hpp"

namespace hostrpc
{
using x64_ptx_type_base = client_server_pair_t<
    hostrpc::size_runtime, copy_functor_given_alias, uint32_t,
    hostrpc::allocator::cuda_shared<alignof(page_t)>,
    hostrpc::allocator::cuda_shared<64>, hostrpc::allocator::host_libc<64>,
    hostrpc::allocator::cuda_gpu<64>>;

struct x64_ptx_type : public x64_ptx_type_base
{
  using base = x64_ptx_type_base;
  HOSTRPC_ANNOTATE x64_ptx_type(size_t N)
      : x64_ptx_type_base(
            hostrpc::size_runtime(N), typename base::AllocBuffer(),
            typename base::AllocInboxOutbox(), typename base::AllocLocal(),
            typename base::AllocRemote())
  {
  }
};

}  // namespace hostrpc

#endif

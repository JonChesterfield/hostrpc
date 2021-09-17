#ifndef HOSTRPC_PLATFORM_UTILS_HPP_INCLUDED
#define HOSTRPC_PLATFORM_UTILS_HPP_INCLUDED

#include "platform.hpp"
#include "platform_detect.hpp"

namespace platform
{
HOSTRPC_ANNOTATE inline uint32_t reduction_sum(uint32_t);

#if HOSTRPC_HOST
HOSTRPC_ANNOTATE inline uint32_t reduction_sum(uint32_t x) { return x; }
#endif

namespace detail
{
#if HOSTRPC_AMDGCN
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline int32_t
__impl_shfl_down_sync(int32_t var, uint32_t laneDelta)
{
  // derived from openmp runtime
  int32_t width = 64;
  int self = get_lane_id();
  int index = self + laneDelta;
  index = (int)(laneDelta + (self & (width - 1))) >= width ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}
#endif

#if HOSTRPC_NVPTX

HOSTRPC_ANNOTATE
inline int32_t __impl_shfl_down_sync(int32_t var, uint32_t laneDelta)
{
  enum
  {
    warpsize = 32,
  };

  // danger: Probably want something more like:
  // return __nvvm_shfl_sync_down_i32(Mask, Var, Delta, (( warpsize - Width) <<
  // 8) | 0x1f);
  return __nvvm_shfl_sync_down_i32(UINT32_MAX, var, laneDelta, warpsize - 1);
}
#endif
}  // namespace detail

#if (HOSTRPC_AMDGCN || HOSTRPC_NVPTX)
HOSTRPC_ANNOTATE
__attribute__((always_inline)) inline uint32_t reduction_sum(uint32_t x)
{
  x += detail::__impl_shfl_down_sync(x, 32);
  x += detail::__impl_shfl_down_sync(x, 16);
  x += detail::__impl_shfl_down_sync(x, 8);
  x += detail::__impl_shfl_down_sync(x, 4);
  x += detail::__impl_shfl_down_sync(x, 2);
  x += detail::__impl_shfl_down_sync(x, 1);
  return x;
}
#endif

}  // namespace platform

#endif

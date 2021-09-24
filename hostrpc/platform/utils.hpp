#ifndef HOSTRPC_PLATFORM_UTILS_HPP_INCLUDED
#define HOSTRPC_PLATFORM_UTILS_HPP_INCLUDED

#include "../platform.hpp"

namespace platform
{
template <typename T>
HOSTRPC_ANNOTATE inline uint32_t reduction_sum(T active_threads, uint32_t);

#if HOSTRPC_HOST
template <typename T>
HOSTRPC_ANNOTATE inline uint32_t reduction_sum(T, uint32_t x)
{
  return x;
}
#endif

namespace detail
{
#if HOSTRPC_AMDGCN
template <typename T>
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline int32_t
__impl_shfl_down_sync(T, int32_t var, uint32_t laneDelta)
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
template <typename T>
inline int32_t __impl_shfl_down_sync(T active_threads, int32_t var,
                                     uint32_t laneDelta)
{
  return __nvvm_shfl_sync_down_i32(active_threads, var, laneDelta,
                                   desc::native_width() - 1);
}
#endif
}  // namespace detail

#if (HOSTRPC_AMDGCN || HOSTRPC_NVPTX)
template <typename T>
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline uint32_t reduction_sum(
    T active_threads, uint32_t x)
{
  x += detail::__impl_shfl_down_sync(active_threads, x, 32);
  x += detail::__impl_shfl_down_sync(active_threads, x, 16);
  x += detail::__impl_shfl_down_sync(active_threads, x, 8);
  x += detail::__impl_shfl_down_sync(active_threads, x, 4);
  x += detail::__impl_shfl_down_sync(active_threads, x, 2);
  x += detail::__impl_shfl_down_sync(active_threads, x, 1);
  return x;
}
#endif

}  // namespace platform

#endif

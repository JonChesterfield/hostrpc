#include <cuda.h>

// Intent is to use the cuda calls initially, then transform to clang intrinsics
// and move into platform.hpp

#define DEVICE __device__

#define WARPSIZE 32
DEVICE uint32_t get_lane_id(void) { return threadIdx.x & (WARPSIZE - 1); }

namespace detail
{
static DEVICE uint32_t ballot()
{
#if CUDA_VERSION >= 9000
  return __activemask();
#else
  return __ballot(1);
#endif
}

DEVICE uint32_t get_master_lane_id(void)
{
  uint32_t activemask = ballot();

  uint32_t lowest_active = __builtin_ffs(activemask) - 1;
  uint32_t lane_id = get_lane_id();

  return lane_id == lowest_active;

  // TODO: openmp deviceRTL uses:
  // return (blockDim.x - 1) & ~(WARPSIZE - 1);
}

DEVICE int32_t __impl_shfl_down_sync(int32_t var, uint32_t laneDelta)
{
  return __shfl_down_sync(UINT32_MAX, var, laneDelta, WARPSIZE);
}

}  // namespace detail

DEVICE uint32_t broadcast_master(uint32_t x)
{
  uint32_t master_id = detail::get_master_lane_id();
  // __nvvm_shfl_sync_idx_i32(UINT32_MAX, x, master_id, 31)
#if CUDA_VERSION >= 9000
  // Use activemask?
  return __shfl_sync(UINT32_MAX, x, master_id);
#else
  // This may be UB if some lanes are inactive
  return __shfl(x, master_id);
#endif
}

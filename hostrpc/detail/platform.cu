#include <cuda.h>

// Intent is to use the cuda calls initially, then transform to clang intrinsics
// and move into platform.hpp

#define DEVICE __device__ __attribute__((always_inline))

#define WARPSIZE 32

namespace platform
{
DEVICE uint32_t get_lane_id(void)
{
  return __nvvm_read_ptx_sreg_tid_x() /*threadIdx.x*/ & (WARPSIZE - 1);
}

#ifndef CUDA_VERSION
#error "Require CUDA_VERSION definition"
#endif

// Something strange here. CUDA_VERSION picks activemask, but
// sm_50 maps onto ptx 4.0 by default which doesn't support that
// Compiling with cuda overrides to ptx 6.3, passing Xclang to match.

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

// TODO: Check the differences between threadfence, threadfence_block,
// threadfence_system
DEVICE void fence_acquire() { __threadfence_system(); }
DEVICE void fence_release() { __threadfence_system(); }

namespace detail
{
// Might be able to use volatile _Atomic as the top level type if
// opencl load/store is compiled correctly when called from cuda

#define HOSTRPC_STAMP_MEMORY(TYPE)                                            \
  DEVICE TYPE atomic_load_relaxed(volatile _Atomic(TYPE) const *addr)         \
  {                                                                           \
    return __opencl_atomic_load(addr, __ATOMIC_RELAXED,                       \
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);       \
  }                                                                           \
                                                                              \
  DEVICE void atomic_store_relaxed(volatile _Atomic(TYPE) * addr, TYPE value) \
  {                                                                           \
    return __opencl_atomic_store(addr, value, __ATOMIC_RELAXED,               \
                                 __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);      \
  }

#define HOSTRPC_STAMP_FETCH(TYPE, NAME)                                   \
  DEVICE TYPE atomic_##NAME##_relaxed(volatile _Atomic(TYPE) * addr,      \
                                      TYPE value)                         \
  {                                                                       \
    return __opencl_atomic_##NAME(addr, value, __ATOMIC_RELAXED,          \
                                  __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES); \
  }

#define HOSTRPC_STAMP_FETCH_OPS(TYPE)                                 \
  HOSTRPC_STAMP_FETCH(TYPE, fetch_add)                                \
  HOSTRPC_STAMP_FETCH(TYPE, fetch_and)                                \
  HOSTRPC_STAMP_FETCH(TYPE, fetch_or)                                 \
  DEVICE bool atomic_compare_exchange_weak_relaxed(                   \
      volatile _Atomic(TYPE) * addr, TYPE expected, TYPE desired,     \
      TYPE * loaded)                                                  \
  {                                                                   \
    bool r = __opencl_atomic_compare_exchange_weak(                   \
        addr, &expected, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, \
        __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);                       \
    *loaded = expected;                                               \
    return r;                                                         \
  }

HOSTRPC_STAMP_MEMORY(uint8_t)
HOSTRPC_STAMP_MEMORY(uint16_t)
HOSTRPC_STAMP_MEMORY(uint32_t)
HOSTRPC_STAMP_MEMORY(uint64_t)

HOSTRPC_STAMP_FETCH_OPS(uint32_t)
HOSTRPC_STAMP_FETCH_OPS(uint64_t)

#undef HOSTRPC_STAMP_MEMORY
#undef HOSTRPC_STAMP_FETCH
#undef HOSTRPC_STAMP_FETCH_OPS

}  // namespace detail

}  // namespace platform

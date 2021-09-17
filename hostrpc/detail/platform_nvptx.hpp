#ifndef HOSTRPC_PLATFORM_NVPTX_HPP_INCLUDED
#define HOSTRPC_PLATFORM_NVPTX_HPP_INCLUDED

#ifndef HOSTRPC_PLATFORM_HPP_INCLUDED
#error "Expected to be included within platform.hpp"
#endif

#if !HOSTRPC_NVPTX
#error "Expected HOSTRPC_NVPTX"
#endif

namespace platform
{
namespace detail
{
enum
{
  warpsize = 32,
};

inline HOSTRPC_ANNOTATE uint32_t get_master_lane_id(void)
{
  uint32_t activemask;
  asm volatile("activemask.b32 %0;" : "=r"(activemask));

  uint32_t lowest_active = __builtin_ffs(activemask) - 1;
  return lowest_active;
}

// TODO: Check the differences between threadfence, threadfence_block,
// threadfence_system
static HOSTRPC_ANNOTATE void fence_acquire_release() { __nvvm_membar_sys(); }

}  // namespace detail

// todo
HOSTRPC_ANNOTATE
inline void sleep_briefly() {}
HOSTRPC_ANNOTATE
inline void sleep() { sleep_briefly(); }

HOSTRPC_ANNOTATE
inline uint32_t get_lane_id()
{
  return __nvvm_read_ptx_sreg_tid_x() /*threadIdx.x*/ & (detail::warpsize - 1);
}

HOSTRPC_ANNOTATE
inline bool is_master_lane()
{
  return get_lane_id() == detail::get_master_lane_id();
}

HOSTRPC_ANNOTATE
inline uint32_t broadcast_master(uint32_t x)
{
  uint32_t master_id = detail::get_master_lane_id();
  return __nvvm_shfl_sync_idx_i32(UINT32_MAX, x, master_id,
                                  detail::warpsize - 1);
}

// todo: smid based
HOSTRPC_ANNOTATE inline uint32_t client_start_slot() { return 0; }

HOSTRPC_ANNOTATE
void fence_acquire() { detail::fence_acquire_release(); }

HOSTRPC_ANNOTATE
void fence_release() { detail::fence_acquire_release(); }

}  // namespace platform

#endif

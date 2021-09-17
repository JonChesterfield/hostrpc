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
namespace
{
HOSTRPC_ANNOTATE constexpr uint64_t desc::native_width() { return 32; }

HOSTRPC_ANNOTATE uint64_t desc::active_threads()
{
  uint32_t activemask;
  asm volatile("activemask.b32 %0;" : "=r"(activemask));
  return activemask;
}

namespace detail
{
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
  return __nvvm_read_ptx_sreg_tid_x() /*threadIdx.x*/ &
         (desc::native_width() - 1);
}

template <typename T>
inline HOSTRPC_ANNOTATE uint32_t get_master_lane_id(T active_threads)
{
  auto f = active_threads.findFirstSet();
  return f.template subtract<1>();
}

template <typename T>
HOSTRPC_ANNOTATE inline uint32_t broadcast_master(T active_threads, uint32_t x)
{
  uint32_t master_id = get_master_lane_id(active_threads);
  return __nvvm_shfl_sync_idx_i32(UINT32_MAX, x, master_id,
                                  desc::native_width() - 1);
}

// todo: smid based
HOSTRPC_ANNOTATE inline uint32_t client_start_slot() { return 0; }

HOSTRPC_ANNOTATE
void fence_acquire() { detail::fence_acquire_release(); }

HOSTRPC_ANNOTATE
void fence_release() { detail::fence_acquire_release(); }

}  // namespace
}  // namespace platform

#endif

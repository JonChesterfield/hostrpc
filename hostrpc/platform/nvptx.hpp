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
inline HOSTRPC_ANNOTATE constexpr uint64_t native_width() { return 32; }

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

inline HOSTRPC_ANNOTATE auto get_lane_id()
{
  hostrpc::fastint_runtime<uint32_t> r =
      __nvvm_read_ptx_sreg_tid_x() /*threadIdx.x*/ & (native_width() - 1);
  return r;
}

inline HOSTRPC_ANNOTATE auto active_threads()
{
  uint32_t activemask;
  asm volatile("activemask.b32 %0;" : "=r"(activemask));
  hostrpc::fastint_runtime<uint32_t> r = activemask;
  return r;
}

template <typename T>
HOSTRPC_ANNOTATE inline uint32_t broadcast_master(T active_threads, uint32_t x)
{
  uint32_t master_id = platform::get_master_lane_id(active_threads);
  return __nvvm_shfl_sync_idx_i32(active_threads, x, master_id,
                                  native_width() - 1);
}

// todo: smid based
HOSTRPC_ANNOTATE inline uint32_t client_start_slot() { return 0; }

HOSTRPC_ANNOTATE
void fence_acquire() { detail::fence_acquire_release(); }

HOSTRPC_ANNOTATE
void fence_release() { detail::fence_acquire_release(); }

inline HOSTRPC_ANNOTATE auto all_threads_active_constant()
{
  return hostrpc::fastint_compiletime<UINT32_MAX>();
}

}  // namespace
}  // namespace platform

#endif

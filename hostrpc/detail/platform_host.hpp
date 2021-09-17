#ifndef HOSTRPC_PLATFORM_HOST_HPP_INCLUDED
#define HOSTRPC_PLATFORM_HOST_HPP_INCLUDED

#ifndef HOSTRPC_PLATFORM_HPP_INCLUDED
#error "Expected to be included within platform.hpp"
#endif

#if !HOSTRPC_HOST
#error "Expected HOSTRPC_HOST"
#endif

// should probably implement this in platform::host_libc source
extern "C" int usleep(unsigned);  // #include <unistd.h>

namespace platform
{
namespace
{
inline HOSTRPC_ANNOTATE constexpr uint64_t native_width() { return 1; }

// local toolchain thinks usleep might throw. That induces a bunch of exception
// control flow where there otherwise wouldn't be any. Will fix by calling into
// std::chrono, bodge for now
namespace detail
{
HOSTRPC_ANNOTATE static __attribute__((noinline)) void sleep_noexcept(
    unsigned int t) noexcept
{
#if !defined(__OPENCL_C_VERSION__)
  usleep(t);
#else
  (void)t;
#endif
}
}  // namespace detail

HOSTRPC_ANNOTATE inline void sleep_briefly()
{
  // <thread> conflicts with <stdatomic.h>
  // stdatomic is no longer in use so may be able to use <thread> again
  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  detail::sleep_noexcept(10);
}

HOSTRPC_ANNOTATE inline void sleep() { detail::sleep_noexcept(1000); }

inline HOSTRPC_ANNOTATE auto active_threads()
{
  return hostrpc::fastint_compiletime<1>();
}

inline HOSTRPC_ANNOTATE auto get_lane_id()
{
  return hostrpc::fastint_compiletime<0>();
}

template <typename T>
HOSTRPC_ANNOTATE inline auto get_master_lane_id(T)
{
  return hostrpc::fastint_compiletime<0>();
}

template <typename T>
HOSTRPC_ANNOTATE inline uint32_t broadcast_master(T, uint32_t x)
{
  return x;
}

HOSTRPC_ANNOTATE inline uint32_t client_start_slot() { return 0; }

HOSTRPC_ANNOTATE inline void fence_acquire()
{
  __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
}
HOSTRPC_ANNOTATE inline void fence_release()
{
  __c11_atomic_thread_fence(__ATOMIC_RELEASE);
}

}  // namespace
}  // namespace platform

#endif

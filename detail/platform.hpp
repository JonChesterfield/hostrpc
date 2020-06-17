#ifndef PLATFORM_HPP_INCLUDED
#define PLATFORM_HPP_INCLUDED

#include <stdint.h>

namespace platform
{
void sleep_briefly(void);
}

#if defined(__x86_64__)
#include <chrono>
#include <unistd.h>

#include <cassert>
#include <cstdio>

namespace platform
{
// local toolchain thinks usleep might throw. That induces a bunch of exception
// control flow where there otherwise wouldn't be any. Will fix by calling into
// std::chrono, bodge for now
static __attribute__((noinline)) void sleep_noexcept(unsigned int t) noexcept
{
  usleep(t);
}

inline void sleep_briefly(void)
{
  // <thread> conflicts with <stdatomic.h>
  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  sleep_noexcept(10);
}
inline void sleep(void) { sleep_noexcept(1000); }

inline bool is_master_lane(void) { return true; }
inline uint32_t get_lane_id(void) { return 0; }
inline uint32_t broadcast_master(uint32_t x) { return x; }
inline uint64_t broadcast_master(uint64_t x) { return x; }

}  // namespace platform
#endif

#if defined(__AMDGCN__) || defined(__CUDACC__)
// Enough of assert.h, derived from musl
#ifdef NDEBUG
#define assert(x) (void)0
#else
#define assert_str(x) assert_str_1(x)
#define assert_str_1(x) #x
#define assert(x)                                                           \
  ((void)((x) || (__assert_fail("L:" assert_str(__LINE__) " " #x, __FILE__, \
                                __LINE__, __func__),                        \
                  0)))
#endif

#undef static_assert
#define static_assert _Static_assert

__attribute__((always_inline)) inline void __assert_fail(const char *str,
                                                         const char *,
                                                         unsigned int line,
                                                         const char *)
{
  asm("// Assert fail " ::"r"(line), "r"(str));
  __builtin_trap();
}

// stub printf for now
__attribute__((always_inline)) inline int printf(const char *, ...)
{
  // printf is implement with hostcall, so going to have to do without
  return 0;
}

#endif

#if defined(__AMDGCN__)

namespace platform
{
inline void sleep_briefly(void) { __builtin_amdgcn_s_sleep(0); }
inline void sleep(void) { __builtin_amdgcn_s_sleep(100); }

__attribute__((always_inline)) inline uint32_t get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
__attribute__((always_inline)) inline bool is_master_lane(void)
{
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  // TODO: ffs is lifted from openmp runtime, looks like it should be ctz
  uint32_t lowest_active = __builtin_ffsl(activemask) - 1;
  uint32_t lane_id = get_lane_id();

  // TODO: readfirstlane(lane_id) == lowest_active?
  return lane_id == lowest_active;
}

__attribute__((always_inline)) inline uint32_t broadcast_master(uint32_t x)
{
  return __builtin_amdgcn_readfirstlane(x);
}

__attribute__((always_inline)) inline uint64_t broadcast_master(uint64_t x)
{
  uint32_t lo = x;
  uint32_t hi = x >> 32u;
  lo = broadcast_master(lo);
  hi = broadcast_master(hi);
  return ((uint64_t)hi << 32u) | lo;
}

}  // namespace platform
#endif

#if defined(__CUDACC__)

namespace platform
{
inline void sleep_briefly(void) {}
inline void sleep(void) {}

namespace detail
{
__attribute__((always_inline)) inline uint32_t ballot()
{
#if CUDA_VERSION >= 9000
  return __activemask();
#else
  return 0;  // __ballot undeclared, definitely can't include cuda.h though
             // return __ballot(1);
#endif
}

__attribute__((always_inline)) inline uint32_t get_master_lane_id(void)
{
  return 0;  // todo
}

}  // namespace detail
__attribute__((always_inline)) inline uint32_t get_lane_id(void)
{
  // uses threadIdx.x & 0x1f from cuda, need to find the corresponding intrinsic
  return 0;
}

__attribute__((always_inline)) inline bool is_master_lane(void)
{
  return get_lane_id() == detail::get_master_lane_id();
}

__attribute__((always_inline)) inline uint32_t broadcast_master(uint32_t x)
{
  // involves __shfl_sync
  uint32_t lane_id = get_lane_id();
  uint32_t master_id = detail::get_master_lane_id();
  uint32_t v;
  if (lane_id == master_id)
    {
      v = x;
    }
  // shfl_sync isn't declared either
  // v = __shfl_sync(UINT32_MAX, v, master_id);
  return v;
}

__attribute__((always_inline)) inline uint64_t broadcast_master(uint64_t x)
{
  // probably don't want to model 64 wide warps on nvptx
  return x;
}
}  // namespace platform
#endif

namespace platform
{
template <typename U, typename F>
U critical(F f)
{
  U res = {};
  if (is_master_lane())
    {
      res = f();
    }
  res = broadcast_master(res);
  return res;
}

}  // namespace platform

#endif
#ifndef PLATFORM_HPP_INCLUDED
#define PLATFORM_HPP_INCLUDED

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
inline void sleep_briefly(void)
{
  // <thread> conflicts with <stdatomic.h>
  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  usleep(10);
}
inline void sleep(void) { usleep(1000); }

bool is_master_lane(void) { return true; }
}  // namespace platform
#endif

#if defined(__AMDGCN__)

// Enough of assert.h, derived from musl
#ifdef NDEBUG
#define assert(x) (void)0
#else
#define assert(x) \
  ((void)((x) || (__assert_fail(#x, __FILE__, __LINE__, __func__), 0)))
#endif

#undef static_assert
#define static_assert _Static_assert

__attribute__((always_inline)) inline void __assert_fail(const char *,
                                                         const char *,
                                                         unsigned int,
                                                         const char *)
{
  __builtin_trap();
}

__attribute__((always_inline)) inline int printf(const char *, ...)
{
  // printf is implement with hostcall, so going to have to do without
  return 0;
}

namespace platform
{
inline void sleep_briefly(void) { __builtin_amdgcn_s_sleep(0); }
inline void sleep(void) { __builtin_amdgcn_s_sleep(100); }


bool is_master_lane(void) {
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  uint64_t lowest_active = __builtin_ffsl(activemask) - 1; // should this be clz?
  uint32_t lane_id =  __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
  return lane_id == lowest_active;
}
  
}  // namespace platform
#endif

#endif

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

inline void __assert_fail(const char *, const char *, unsigned int,
                          const char *)
{
  __builtin_trap();
}

inline int printf(const char *, ...)
{
  // printf is implement with hostcall, so going to have to do without
  return 0;
}

namespace platform
{
inline void sleep_briefly(void) { __builtin_amdgcn_s_sleep(0); }
inline void sleep(void) { __builtin_amdgcn_s_sleep(100); }
}  // namespace platform
#endif

#endif

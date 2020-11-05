#ifndef PLATFORM_HPP_INCLUDED
#define PLATFORM_HPP_INCLUDED

#include <stdint.h>

#include "../base_types.hpp"  // page_t
#include "platform_detect.h"

#if HOSTRPC_NVPTX
#define HOSTRPC_ATOMIC(X) _Atomic(X)  // will be volatile
#else
#define HOSTRPC_ATOMIC(X) _Atomic(X)
#endif

namespace platform
{
// Functions implemented for each platform
void sleep_briefly();
void sleep();
bool is_master_lane();
uint32_t get_lane_id();
uint32_t broadcast_master(uint32_t);
uint32_t reduction_sum(uint32_t);
uint32_t client_start_slot();
void fence_acquire();
void fence_release();

template <typename T, size_t memorder, size_t scope>
T atomic_load(HOSTRPC_ATOMIC(T) const *);

template <typename T, size_t memorder, size_t scope>
void atomic_store(HOSTRPC_ATOMIC(T) *, T);

}  // namespace platform

// This is exciting. Nvptx doesn't implement atomic, so one must use volatile +
// fences C++ doesn't let one write operator new() for a volatile pointer Net
// effect is that we can't have code that conforms to the C++ object model and
// volatile qualifies the underlying object

namespace platform
{
// related functions derived from the above
uint32_t reduction_sum(uint32_t);
uint64_t broadcast_master(uint64_t);

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

namespace platform
{
namespace detail
{
template <size_t scope>
constexpr bool atomic_params_scope()
{
  return (scope == __OPENCL_MEMORY_SCOPE_DEVICE) ||
         (scope == __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
}
template <size_t memorder, size_t scope>
constexpr bool atomic_params_load()
{
  return (atomic_params_scope<scope>() &&
          ((memorder == __ATOMIC_RELAXED) || (memorder == __ATOMIC_ACQUIRE)));
}

template <size_t memorder, size_t scope>
constexpr bool atomic_params_store()
{
  return (atomic_params_scope<scope>() &&
          ((memorder == __ATOMIC_RELAXED) || (memorder == __ATOMIC_RELEASE)));
}
}  // namespace detail
}  // namespace platform

// Jury rig some pieces of libc for freestanding
// Assert is based on the implementation in musl
#if (HOSTRPC_AMDGCN || HOSTRPC_NVPTX)
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

#ifdef static_assert
#undef static_assert
#endif
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
// aomp clang currently rewrites any variadic function to a pair of
// allocate/execute functions, which don't necessarily exist.
// Clobber it with the preprocessor as a workaround.
#define printf(...) __inline_printf()
__attribute__((always_inline)) inline int __inline_printf()
{
  // printf is implement with hostcall, so going to have to do without
  return 0;
}

#endif

#if HOSTRPC_HOST
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

inline void sleep_briefly()
{
  // <thread> conflicts with <stdatomic.h>
  // stdatomic is no longer in use so may be able to use <thread> again
  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  sleep_noexcept(10);
}
inline void sleep() { sleep_noexcept(1000); }

inline bool is_master_lane() { return true; }
inline uint32_t get_lane_id() { return 0; }
inline uint32_t broadcast_master(uint32_t x) { return x; }
inline uint32_t reduction_sum(uint32_t x) { return x; }
inline uint32_t client_start_slot() { return 0; }
inline void fence_acquire() { __c11_atomic_thread_fence(__ATOMIC_ACQUIRE); }
inline void fence_release() { __c11_atomic_thread_fence(__ATOMIC_RELEASE); }

template <typename T, size_t memorder, size_t scope>
T atomic_load(HOSTRPC_ATOMIC(T) const *addr)
{
  static_assert(detail::atomic_params_load<memorder, scope>(), "");
  return __opencl_atomic_load(addr, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
void atomic_store(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(detail::atomic_params_store<memorder, scope>(), "");
  return __opencl_atomic_store(addr, value, memorder, scope);
}

}  // namespace platform
#endif

// HIP seems to want every function to be explicitly marked device or host
// even when it's only compiling for, say, device
// That may mean every single function needs a MACRO_ANNOTATION for hip to
// work as intended.
// Cuda does not appear to require this. TODO: See if openmp does

#if HOSTRPC_AMDGCN && defined(__HIP__)
#define BODGE_HIP __attribute__((device))
#else
#define BODGE_HIP
#endif

#if HOSTRPC_AMDGCN

namespace platform
{
namespace detail
{
BODGE_HIP __attribute__((always_inline)) inline int32_t __impl_shfl_down_sync(
    int32_t var, uint32_t laneDelta)
{
  // derived from openmp runtime
  int32_t width = 64;
  int self = get_lane_id();
  int index = self + laneDelta;
  index = (int)(laneDelta + (self & (width - 1))) >= width ? self : index;
  return __builtin_amdgcn_ds_bpermute(index << 2, var);
}
}  // namespace detail

BODGE_HIP inline void sleep_briefly() { __builtin_amdgcn_s_sleep(0); }
BODGE_HIP inline void sleep() { __builtin_amdgcn_s_sleep(100); }

BODGE_HIP __attribute__((always_inline)) inline uint32_t get_lane_id()
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
BODGE_HIP __attribute__((always_inline)) inline bool is_master_lane()
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

BODGE_HIP __attribute__((always_inline)) inline uint32_t broadcast_master(
    uint32_t x)
{
  return __builtin_amdgcn_readfirstlane(x);
}

BODGE_HIP inline void fence_acquire()
{
  __c11_atomic_thread_fence(__ATOMIC_ACQUIRE);
}
BODGE_HIP inline void fence_release()
{
  __c11_atomic_thread_fence(__ATOMIC_RELEASE);
}

template <typename T, size_t memorder, size_t scope>
BODGE_HIP T atomic_load(HOSTRPC_ATOMIC(T) const *addr)
{
  static_assert(detail::atomic_params_load<memorder, scope>(), "");
  return __opencl_atomic_load(addr, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
BODGE_HIP T atomic_load(__attribute__((address_space(1))) HOSTRPC_ATOMIC(T)
                            const *addr)
{
  static_assert(detail::atomic_params_load<memorder, scope>(), "");
  return __opencl_atomic_load(addr, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
BODGE_HIP void atomic_store(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(detail::atomic_params_store<memorder, scope>(), "");
  return __opencl_atomic_store(addr, value, memorder, scope);
}

template <typename T, size_t memorder, size_t scope>
BODGE_HIP void atomic_store(
    __attribute__((address_space(1))) HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(detail::atomic_params_store<memorder, scope>(), "");
  return __opencl_atomic_store(addr, value, memorder, scope);
}

BODGE_HIP inline uint32_t client_start_slot()
{
  // Ideally would return something < size
  // Attempt to distibute clients roughly across the array
  // compute unit currently executing the wave is a version of that
  enum
  {
    HW_ID = 4,  // specify that the hardware register to read is HW_ID

    HW_ID_CU_ID_SIZE = 4,    // size of CU_ID field in bits
    HW_ID_CU_ID_OFFSET = 8,  // offset of CU_ID from start of register

    HW_ID_SE_ID_SIZE = 2,     // sizeof SE_ID field in bits
    HW_ID_SE_ID_OFFSET = 13,  // offset of SE_ID from start of register
  };
#define ENCODE_HWREG(WIDTH, OFF, REG) (REG | (OFF << 6) | ((WIDTH - 1) << 11))
  uint32_t cu_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_CU_ID_SIZE, HW_ID_CU_ID_OFFSET, HW_ID));
  uint32_t se_id = __builtin_amdgcn_s_getreg(
      ENCODE_HWREG(HW_ID_SE_ID_SIZE, HW_ID_SE_ID_OFFSET, HW_ID));
  return (se_id << HW_ID_CU_ID_SIZE) + cu_id;
#undef ENCODE_HWREG
}

}  // namespace platform
#endif  // defined(__AMDGCN__)

#if HOSTRPC_NVPTX
namespace platform
{
inline void sleep_briefly() {}
inline void sleep() {}

namespace detail
{
// clang --target=nvptx64-nvidia-cuda doesn't seem to enabled the various
// intrinsics. Can't compile the majority of the code as cuda, so moving the
// platform functions out of line.

uint32_t get_master_lane_id();

int32_t __impl_shfl_down_sync(int32_t var, uint32_t laneDelta);

}  // namespace detail

uint32_t get_lane_id();

__attribute__((always_inline)) bool is_master_lane()
{
  return get_lane_id() == detail::get_master_lane_id();
}

uint32_t broadcast_master(uint32_t x);

void fence_acquire();
void fence_release();

template <typename T, size_t memorder, size_t scope>
T atomic_load(HOSTRPC_ATOMIC(T) const *addr)
{
  static_assert(detail::atomic_params_load<memorder, scope>(), "");
  // if T is _Atomic, this calls into a non-existent library
  // if T is volatile, emits a volatile load
  T res = *addr;
  if (memorder == __ATOMIC_ACQUIRE)
    {
      fence_acquire();
    }
  return res;
}

template <typename T, size_t memorder, size_t scope>
void atomic_store(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(detail::atomic_params_store<memorder, scope>(), "");
  if (memorder == __ATOMIC_RELEASE)
    {
      fence_release();
    }
  *addr = value;
}

inline uint32_t client_start_slot() { return 0; }

}  // namespace platform
#endif  // defined(__CUDACC__)

namespace platform
{
// Related functions derived from the platform specific ones

#if (HOSTRPC_AMDGCN || HOSTRPC_NVPTX)
BODGE_HIP __attribute__((always_inline)) inline uint32_t reduction_sum(
    uint32_t x)
{
  // could implement shfl_down for x64 and drop the macro
  x += detail::__impl_shfl_down_sync(x, 32);
  x += detail::__impl_shfl_down_sync(x, 16);
  x += detail::__impl_shfl_down_sync(x, 8);
  x += detail::__impl_shfl_down_sync(x, 4);
  x += detail::__impl_shfl_down_sync(x, 2);
  x += detail::__impl_shfl_down_sync(x, 1);
  return x;
}
#endif

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

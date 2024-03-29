#ifndef HOSTRPC_PLATFORM_HPP_INCLUDED
#define HOSTRPC_PLATFORM_HPP_INCLUDED

#include <stdint.h>

#include "base_types.hpp"  // page_t
#include "detail/fastint.hpp"
#include "platform/detect.hpp"

// todo: this should all be under namespace hostrpc

#if defined(__OPENCL_C_VERSION__)
// OpenCL requires _Atomic qualified types, but fails to parse
// _Atomic(type).

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
namespace platform
{
template <typename T>
struct ocl_atomic;

static_assert(sizeof(atomic_uint) == sizeof(uint32_t), "");
static_assert(sizeof(atomic_ulong) == sizeof(uint64_t), "");

template <>
struct ocl_atomic<uint64_t>
{
  using type = atomic_ulong;
};

template <>
struct ocl_atomic<uint32_t>
{
  using type = atomic_uint;
};

}  // namespace platform

#define HOSTRPC_ATOMIC(X) typename platform::ocl_atomic<X>::type
#else
#if HOSTRPC_NVPTX

#if _OPENMP
#define HOSTRPC_ATOMIC(X) X
#else
// cuda
#define HOSTRPC_ATOMIC(X) volatile _Atomic(X)
#endif
#else
#define HOSTRPC_ATOMIC(X) _Atomic(X)
#endif
#endif

#if 0
#if defined(__OPENCL_C_VERSION__)
#if HOSTRPC_HOST
extern "C" int printf(const char *format, ...);
#endif
#endif
#endif

namespace platform
{

// Functions implemented for each platform in platform_arch.hpp
namespace
{
// warp/wavefront width
inline HOSTRPC_ANNOTATE constexpr uint64_t native_width();

inline HOSTRPC_ANNOTATE void sleep_briefly();
inline HOSTRPC_ANNOTATE void sleep();

// auto is hiding some messy things here.
// the (probably x64) host wants to use a compile time constant
// where the gpu caller knows the thread mask as a compile time constant,
// those should also use a compile time constant
// however sometimes they don't
// nvptx and gfx10 want a 32 bit wide result
// gfx9 wants a 64 bit wide result
// Functions on runtime integers usually return the same type, but functions on
// the compile time ones return different types

inline HOSTRPC_ANNOTATE auto active_threads();

// get_lane_id is a runtime thing for GPUs and return 0 for the host
inline HOSTRPC_ANNOTATE auto get_lane_id();
 
template <typename T>
inline HOSTRPC_ANNOTATE auto get_master_lane_id(T active_threads);

template <typename T>
inline HOSTRPC_ANNOTATE bool is_master_lane(T active_threads);
 
template <typename T>
inline HOSTRPC_ANNOTATE uint32_t broadcast_master(T active_threads, uint32_t);

template <typename T>
HOSTRPC_ANNOTATE inline uint64_t
broadcast_master(T active_threads, uint64_t x);

inline HOSTRPC_ANNOTATE uint32_t client_start_slot();

inline HOSTRPC_ANNOTATE void fence_acquire();
inline HOSTRPC_ANNOTATE void fence_release();

// TODO: Move to utils?
inline HOSTRPC_ANNOTATE auto all_threads_active_constant();

}  // namespace
}  // namespace platform

#if HOSTRPC_HOST
#include "platform/host.hpp"
#endif

#if HOSTRPC_AMDGCN
#include "platform/amdgcn.hpp"
#endif

#if HOSTRPC_NVPTX
#include "platform/nvptx.hpp"
#endif

namespace platform
{

static_assert(native_width() > 0, "");
static_assert(native_width() <= 64, "");

namespace
{

}  // namespace

// related functions derived from the above
namespace
{

template <typename T>
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline uint64_t
broadcast_master(T active_threads, uint64_t x)
{
  uint32_t lo = x;
  uint32_t hi = x >> 32u;
  lo = broadcast_master(active_threads, lo);
  hi = broadcast_master(active_threads, hi);
  return ((uint64_t)hi << 32u) | lo;
}

template <typename T>
inline HOSTRPC_ANNOTATE auto get_master_lane_id(T active_threads)
{
  // The lowest bit set. If changing, need to change broadcast_master to match.
  auto f = active_threads.findFirstSet();
  return f.template subtract<1>();
}

template <typename T>
inline HOSTRPC_ANNOTATE bool is_master_lane(T active_threads)
{
  return get_lane_id() == get_master_lane_id(active_threads);
}
 
}  // namespace

// atomics may also be overloaded on different address spaces for some platforms
// implemented for a slight superset of the subset of T that are presently in
// use. Probably clearer to implement on specific integer types.

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_load(HOSTRPC_ATOMIC(T) const *);

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE void atomic_store(HOSTRPC_ATOMIC(T) *, T);

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_add(HOSTRPC_ATOMIC(T) *, T);

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_sub(HOSTRPC_ATOMIC(T) *, T);

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_and(HOSTRPC_ATOMIC(T) *, T);

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_or(HOSTRPC_ATOMIC(T) *, T);

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_xor(HOSTRPC_ATOMIC(T) *, T);

}  // namespace platform

// This is exciting. Nvptx doesn't implement atomic, so one must use volatile +
// fences. C++ doesn't let one write operator new() for a volatile pointer Net
// effect is that we can't have code that conforms to the C++ object model and
// volatile qualifies the underlying object

namespace platform
{
namespace detail
{
template <size_t scope>
HOSTRPC_ANNOTATE constexpr bool atomic_params_scope()
{
  return (scope == __OPENCL_MEMORY_SCOPE_DEVICE) ||
         (scope == __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
}
template <size_t memorder, size_t scope>
HOSTRPC_ANNOTATE constexpr bool atomic_params_load()
{
  return (atomic_params_scope<scope>() &&
          ((memorder == __ATOMIC_RELAXED) || (memorder == __ATOMIC_ACQUIRE)));
}

template <size_t memorder, size_t scope>
HOSTRPC_ANNOTATE constexpr bool atomic_params_store()
{
  return (atomic_params_scope<scope>() &&
          ((memorder == __ATOMIC_RELAXED) || (memorder == __ATOMIC_RELEASE)));
}

template <size_t memorder, size_t scope>
HOSTRPC_ANNOTATE constexpr bool atomic_params_readmodifywrite()
{
  return (atomic_params_scope<scope>() &&
          ((memorder == __ATOMIC_RELAXED) || (memorder == __ATOMIC_ACQ_REL)));
}

}  // namespace detail
}  // namespace platform

// Jury rig some pieces of libc for freestanding
// Assert is based on the implementation in musl

#if (HOSTRPC_HOST && defined(__OPENCL_C_VERSION__))
// No assert available. Probably want to change to platform::require
// and provide implementations for each arch.
#ifdef NDEBUG
#define assert(x) (void)0
#else
#define assert(x) (void)(x)
#endif
#define printf(...) __hostrpc_printf(__VA_ARGS__)

extern "C" HOSTRPC_ANNOTATE __attribute__((always_inline)) inline int
hostrpc_inline_printf()
{
  // opencl has no stdio
  return 0;
}
#endif

#if (HOSTRPC_AMDGCN || HOSTRPC_NVPTX)

// stub printf for now
// aomp clang currently rewrites any variadic function to a pair of
// allocate/execute functions, which don't necessarily exist.
// Clobber it with the preprocessor as a workaround.

// need to do something more robust with this
#if defined(_OPENMP) && HOSTRPC_AMDGCN
#define printf(...)  //__hostrpc_printf(__VA_ARGS__)
#endif

// trying to get hip code to compile
#if defined(assert)
#undef assert
#endif

#ifdef NDEBUG
#define assert(x) (void)0
#else
// Testing (x) || assert_fail with potentially non-uniform x leads to a
// complicated CFG, as some lanes trap and some lanes don't. Explictly failing
// the assert when any lane is false avoids this.
// May be better to write assert to disguise the branching so that it does not
// change the CFG
#define assert(x) (void)(x)

#endif

#ifdef static_assert
#undef static_assert
#endif
#define static_assert _Static_assert

#endif

namespace platform
{
#if (HOSTRPC_AMDGCN)
namespace amdgcn
{
namespace detail
{

HOSTRPC_ANNOTATE __attribute__((always_inline)) inline void assert_fail(
    const char *str, const char *, unsigned int line, const char *)
{
#ifdef NDEBUG
  asm("// Assert fail " ::"r"(line), "r"(str));
#else
  (void)str;
  (void)line;
#endif
  __builtin_trap();
}
}  // namespace detail
}  // namespace amdgcn
#endif

#if (HOSTRPC_NVPTX)
namespace nvptx
{
namespace detail
{
static_assert(sizeof(uint64_t) == sizeof(unsigned long long),
              "yet they mangle differently");

HOSTRPC_ANNOTATE void assert_fail(const char *str, const char *,
                                  unsigned int line, const char *);
}  // namespace detail
}  // namespace nvptx
#endif

}  // namespace platform

#if HOSTRPC_HOST

#if !defined(__OPENCL_C_VERSION__)
#include <cassert>
#endif

namespace platform
{
namespace host
{

namespace detail
{
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline void assert_fail(
    const char *str, const char *file, unsigned int line, const char *func)
{
  // todo
  (void)str;
  (void)file;
  (void)line;
  (void)func;
}
}  // namespace detail

#define HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE
#include "platform/atomic.inc"

}  // namespace host
}  // namespace platform
#endif

// HIP seems to want every function to be explicitly marked device or host
// even when it's only compiling for, say, device
// That may mean every single function needs a MACRO_ANNOTATION for hip to
// work as intended.
// Cuda does not appear to require this. TODO: See if openmp does

#if HOSTRPC_AMDGCN

namespace platform
{
namespace amdgcn
{

#define HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE
#include "platform/atomic.inc"

}  // namespace amdgcn

// amdgcn uses an overload in address space 1 for these functions
// todo: see if these overloads can be dropped by casting at call site
#define HOSTRPC_PLATFORM_ATOMIC_ADDRSPACE_ATTRIBUTE \
  __attribute__((address_space(1)))
#include "platform/atomic.inc"

}  // namespace platform
#endif  // defined(__AMDGCN__)

#if HOSTRPC_NVPTX

namespace platform
{

#ifdef _OPENMP

namespace nvptx
{

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_load( HOSTRPC_ATOMIC(T) const *addr)
{
  static_assert(platform::detail::atomic_params_load<memorder, scope>(), "");

  return __atomic_fetch_add(const_cast<HOSTRPC_ATOMIC(T)*>(addr), T(0), memorder);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE void atomic_store(
  HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(platform::detail::atomic_params_store<memorder, scope>(), "");
  _atomic_store_n(addr, value, memorder);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_add(
     HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __atomic_fetch_add(addr, value, memorder);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_sub(
     HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __atomic_fetch_sub(addr, value, memorder);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_and(
     HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __atomic_fetch_and(addr, value, memorder);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_or(
     HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __atomic_fetch_or(addr, value, memorder);
}


  template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_xor(
     HOSTRPC_ATOMIC(T) * addr,
    T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return __atomic_fetch_xor(addr, value, memorder);
}


}
  

#else // try to get something working though cuda, not very well tested
  
namespace nvptx
{

namespace detail
{
// clang --target=nvptx64-nvidia-cuda doesn't seem to enabled the various
// intrinsics. Can't compile the majority of the code as cuda, so moving the
// platform functions out of line.

}  // namespace detail

// The cuda/ptx compiler lowers opencl intrinsics to IR atomics if compiling as
// cuda. If compiling as C++, it leaves them as external function calls. As
// cuda, the scope parameter is presently ignored. Memory order acq_rel is
// accepted, but as cuda only provides relaxed semantics, assuming it is at risk
// of miscompilation

namespace detail
{
#ifdef HOSTRPC_STAMP_MEMORY
#error "HOSTRPC_STAMP_MEMORY already defined"
#endif
#define HOSTRPC_STAMP_MEMORY(TYPE)                                             \
  HOSTRPC_ANNOTATE TYPE atomic_load_relaxed(HOSTRPC_ATOMIC(TYPE) const *addr); \
  HOSTRPC_ANNOTATE void atomic_store_relaxed(HOSTRPC_ATOMIC(TYPE) * addr, TYPE)

#ifdef HOSTRPC_STAMP_FETCH_OPS
#error "HOSTRPC_STAMP_FETCH_OPS already defined"
#endif
#define HOSTRPC_STAMP_FETCH_OPS(TYPE)                                         \
  HOSTRPC_ANNOTATE TYPE atomic_fetch_add_relaxed(HOSTRPC_ATOMIC(TYPE) * addr, \
                                                 TYPE value);                 \
  HOSTRPC_ANNOTATE TYPE atomic_fetch_sub_relaxed(HOSTRPC_ATOMIC(TYPE) * addr, \
                                                 TYPE value);                 \
  HOSTRPC_ANNOTATE TYPE atomic_fetch_and_relaxed(HOSTRPC_ATOMIC(TYPE) * addr, \
                                                 TYPE value);                 \
  HOSTRPC_ANNOTATE TYPE atomic_fetch_or_relaxed(HOSTRPC_ATOMIC(TYPE) * addr,  \
                                                TYPE value);                  \
  HOSTRPC_ANNOTATE TYPE atomic_fetch_xor_relaxed(HOSTRPC_ATOMIC(TYPE) * addr, \
                                                 TYPE value);                 

HOSTRPC_STAMP_MEMORY(uint8_t);
HOSTRPC_STAMP_MEMORY(uint16_t);
HOSTRPC_STAMP_MEMORY(uint32_t);
HOSTRPC_STAMP_MEMORY(uint64_t);

HOSTRPC_STAMP_FETCH_OPS(uint32_t);
HOSTRPC_STAMP_FETCH_OPS(uint64_t);

#undef HOSTRPC_STAMP_MEMORY
#undef HOSTRPC_STAMP_FETCH_OPS

template <typename T, T (*op)(HOSTRPC_ATOMIC(T) *, T), size_t memorder,
          size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_op(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");

  if (memorder == __ATOMIC_ACQ_REL)
    {
      fence_release();
    }

  T res = op(addr, value);

  if (memorder == __ATOMIC_ACQ_REL)
    {
      fence_acquire();
    }

  return res;
}
}  // namespace detail

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_load(HOSTRPC_ATOMIC(T) const *addr)
{
  static_assert(sizeof(T) <= 8, "");
  static_assert(platform::detail::atomic_params_load<memorder, scope>(), "");
  T res = detail::atomic_load_relaxed(addr);
  if (memorder == __ATOMIC_ACQUIRE)
    {
      fence_acquire();
    }
  return res;
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE void atomic_store(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(platform::detail::atomic_params_store<memorder, scope>(), "");
  if (memorder == __ATOMIC_RELEASE)
    {
      fence_release();
    }
  detail::atomic_store_relaxed(addr, value);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_add(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return detail::atomic_fetch_op<T, detail::atomic_fetch_add_relaxed, memorder,
                                 scope>(addr, value);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_sub(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return detail::atomic_fetch_op<T, detail::atomic_fetch_sub_relaxed, memorder,
                                 scope>(addr, value);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_and(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return detail::atomic_fetch_op<T, detail::atomic_fetch_and_relaxed, memorder,
                                 scope>(addr, value);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_or(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return detail::atomic_fetch_op<T, detail::atomic_fetch_or_relaxed, memorder,
                                 scope>(addr, value);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE T atomic_fetch_xor(HOSTRPC_ATOMIC(T) * addr, T value)
{
  static_assert(
      platform::detail::atomic_params_readmodifywrite<memorder, scope>(), "");
  return detail::atomic_fetch_op<T, detail::atomic_fetch_xor_relaxed, memorder,
                                 scope>(addr, value);
}


}  // namespace nvptx


#endif
}  // namespace platform
#endif  // HOSTRPC_NVPTX

// Dispatch in a fashion compatible with c++, cuda, hip, opencl, openmp

#if HOSTRPC_HOST
#define HOSTRPC_IMPL_NS platform::host
#elif HOSTRPC_AMDGCN
#define HOSTRPC_IMPL_NS platform::amdgcn
#elif HOSTRPC_NVPTX
#define HOSTRPC_IMPL_NS platform::nvptx
#else
#error "Unknown compile mode"
#endif

namespace platform
{
// Functions implemented for each platform

namespace detail
{
HOSTRPC_ANNOTATE __attribute__((always_inline)) inline void assert_fail(
    const char *str, const char *file, unsigned int line, const char *func)
{
  HOSTRPC_IMPL_NS::detail::assert_fail(str, file, line, func);
}
}  // namespace detail

// atomics may also be overloaded on different address spaces for some platforms
// implemented for a slight superset of the subset of T that are presently in
// use

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE inline T atomic_load(HOSTRPC_ATOMIC(T) const *a)
{
  return HOSTRPC_IMPL_NS::atomic_load<T, memorder, scope>(a);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE inline void atomic_store(HOSTRPC_ATOMIC(T) * a, T v)
{
  return HOSTRPC_IMPL_NS::atomic_store<T, memorder, scope>(a, v);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE inline T atomic_fetch_add(HOSTRPC_ATOMIC(T) * a, T v)
{
  return HOSTRPC_IMPL_NS::atomic_fetch_add<T, memorder, scope>(a, v);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE inline T atomic_fetch_sub(HOSTRPC_ATOMIC(T) * a, T v)
{
  return HOSTRPC_IMPL_NS::atomic_fetch_sub<T, memorder, scope>(a, v);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE inline T atomic_fetch_and(HOSTRPC_ATOMIC(T) * a, T v)
{
  return HOSTRPC_IMPL_NS::atomic_fetch_and<T, memorder, scope>(a, v);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE inline T atomic_fetch_or(HOSTRPC_ATOMIC(T) * a, T v)
{
  return HOSTRPC_IMPL_NS::atomic_fetch_or<T, memorder, scope>(a, v);
}

template <typename T, size_t memorder, size_t scope>
HOSTRPC_ANNOTATE inline T atomic_fetch_xor(HOSTRPC_ATOMIC(T) * a, T v)
{
  return HOSTRPC_IMPL_NS::atomic_fetch_xor<T, memorder, scope>(a, v);
}


}  // namespace platform

#endif

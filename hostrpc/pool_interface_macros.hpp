#ifndef POOL_INTERFACE_MACROS_HPP_INCLUDED
#define POOL_INTERFACE_MACROS_HPP_INCLUDED

#include "platform/detect.hpp"

// Macros that expand into various boilerplate definitions based on a symbol
// Set up so macros can be used on amdgcn, host or amdgcn-opencl, such
// that a file that uses these at the top level can compile as all three
// Much of this can be removed if amdgpu_kernel becomes a clang-accessible
// calling convention, and some parts could be factored into a gpu lib for
// launching kernels

// opencl kernel shims to work around __kernel being unavailable in c++
// creates extern "C" functions that call into the type's static members
// and opencl kernels that call said extern "C" functions, then finally
// creates implementations for host functions that use the hsa api to call
// those kernels

#ifdef __cplusplus
#define POOL_INTERFACE_EXTERN_C() extern "C"
#else
#define POOL_INTERFACE_EXTERN_C()
#endif

#if HOSTRPC_HOST
#define POOL_INTERFACE_GPU_WRAPPERS(SYMBOL)
#endif

#if HOSTRPC_AMDGCN

#if defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(NAME) \
  POOL_INTERFACE_EXTERN_C() void NAME(void);            \
  POOL_INTERFACE_EXTERN_C() kernel void __device_##NAME(void) { NAME(); }
#else
#define POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(NAME)
#endif

#define POOL_INTERFACE_GPU_OPENCL_WRAPPERS(SYMBOL)       \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(             \
      pool_interface_##SYMBOL##_set_requested)           \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(             \
      pool_interface_##SYMBOL##_bootstrap_entry)         \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(             \
      pool_interface_##SYMBOL##_bootstrap_kernel_target) \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(pool_interface_##SYMBOL##_teardown)

// extern C functions that call into the type with run() implemented

#if defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_GPU_C_WRAPPERS(SYMBOL)
#else
#define POOL_INTERFACE_GPU_C_WRAPPERS(SYMBOL)                                \
  extern "C"                                                                 \
  {                                                                          \
    void pool_interface_##SYMBOL##_set_requested(void)                       \
    {                                                                        \
      SYMBOL::set_requested(                                                 \
          pool_interface::getlo(pool_interface::load_from_reserved_addr())); \
    }                                                                        \
    void pool_interface_##SYMBOL##_bootstrap_kernel_target(void)             \
    {                                                                        \
      SYMBOL::expose_loop_for_bootstrap_implementation();                    \
    }                                                                        \
    void pool_interface_##SYMBOL##_teardown(void) { SYMBOL::teardown(); }    \
    void pool_interface_##SYMBOL##_bootstrap_entry(void)                     \
    {                                                                        \
      SYMBOL::bootstrap(                                                     \
          pool_interface::getlo(pool_interface::load_from_reserved_addr())); \
    }                                                                        \
  }
#endif

#define POOL_INTERFACE_GPU_WRAPPERS(SYMBOL)  \
  POOL_INTERFACE_GPU_OPENCL_WRAPPERS(SYMBOL) \
  POOL_INTERFACE_GPU_C_WRAPPERS(SYMBOL)

#endif

#ifndef POOL_INTERFACE_GPU_WRAPPERS
#error "Missing defn of POOL_INTERFACE_GPU_WRAPPERS"
#endif

#endif

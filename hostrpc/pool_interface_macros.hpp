#ifndef POOL_INTERFACE_MACROS_HPP_INCLUDED
#define POOL_INTERFACE_MACROS_HPP_INCLUDED

#include "detail/platform_detect.hpp"

// Macros that expand into various boilerplate definitions based on a symbol
// Set up so macros can be used on amdgcn, host or amdgcn-opencl, such
// that a file that uses these at the top level can compile as all three
// Much of this can be removed if amdgpu_kernel becomes a clang-accessible
// calling convention, and some parts could be factored into a gpu lib for
// launching kernels

// opencl kernel shims to work around __kernel being unavailable in c++
#if HOSTRPC_AMDGCN && defined(__OPENCL_C_VERSION__)
#ifdef __cplusplus
#define POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(NAME) \
  extern "C" void NAME(void);                           \
  extern "C" kernel void __device_##NAME(void) { NAME(); }
#else
#define POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(NAME) \
  void NAME(void);                                      \
  kernel void __device_##NAME(void) { NAME(); }
#endif
#else
#define POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(NAME)
#endif

#define POOL_INTERFACE_GPU_OPENCL_WRAPPERS(SYMBOL)                     \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_set_requested)    \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_bootstrap_entry)  \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_bootstrap_target) \
  POOL_INTERFACE_WRAP_VOID_IN_OPENCL_KERNEL(SYMBOL##_teardown)

// extern C functions that call into the type with run() implemented
#if HOSTRPC_AMDGCN && !defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_GPU_C_WRAPPERS(SYMBOL)                            \
  extern "C"                                                             \
  {                                                                      \
    void SYMBOL##_set_requested(void)                                    \
    {                                                                    \
      SYMBOL::set_requested(pool_interface::load_from_reserved_addr());  \
    }                                                                    \
    void SYMBOL##_bootstrap_target(void) { SYMBOL::bootstrap_target(); } \
    void SYMBOL##_teardown(void) { SYMBOL::teardown(); }                 \
                                                                         \
    void SYMBOL##_bootstrap_entry(void)                                  \
    {                                                                    \
      __attribute__(                                                     \
          (visibility("default"))) extern hsa_packet::kernel_descriptor  \
          SYMBOL##_bootstrap_target_desc asm("__device_" #SYMBOL         \
                                             "_bootstrap_target.kd");    \
      SYMBOL::instance()->bootstrap(                                     \
          pool_interface::load_from_reserved_addr(),                     \
          (const unsigned char *)&SYMBOL##_bootstrap_target_desc);       \
    }                                                                    \
  }
#else
#define POOL_INTERFACE_GPU_C_WRAPPERS(SYMBOL)
#endif

// functions that, post _initialize, run the corresponding function on the pool
#if HOSTRPC_HOST && !defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_HSA_KERNEL_WRAPPERS(SYMBOL)                            \
                                                                              \
  hsa_host_pool_state SYMBOL##_global;                                        \
                                                                              \
  int SYMBOL##_initialize(hsa::executable &ex, hsa_queue_t *queue)            \
  {                                                                           \
    int rc = 0;                                                               \
    rc += initialize_kernel_info(ex, "__device_" #SYMBOL "_set_requested.kd", \
                                 &SYMBOL##_global.set_requested);             \
    rc +=                                                                     \
        initialize_kernel_info(ex, "__device_" #SYMBOL "_bootstrap_entry.kd", \
                               &SYMBOL##_global.bootstrap_entry);             \
    rc += initialize_kernel_info(ex, "__device_" #SYMBOL "_teardown.kd",      \
                                 &SYMBOL##_global.teardown);                  \
    if (rc != 0)                                                              \
      {                                                                       \
        return 1;                                                             \
      }                                                                       \
                                                                              \
    SYMBOL##_global.queue = queue;                                            \
    if (hsa_signal_create(1, 0, NULL, &SYMBOL##_global.signal) !=             \
        HSA_STATUS_SUCCESS)                                                   \
      {                                                                       \
        return 1;                                                             \
      }                                                                       \
    return 0;                                                                 \
  }                                                                           \
  int SYMBOL##_finalize()                                                     \
  {                                                                           \
    if (SYMBOL##_global.signal.handle)                                        \
      {                                                                       \
        hsa_signal_destroy(SYMBOL##_global.signal);                           \
      }                                                                       \
    return 0;                                                                 \
  }                                                                           \
  void SYMBOL##_set_requested(uint64_t requested)                             \
  {                                                                           \
    hsa_queue_t *queue = SYMBOL##_global.queue;                               \
    gpu_kernel_info &req = SYMBOL##_global.set_requested;                     \
    hsa::launch_kernel(req.symbol_address, req.private_segment_fixed_size,    \
                       req.group_segment_fixed_size, queue, requested, 0,     \
                       {0});                                                  \
  }                                                                           \
                                                                              \
  void SYMBOL##_bootstrap_entry(uint64_t requested)                           \
  {                                                                           \
    hsa_queue_t *queue = SYMBOL##_global.queue;                               \
    gpu_kernel_info &req = SYMBOL##_global.bootstrap_entry;                   \
    hsa::launch_kernel(req.symbol_address, req.private_segment_fixed_size,    \
                       req.group_segment_fixed_size, queue, requested, 0,     \
                       {0});                                                  \
  }                                                                           \
                                                                              \
  void SYMBOL##_teardown()                                                    \
  {                                                                           \
    hsa_queue_t *queue = SYMBOL##_global.queue;                               \
    invoke_teardown(SYMBOL##_global.teardown, SYMBOL##_global.signal, queue); \
  }
#else
#define POOL_INTERFACE_HSA_KERNEL_WRAPPERS(SYMBOL)
#endif

#endif

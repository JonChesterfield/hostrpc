#ifndef RUN_ON_HSA_HPP_INCLUDED
#define RUN_ON_HSA_HPP_INCLUDED

// Call a function with a given name and type:
// void known_function_name(void*);
// from:
// kernel void kernel_known_function_name(__global void *){}
// provided that source containing 'HOSTRPC_ENTRY_POINT(foo)'
// is compiled as opencl and as c++, which are then linked

// Call extern "C" void NAME(TYPE*); defined in amdgcn source from the host
// from an instantiation of HOSTRPC_ENTRY_POINT(NAME, TYPE) compiled as
// amdgcn ocl/cxx and on the host

#include "detail/platform_detect.h"

#define HOSTRPC_ENTRY_POINT(NAME, TYPE) \
  /* void NAME(TYPE*); */               \
  HOSTRPC_OPENCL_PART(NAME, TYPE)       \
  HOSTRPC_CXX_GCN_PART(NAME, TYPE)      \
  HOSTRPC_CXX_X64_PART(NAME, TYPE)
#define HOSTRPC_CAT(X, Y) X##Y

#if HOSTRPC_HOST

#include <stddef.h>

namespace hostrpc
{
void run_on_hsa(void *arg, size_t len, const char *name);

template <typename T>
void run_on_hsa_typed(T *arg, const char *name)
{
  run_on_hsa((void *)arg, sizeof(T), name);
}
}  // namespace hostrpc

#define HOSTRPC_OPENCL_PART(NAME, TYPE)
#define HOSTRPC_CXX_GCN_PART(NAME, TYPE)
#define HOSTRPC_CXX_X64_PART(NAME, TYPE)                   \
  extern "C" void NAME(TYPE *arg)                          \
  {                                                        \
    hostrpc::run_on_hsa_typed<TYPE>(arg, "kernel_" #NAME); \
  }

#else

#if defined(__OPENCL_C_VERSION__)

#define HOSTRPC_OPENCL_PART(NAME, TYPE)                       \
  void HOSTRPC_CAT(cast_, NAME)(__global TYPE *);             \
  kernel void HOSTRPC_CAT(kernel_, NAME)(__global TYPE * arg) \
  {                                                           \
    HOSTRPC_CAT(cast_, NAME)(arg);                            \
  }
#define HOSTRPC_CXX_GCN_PART(NAME, TYPE)
#define HOSTRPC_CXX_X64_PART(NAME, TYPE)

#else

#define HOSTRPC_OPENCL_PART(NAME, TYPE)
#define HOSTRPC_CXX_GCN_PART(NAME, TYPE)                            \
  extern "C" void NAME(TYPE *);                                          \
  extern "C" void HOSTRPC_CAT(                                      \
      cast_, NAME)(__attribute__((address_space(1))) TYPE * asargs) \
  {                                                                 \
    TYPE *args = (TYPE *)asargs;                                    \
    NAME(args);                                                     \
  }
#define HOSTRPC_CXX_X64_PART(NAME, TYPE)

#endif

#endif

#endif

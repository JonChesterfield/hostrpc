#ifndef RUN_ON_HSA_HPP_INCLUDED
#define RUN_ON_HSA_HPP_INCLUDED

// Call a function with a given name and type:
// void known_function_name(void*);
// from:
// kernel void kernel_known_function_name(__global void *){}
// provided that source containing 'HOSTRPC_ENTRY_POINT(foo)'
// is compiled as opencl and as c++, which are then linked

#include "detail/platform_detect.h"

#if defined(__AMDGCN__)

#define HOSTRPC_ENTRY_POINT(STR) \
  HOSTRPC_OPENCL_PART(STR)       \
  HOSTRPC_CXX_PART(STR)

#define HOSTRPC_CAT(X, Y) X##Y

#if defined(__OPENCL_C_VERSION__)
#define HOSTRPC_OPENCL_PART(STR)                            \
  void HOSTRPC_CAT(cast_, STR)(void *);                     \
  kernel void HOSTRPC_CAT(kernel_, STR)(__global void *arg) \
  {                                                         \
    known_function_name(arg);                               \
  }
#define HOSTRPC_CXX_PART(STR)
#else
#define HOSTRPC_OPENCL_PART(STR)
#define HOSTRPC_CXX_PART(STR)                                     \
  extern "C" void HOSTRPC_CAT(                                    \
      cast_, STR)(__attribute__((address_space(1))) void *asargs) \
  {                                                               \
    void *args = (void *)asargs;                                  \
    STR(args);                                                    \
  }

#endif

HOSTRPC_ENTRY_POINT(some_name)

#endif


struct example_arg
{
  int x;
  int r;
};
void example_call(example_arg*);

HOSTRPC_ENTRY_POINT(example_call)



#endif

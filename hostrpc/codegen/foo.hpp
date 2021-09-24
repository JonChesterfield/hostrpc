#ifndef FOO_HPP_INCLUDED
#define FOO_HPP_INCLUDED

// Implement a function,
// void plaform::foo();
// such that it can be called from a variety of languages

// error: __host__ __device__ function 'foo' cannot overload __device__ function
// 'foo' error: __host__ __device__ function 'foo' cannot overload __host__
// function 'foo'

#include "../detail/platform/detect.hpp"

#if defined(__OPENCL_C_VERSION__)
#if defined(__HIP__) || defined(__CUDA__)
#error "opencl and hip|cuda ?"
#endif

// defines __AMDGCN__, __NVPTX__ for the device compilation only

#endif

#if defined(_OPENMP)

// openmp defines __host__ macro, via
// include/openmp_wrappers/__clang_openmp_device_functions.h:66
// probably to use parts of libm

// a function marked 'declare target' in openmp is also defined on the host
// defining another function with the same name is a redefinition error
// this could be worked around using variant

#define __p(STR) _Pragma(STR)
#define __p2(STR) __p(#STR)

#endif

namespace platform
{
// do something platform dependent in each
#if HOSTRPC_HOST
namespace host
{
HOSTRPC_ANNOTATE_HOST void foo() { __builtin_ia32_sfence(); }
}  // namespace host
#endif

#if HOSTRPC_AMDGCN
namespace amdgcn
{
HOSTRPC_ANNOTATE_DEVICE void foo() { __builtin_amdgcn_s_sleep(0); }
}  // namespace amdgcn
#endif

#if HOSTRPC_NVPTX
namespace nvptx
{
HOSTRPC_ANNOTATE_DEVICE void foo() { (void)__nvvm_read_ptx_sreg_tid_x(); }
}  // namespace nvptx
#endif
}  // namespace platform

#if HOSTRPC_HOST
#define HOSTRPC_IMPL_NS host
#elif HOSTRPC_AMDGCN
#define HOSTRPC_IMPL_NS amdgcn
#elif HOSTRPC_NVPTX
#define HOSTRPC_IMPL_NS nvptx
#else
#error "Unknown compile mode"
#endif

#pragma omp declare target

namespace platform
{
// marks it __host__ __device__
HOSTRPC_ANNOTATE
void foo() { HOSTRPC_IMPL_NS::foo(); }

}  // namespace platform

#pragma omp end declare target

#endif

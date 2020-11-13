#ifndef FOO_HPP_INCLUDED
#define FOO_HPP_INCLUDED

// Implement a function,
// void plaform::foo();
// such that it can be called from a variety of languages

// error: __host__ __device__ function 'foo' cannot overload __device__ function
// 'foo' error: __host__ __device__ function 'foo' cannot overload __host__
// function 'foo'

#if defined(_OPENMP)

// openmp defines __host__ macro, via
// include/openmp_wrappers/__clang_openmp_device_functions.h:66
// probably to use parts of libm

// a function marked 'declare target' in openmp is also defined on the host
// defining another function with the same name is a redefinition error
// this could be worked around using variant

#define HOSTRPC_HOST
#define HOSTRPC_DEVICE

#else

#if defined(__HIP__) | defined(__CUDA__)
#define HOSTRPC_HOST __attribute__((host))
#define HOSTRPC_DEVICE __attribute__((device))
#else
#define HOSTRPC_HOST
#define HOSTRPC_DEVICE
#endif

#endif

namespace platform
{
void foo();

HOSTRPC_HOST void foo();

HOSTRPC_DEVICE void foo();

#pragma omp declare target
void foo();
#pragma omp end declare target

// If using this declaration, it must be the only declaration
//  __host__   __device__   void foo();

}  // namespace platform

#endif

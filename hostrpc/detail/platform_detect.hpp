#ifndef PLATFORM_DETECT_HPP_INCLUDED
#define PLATFORM_DETECT_HPP_INCLUDED

#if defined(__NVPTX__) && defined (__AMDGCN__)
#error "NVPTX and AMDGCN both defined"
#endif

#if defined(__CUDA__) && defined(__HIP__)
// Warning? Probably treating these two the same
#error "Cuda and hip both defined"
#endif


// A few compilation modes exist.
// Simple one is compiling for one architecture (x64, gcn, ptx) at a time
// This is used by C++ and by opencl. x64 is referred to as 'host' here, but
// hosts other than x64 have not yet been tested

#define HOSTRPC_AMDGCN 0
#define HOSTRPC_NVPTX 0
#define HOSTRPC_HOST 0

#ifndef HOSTRPC_HAVE_STDIO

#if defined (__OPENCL_C_VERSION__)
#define HOSTRPC_HAVE_STDIO 0
#endif

#if defined (__CUDA__) || defined (__HIP__)
#define HOSTRPC_HAVE_STDIO 0
#endif

#if defined(_OPENMP)
#define HOSTRPC_HAVE_STDIO 0
#endif

#ifndef HOSTRPC_HAVE_STDIO
#define HOSTRPC_HAVE_STDIO HOSTRPC_HOST
#endif

#endif


#if !defined (__NVPTX__) & !defined(__AMDGCN__)
// TODO: Consider simplifying the following based on this
#undef HOSTRPC_HOST
#define HOSTRPC_HOST 1
#endif

#if defined(_OPENMP)
  #if defined (__AMDGCN__)
    //#warning "OpenMP gcn gpu"
    #undef HOSTRPC_AMDGCN
    #define HOSTRPC_AMDGCN 1
  #endif
  #if defined (__NVPTX__)
    //#warning "OpenMP ptx gpu"
    #undef HOSTRPC_NVPTX
    #define HOSTRPC_NVPTX 1
  #endif
  #if !defined (__AMDGCN__) && !defined (__NVPTX__)
    //#warning "OpenMP host"
    #undef HOSTRPC_HOST
    #define HOSTRPC_HOST 1
  #endif
#endif

#if defined (__OPENCL_C_VERSION__)
  #if defined (__AMDGCN__)
    //# warning "OpenCL gcn gpu"
    #undef HOSTRPC_AMDGCN
    #define HOSTRPC_AMDGCN 1
  #endif
  #if defined (__NVPTX__)
    //#warning "OpenCL ptx gpu"
    #undef HOSTRPC_NVPTX
    #define HOSTRPC_NVPTX 1
  #endif
    #if !defined (__AMDGCN__) && !defined (__NVPTX__)
    //#warning "OpenCL host"
    #undef HOSTRPC_HOST
    #define HOSTRPC_HOST 1
  #endif
#endif
  
#if !defined(_OPENMP) && defined(__NVPTX__)
  #if defined (__CUDA__)
    #if defined(__CUDA_ARCH__)
      //#warning "Cuda gpu"
      #undef HOSTRPC_NVPTX
      #define HOSTRPC_NVPTX 1
    #else
      //#warning "Cuda host"
      #undef HOSTRPC_HOST
      #define HOSTRPC_HOST 1
    #endif
  #else
    //#warning "Ptx freestanding"
    #undef HOSTRPC_NVPTX
    #define HOSTRPC_NVPTX 1
  #endif
#endif
  
#if !defined(_OPENMP) && defined (__AMDGCN__)
  #if defined(__HIP__)
    #if defined(__HIP_DEVICE_COMPILE__)
      //#warning "Hip gpu"
      #undef HOSTRPC_AMDGCN
      #define HOSTRPC_AMDGCN 1
    #else
      //#warning "Hip host"
      #undef HOSTRPC_HOST
      #define HOSTRPC_HOST 1
    #endif
  #else
    //#warning "GCN freestanding"
    #undef HOSTRPC_AMDGCN
    #define HOSTRPC_AMDGCN 1
  #endif
#endif

#define HOSTRPC_GPU (HOSTRPC_AMDGCN | HOSTRPC_NVPTX)

#if (HOSTRPC_AMDGCN + HOSTRPC_NVPTX + HOSTRPC_HOST) != 1
#error "Platform detection failed"
#endif


// clang -x cuda errors on __device__, __host__ but seems to do the right thing with __attribute__

// openmp presently defines __HIP__
// distinction between host/device is used in some coodegen tests, can be
// removed before production
#if (defined(__CUDA__) || defined (__HIP__)) && !defined(_OPENMP)
  #define HOSTRPC_ANNOTATE_HOST __attribute__((host))
  #define HOSTRPC_ANNOTATE_DEVICE __attribute__((device))
#else
  #if HOSTRPC_HOST
    #define HOSTRPC_ANNOTATE_HOST
    #define HOSTRPC_ANNOTATE_DEVICE
  #else
    #define HOSTRPC_ANNOTATE_HOST
    #define HOSTRPC_ANNOTATE_DEVICE __attribute__((convergent))
  #endif
#endif
#define HOSTRPC_ANNOTATE HOSTRPC_ANNOTATE_HOST HOSTRPC_ANNOTATE_DEVICE


#endif

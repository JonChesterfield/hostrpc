#ifndef PLATFORM_DETECT_H_INCLUDED
#define PLATFORM_DETECT_H_INCLUDED

#if defined(__NVPTX__) && defined (__AMDGCN__)
#error "NVPTX and AMDGCN both defined"
#endif

#define HOSTRPC_AMDGCN 0
#define HOSTRPC_NVPTX 0
#define HOSTRPC_HOST 0

#if defined(_OPENMP)
  #if defined (__AMDGCN__)
    #warning "OpenMP gcn gpu"
    #undef HOSTRPC_AMDGCN
    #define HOSTRPC_AMDGCN 1
  #endif
  #if defined (__NVPTX__)
    #warning "OpenMP ptx gpu"
    #undef HOSTRPC_NVPTX
    #define HOSTRPC_NVPTX 1
  #endif
  #if !defined (__AMDGCN__) && !defined (__NVPTX__)
    #warning "OpenMP host"
    #undef HOSTRPC_HOST
    #define HOSTRPC_HOST 1
  #endif
#endif


#if !defined(_OPENMP) && defined(__NVPTX__)
  #if defined (__CUDA__)
    #if defined(__CUDA_ARCH__)
      #warning "Cuda gpu"
      #undef HOSTRPC_NVPTX
      #define HOSTRPC_NVPTX 1
    #else
      #warning "Cuda host"
      #undef HOSTRPC_HOST
      #define HOSTRPC_HOST 1
    #endif
  #else
    #warning "Ptx freestanding"
    #undef HOSTRPC_NVPTX
    #define HOSTRPC_NVPTX 1
  #endif
#endif
  
#if !defined(_OPENMP) && defined (__AMDGCN__)
  #if defined(__HIP__)
    #if defined(__HIP_DEVICE_COMPILE__)
      #warning "Hip gpu"
      #undef HOSTRPC_AMDGCN
      #define HOSTRPC_AMDGCN 1
    #else
      #warning "Hip host"
      #undef HOSTRPC_HOST
      #define HOSTRPC_HOST 1
    #endif
  #else
    #warning "GCN freestanding"
    #undef HOSTRPC_AMDGCN
    #define HOSTRPC_AMDGCN 1
  #endif
#endif

#if (HOSTRPC_AMDGCN + HOSTRPC_NVPTX + HOSTRPC_HOST) != 1
#error "Platform detection failed"
#endif

#endif

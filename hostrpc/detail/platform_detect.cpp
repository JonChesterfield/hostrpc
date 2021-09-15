#include "platform_detect.hpp"

#define NOISY 0

#if NOISY
#if defined(__x86_64__)
#warning "Defined x64"
#endif
#if defined(__AMDGCN__)
#warning "Defined __AMDGCN__"
#endif
#if defined(__NVPTX__)
#warning "Defined __NVPTX__"
#endif

#if defined(__CUDA__)
#warning "Defined __CUDA__"
#endif
#if defined(__CUDA_ARCH__)
#warning "Defined __CUDA_ARCH__"
#endif

#if defined(__HIP__)
#warning "Defined __HIP__"
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#warning "Defined __HIP_DEVICE_COMPILE__"
#endif

#if defined(_OPENMP)
#warning "Defined _OPENMP"
#endif

#endif

#if HOSTRPC_AMDGCN
#warning "Define HOSTRPC_AMDGCN"
#endif
#if HOSTRPC_NVPTX
#warning "Define HOSTRPC_NVPTX"
#endif
#if HOSTRPC_HOST
#warning "Define HOSTRPC_HOST"
#endif

#if 0

#set - x

NOGPU='-nocudainc -nocudalib -nogpuinc -nogpulib -Wno-unused-command-line-argument -Wno-unknown-cuda-version'

CLANG="$HOME/llvm-install/bin/clang++"

echo '#warning "Ptx freestanding"'
$CLANG -g -O2 -emit-llvm -ffreestanding -fno-exceptions -Wno-atomic-alignment --target=nvptx64-nvidia-cuda -march=sm_50 $NOGPU  platform_detect.cpp -c -o /dev/null
  
echo '#warning "GCN freestanding"'
$CLANG -std=c++14 -emit-llvm -O2 -ffreestanding -fno-exceptions --target=amdgcn-amd-amdhsa -march=gfx906 -mcpu=gfx906 $NOGPU platform_detect.cpp -c -o /dev/null

echo '#warning "Cuda gpu"'
$CLANG -x cuda --cuda-gpu-arch=sm_50 $NOGPU  --cuda-path=/usr/local/cuda --cuda-device-only platform_detect.cpp -c -o /dev/null

echo '#warning "Cuda host"'
$CLANG -x cuda --cuda-gpu-arch=sm_50 $NOGPU --cuda-host-only platform_detect.cpp -c -o /dev/null

echo '#warning "Cuda gpu"'
echo '#warning "Cuda host"'
$CLANG -x cuda --cuda-gpu-arch=sm_50 $NOGPU --cuda-path=/usr/local/cuda platform_detect.cpp -c -o /dev/null

echo '#warning "Hip gpu"'
$CLANG -x hip --cuda-gpu-arch=gfx906 $NOGPU --cuda-device-only   platform_detect.cpp -c -o /dev/null

echo '#warning "Hip host"'
$CLANG -x hip --cuda-gpu-arch=gfx906 $NOGPU --cuda-host-only  platform_detect.cpp -c -o /dev/null

echo '#warning "Hip gpu"'
echo '#warning "Hip host"'
$CLANG -x hip --cuda-gpu-arch=gfx906 $NOGPU platform_detect.cpp -c -o /dev/null

#// cuda-host-only / cuda-device-only (or gpu) appear to be ignored in openmp

echo '#warning "OpenMP host"'
echo '#warning "OpenMP gcn gpu"'
$CLANG -std=c++14 -emit-llvm -O2 -fno-exceptions -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 $NOGPU platform_detect.cpp -c -o /dev/null

echo '#warning "OpenMP host"'
echo '#warning "OpenMP ptx gpu"'
$CLANG -std=c++14 -emit-llvm -O2 -fno-exceptions -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_50 $NOGPU platform_detect.cpp -c -o /dev/null

echo '#warning "OpenCL host"'
$CLANG -x cl -Xclang -cl-std=clc++ $NOGPU -emit-llvm -D__OPENCL_C_VERSION__=200 platform_detect.cpp -c -o /dev/null

echo '#warning "OpenCL gcn gpu"'
$CLANG -x cl -Xclang -cl-std=clc++ $NOGPU -emit-llvm -D__OPENCL_C_VERSION__=200 platform_detect.cpp -target amdgcn-amd-amdhsa -c -o /dev/null

echo '#warning "OpenCL ptx gpu"'
$CLANG -x cl -Xclang -cl-std=clc++ $NOGPU -emit-llvm -D__OPENCL_C_VERSION__=200 platform_detect.cpp -target nvptx64-nvidia-cuda -c -o /dev/null

#endif

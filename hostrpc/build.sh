#!/bin/bash
set -x
set -e
set -o pipefail

./clean.sh

DERIVE=${1:-4}

# Aomp
RDIR=$HOME/rocm/aomp
GFX=`$RDIR/bin/mygpu -d gfx906` # lost the entry for gfx750 at some point
DEVICERTL="$RDIR/lib/libdevice/libomptarget-amdgcn-$GFX.bc"

# trunk
RDIR=$HOME/llvm-install
GFX=`$RDIR/bin/amdgpu-arch | uniq`
DEVICERTL="$RDIR/lib/libomptarget-amdgcn-$GFX.bc"

mkdir -p obj
mkdir -p lib

echo "Using toolchain at $RDIR, GFX=$GFX"

have_nvptx=0
if [ -e "/dev/nvidiactl" ]; then
    have_nvptx=1
fi

have_amdgcn=0
if [ -e "/dev/kfd" ]; then
    have_amdgcn=1
fi

if (($have_nvptx)); then
    # Clang looks for this file, but cuda stopped shipping it
    if [ -e /usr/local/cuda/version.txt ]; then
        VER=`cat /usr/local/cuda/version.txt`
        echo "Found version: $VER"
    else
        VER=`/usr/local/cuda/bin/nvcc  --version | awk '/Cuda compilation/ {print $6}'`
        echo "CUDA Version $VER" > /usr/local/cuda/version.txt
    fi
fi

# A poorly named amd-stg-open, does not hang
# RDIR=$HOME/rocm-3.5-llvm-install

# Trunk, hangs
# RDIR=$HOME/llvm-install

HSAINC="$RDIR/include/hsa/"
DEVLIBINC="$HOME/aomp/rocm-device-libs/ockl/inc"
OCKL_DIR="$HOME/rocm/aomp/amdgcn/bitcode"

GFXNUM=`echo $GFX | sed 's$gfx$$'`
if (($have_amdgcn)); then
OCKL_LIBS="$OCKL_DIR/ockl.bc $OCKL_DIR/oclc_isa_version_$GFXNUM.bc $OCKL_DIR/oclc_wavefrontsize64_on.bc"
else
OCKL_LIBS=""
fi

HSALIBDIR="$RDIR/lib"
HSALIB="$HSALIBDIR/libhsa-runtime64.so" # $RDIR/lib/libomptarget.rtl.hsa.so"

# Shouldn't need these, but copying across from initial for reference 
# DLIBS="$RDIR/lib/libdevice/libhostcall-amdgcn-$GFX.bc $RDIR/lib/ockl.amdgcn.bc $RDIR/lib/oclc_wavefrontsize64_on.amdgcn.bc $RDIR/lib/oclc_isa_version_$GFXNUM.amdgcn.bc"

CLANG="$RDIR/bin/clang"
CLANGXX="$RDIR/bin/clang++"
LLC="$RDIR/bin/llc"
DIS="$RDIR/bin/llvm-dis"
LINK="$RDIR/bin/llvm-link"
OPT="$RDIR/bin/opt"

#CLANG="g++"
#LINK="ld -r"

CXX="$CLANGXX -std=c++14 -Wall -Wextra"
LDFLAGS="-pthread $HSALIB -Wl,-rpath=$HSALIBDIR -lelf"

AMDGPU="--target=amdgcn-amd-amdhsa -march=$GFX -mcpu=$GFX -mllvm -amdgpu-fixed-function-abi -Xclang -fconvergent-functions -nogpulib"

PTX_VER="-Xclang -target-feature -Xclang +ptx63"
NVGPU="--target=nvptx64-nvidia-cuda -march=sm_50 $PTX_VER -Xclang -fconvergent-functions"

COMMONFLAGS="-Wall -Wextra -emit-llvm " # -DNDEBUG -Wno-type-limits "
# cuda/openmp pass the host O flag through to ptxas, which crashes on debug info if > 0
X64FLAGS=" -O2 -g -pthread " # nvptx can't handle debug info on x64 for O>0
GCNFLAGS=" -O2 -ffreestanding -fno-exceptions $AMDGPU"
# atomic alignment objection seems reasonable - may want 32 wide atomics on nvptx
# clang/ptx back end is crashing in llvm::DwarfDebug::constructCallSiteEntryDIEs
NVPTXFLAGS=" -O2 -ffreestanding -fno-exceptions -Wno-atomic-alignment -emit-llvm $NVGPU "

CXX_X64="$CLANGXX -std=c++14 $COMMONFLAGS $X64FLAGS"
CXX_GCN="$CLANGXX -std=c++14 $COMMONFLAGS $GCNFLAGS"


CXXCL="$CLANGXX -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 -D__OPENCL__ -D__OPENCL_C_VERSION__=200"
CXXCL_GCN="$CXXCL -emit-llvm -ffreestanding $AMDGPU"
CXXCL_PTX="$CXXCL -emit-llvm -ffreestanding $NVGPU"

TRUNKBIN="$HOME/.emacs.d/bin"
CXX_PTX="$TRUNKBIN/clang++ $NVPTXFLAGS"


XCUDA="-x cuda --cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda"
XHIP="-x hip --cuda-gpu-arch=gfx906 -nogpulib -nogpuinc"
XOPENCL="-x cl -Xclang -cl-std=clc++ -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200  -Dcl_khr_fp64 -Dcl_khr_fp16   -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics" 

CXX_CUDA="$CLANGXX -O2 $COMMONFLAGS $XCUDA -I/usr/local/cuda/include -nocudalib"

CXX_X64_LD="$CXX"
CXX_GCN_LD="$CXX $GCNFLAGS"

if [ ! -f obj/catch.o ]; then
    time $CXX -O3 catch.cpp -c -o obj/catch.o
fi

# Code running on the host can link in host, hsa or cuda support library.
# Fills in gaps in the cuda/hsa libs, implements allocators

$CXX_GCN hostrpc_printf.cpp -O3 -c -o obj/hostrpc_printf.gcn.bc
$CXX_X64 -I$HSAINC hostrpc_printf.cpp -O3 -c -o obj/hostrpc_printf.x64.bc
$CXX_X64 -I$HSAINC incprintf.cpp -O3 -c -o obj/incprintf.x64.bc

# host support library
$CXX_X64 allocator_host_libc.cpp -c -o obj/allocator_host_libc.x64.bc
# wraps pthreads, cuda miscompiled <thread>
$CXX_X64 hostrpc_thread.cpp -c -o obj/hostrpc_thread.x64.bc 
$LINK obj/allocator_host_libc.x64.bc obj/hostrpc_thread.x64.bc -o obj/host_support.x64.bc

# hsa support library
if (($have_amdgcn)); then
$CXX_X64 ../impl/msgpack.cpp -c -o obj/msgpack.x64.bc
$CXX_X64 find_metadata.cpp -c -o obj/find_metadata.x64.bc
$CXX_X64 -I$HSAINC allocator_hsa.cpp -c -o obj/allocator_hsa.x64.bc
$LINK  obj/host_support.x64.bc obj/msgpack.x64.bc obj/find_metadata.x64.bc obj/allocator_hsa.x64.bc obj/hostrpc_printf.x64.bc obj/incprintf.x64.bc -o obj/hsa_support.x64.bc

$CXX_X64 dump_kernels.cpp -I../impl -c -o obj/dump_kernels.x64.bc
$CXX_X64_LD obj/msgpack.x64.bc obj/dump_kernels.x64.bc -lelf -o dump_kernels

$CXX_X64 -I$HSAINC query_system.cpp -c -o obj/query_system.x64.bc
$CXX_X64_LD obj/query_system.x64.bc obj/hsa_support.x64.bc $LDFLAGS -o query_system
fi

# cuda support library
if (($have_nvptx)); then
 $CXX_X64 -I/usr/local/cuda/include allocator_cuda.cpp  -c -emit-llvm -o obj/allocator_cuda.x64.bc
 $LINK obj/host_support.x64.bc obj/allocator_cuda.x64.bc -o obj/cuda_support.x64.bc
fi

# openmp support library
$CXX_X64 -I$RDIR/include allocator_openmp.cpp -c -o obj/allocator_openmp.x64.bc
$CXX_X64 openmp_plugins.cpp -c -o obj/openmp_plugins.x64.bc
$LINK obj/allocator_openmp.x64.bc obj/openmp_plugins.x64.bc -o obj/openmp_support.x64.bc


$CXX_GCN threads.cpp -O3 -c -o threads.gcn.bc
$CXXCL_GCN threads_bootstrap.cpp -O3 -c -o threads_bootstrap.ocl.gcn.bc
$CXX_GCN threads_bootstrap.cpp -O3 -c -o threads_bootstrap.cpp.gcn.bc

$LINK threads.gcn.bc threads_bootstrap.ocl.gcn.bc threads_bootstrap.cpp.gcn.bc obj/hostrpc_printf.gcn.bc | $OPT -O2 -o obj/merged_threads_bootstrap.gcn.bc 
$DIS obj/merged_threads_bootstrap.gcn.bc

$CXX_GCN_LD obj/merged_threads_bootstrap.gcn.bc -o threads_bootstrap.gcn.so

$CXX_X64 threads.cpp -O3 -c -o threads.x64.bc
$CXX_X64 threads_bootstrap.cpp -I$HSAINC -O3 -c -o threads_bootstrap.x64.bc
$CXX_X64_LD threads.x64.bc obj/hsa_support.x64.bc obj/catch.o $LDFLAGS -o threads.x64.exe


$CXX_X64_LD threads_bootstrap.x64.bc obj/hsa_support.x64.bc $LDFLAGS -o threads_bootstrap.x64.exe

./threads_bootstrap.x64.exe

exit


$CXX_GCN x64_gcn_debug.cpp -c -o obj/x64_gcn_debug.gcn.code.bc
$CXXCL_GCN x64_gcn_debug.cpp -c -o obj/x64_gcn_debug.gcn.kern.bc
$LINK obj/x64_gcn_debug.gcn.code.bc obj/x64_gcn_debug.gcn.kern.bc obj/hostrpc_printf.gcn.bc -o obj/x64_gcn_debug.gcn.bc

$CXX_GCN_LD obj/x64_gcn_debug.gcn.bc -o x64_gcn_debug.gcn.so

$CXX_X64 -I$HSAINC x64_gcn_debug.cpp -c -o obj/x64_gcn_debug.x64.bc

$CXX obj/x64_gcn_debug.x64.bc obj/hsa_support.x64.bc $LDFLAGS -o x64_gcn_debug.exe


$CXX_X64 syscall.cpp -c -o obj/syscall.x64.bc 

# amdgcn loader links these, but shouldn't. need to refactor.
$CXX_GCN hostcall.cpp -c -o hostcall.gcn.bc
$CXX_X64 -I$HSAINC hostcall.cpp -c -o hostcall.x64.bc
# Build the device code that uses said library
$CXX_X64 -I$HSAINC amdgcn_main.cpp -c -o amdgcn_main.x64.bc
$CXX_GCN amdgcn_main.cpp -c -o amdgcn_main.gcn.bc

# build loaders - run int main() {} on the gpu

# Register amdhsa elf magic with kernel
# One off
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' > register
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x02\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' > register

# Persistent
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x02\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf

if (($have_amdgcn)); then
  $CXXCL_GCN loader/amdgcn_loader_entry.cl -c -o loader/amdgcn_loader_entry.gcn.bc
  $CXX_GCN loader/opencl_loader_cast.cpp -c -o loader/opencl_loader_cast.gcn.bc
  $LINK loader/amdgcn_loader_entry.gcn.bc loader/opencl_loader_cast.gcn.bc | $OPT -O2 -o amdgcn_loader_device.gcn.bc

  $CXX_X64 -I$HSAINC amdgcn_loader.cpp -c -o loader/amdgcn_loader.x64.bc
  $CXX_X64_LD $LDFLAGS loader/amdgcn_loader.x64.bc obj/hsa_support.x64.bc hostcall.x64.bc amdgcn_main.x64.bc -o ../amdgcn_loader.exe
fi

if (($have_nvptx)); then
 # presently using the cuda entry point but may want the opencl one later
 $CXX_CUDA -std=c++14 --cuda-device-only loader/nvptx_loader_entry.cu -c -emit-llvm -o loader/nvptx_loader_entry.cu.ptx.bc   
 $CXXCL_PTX loader/nvptx_loader_entry.cl -c -o loader/nvptx_loader_entry.cl.ptx.bc
 $CXX_PTX loader/opencl_loader_cast.cpp -c -o loader/opencl_loader_cast.ptx.bc

 $CLANGXX nvptx_loader.cpp obj/cuda_support.x64.bc --cuda-path=/usr/local/cuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcuda -lcudart -pthread -o ../nvptx_loader.exe
fi


$CXX_GCN pool_interface.cpp -O3 -c -o obj/pool_interface.gcn.bc
$CXX_X64 pool_interface.cpp -O3 -c -o obj/pool_interface.x64.bc

$CXX_X64_LD -pthread obj/pool_interface.x64.bc -o obj/pool_interface.x64.exe

$LINK obj/hostrpc_printf.gcn.bc amdgcn_loader_device.gcn.bc obj/pool_interface.gcn.bc -o obj/pool_interface.merged.gcn.bc
$CXX_GCN_LD obj/pool_interface.merged.gcn.bc -o obj/pool_interface.gcn.exe


if (($have_amdgcn)); then
$CLANG -std=c11 $COMMONFLAGS $GCNFLAGS test_example.c -c -o obj/test_example.gcn.bc
$LINK obj/test_example.gcn.bc obj/hostrpc_printf.gcn.bc amdgcn_loader_device.gcn.bc -o test_example.gcn.bc
$CXX_GCN_LD test_example.gcn.bc -o test_example.gcn
set +e
./test_example.gcn
set -e
fi

$CLANG -std=c11 -I$HSAINC $COMMONFLAGS $X64FLAGS printf_test.c -c -o obj/printf_test.x64.bc
if (($have_amdgcn)); then
    $CLANG -std=c11 $COMMONFLAGS $GCNFLAGS printf_test.c -c -o obj/printf_test.gcn.bc
    $LINK obj/printf_test.gcn.bc obj/hostrpc_printf.gcn.bc amdgcn_loader_device.gcn.bc -o printf_test.gcn.bc
    $CXX_GCN_LD printf_test.gcn.bc -o printf_test.gcn
fi

if (($have_amdgcn)); then
    $CXX_GCN devicertl_pteam_mem_barrier.cpp -c -o obj/devicertl_pteam_mem_barrier.gcn.bc
    # todo: refer to lib from RDIR, once that lib has the function non-static    
    $LINK obj/devicertl_pteam_mem_barrier.gcn.bc obj/hostrpc_printf.gcn.bc amdgcn_loader_device.gcn.bc -o devicertl_pteam_mem_barrier.gcn.bc $DEVICERTL
    $CXX_GCN_LD devicertl_pteam_mem_barrier.gcn.bc -o devicertl_pteam_mem_barrier.gcn
    set +e
    echo "This is failing at present, HSA doesn't think the binary is valid"
    ./devicertl_pteam_mem_barrier.gcn
    set -e
fi

$CXX_X64 prototype/states.cpp -c -o prototype/states.x64.bc

$CXX_GCN run_on_hsa_example.cpp -c -o obj/run_on_hsa_example.cxx.gcn.bc
$CXXCL_GCN run_on_hsa_example.cpp -c -o obj/run_on_hsa_example.ocl.gcn.bc
$LINK obj/run_on_hsa_example.cxx.gcn.bc obj/run_on_hsa_example.ocl.gcn.bc -o obj/run_on_hsa_example.gcn.bc

$CXX_GCN_LD obj/run_on_hsa_example.gcn.bc -o lib/run_on_hsa_example.gcn.so

$CXX_X64 -I$HSAINC run_on_hsa_example.cpp -c -o obj/run_on_hsa_example.cxx.x64.bc
$CXX_X64 -I$HSAINC run_on_hsa.cpp -c -o obj/run_on_hsa.x64.bc

$CXX $LDFLAGS obj/run_on_hsa_example.cxx.x64.bc obj/run_on_hsa.x64.bc obj/hsa_support.x64.bc -o run_on_hsa.exe

./run_on_hsa.exe

if true; then
# Sanity checks that the client and server compile successfully
# and provide an example of the generated IR
$CXX_X64 codegen/client.cpp -S -o codegen/client.x64.ll
$CXX_X64 codegen/server.cpp -S -o codegen/server.x64.ll
$CXX_GCN codegen/client.cpp -S -o codegen/client.gcn.ll
$CXX_GCN codegen/server.cpp -S -o codegen/server.gcn.ll
$CXX_PTX codegen/client.cpp -S -o codegen/client.ptx.ll
$CXX_PTX codegen/server.cpp -S -o codegen/server.ptx.ll

$CXX_X64 codegen/foo_cxx.cpp -S -o codegen/foo_cxx.x64.ll
$CXX_GCN codegen/foo_cxx.cpp -S -o codegen/foo_cxx.gcn.ll
$CXX_PTX codegen/foo_cxx.cpp -S -o codegen/foo_cxx.ptx.ll

$TRUNKBIN/clang++ $XCUDA -std=c++14 --cuda-device-only -nocudainc -nocudalib codegen/foo.cu -emit-llvm -S -o codegen/foo.cuda.ptx.ll

$TRUNKBIN/clang++ $XCUDA -std=c++14 --cuda-host-only -nocudainc -nocudalib codegen/foo.cu -emit-llvm -S -o codegen/foo.cuda.x64.ll

cd codegen
$TRUNKBIN/clang++ $XCUDA -std=c++14 -nocudainc -nocudalib foo.cu -emit-llvm -S
mv foo.ll foo.cuda.both_x64.ll
mv foo-cuda-nvptx64-nvidia-cuda-*.ll foo.cuda.both_ptx.ll
cd -


# aomp has broken cuda-device-only
$TRUNKBIN/clang++ -x hip --cuda-gpu-arch=gfx906 -nogpulib -std=c++14 -O1 --cuda-device-only codegen/foo.cu -emit-llvm -S -o codegen/foo.hip.gcn.ll
$TRUNKBIN/clang++ -x hip --cuda-gpu-arch=gfx906 -nogpulib -std=c++14 -O1 --cuda-host-only codegen/foo.cu -emit-llvm -S -o codegen/foo.hip.x64.ll

# hip doesn't understand -emit-llvm (or -S, or -c) when trying to do host and device together
# so can't test that here

# This ignores -S for some reason
$CLANGXX -O2 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GFX  codegen/foo.omp.cpp -c -emit-llvm --cuda-device-only -o codegen/foo.omp.gcn.bc && $DIS codegen/foo.omp.gcn.bc && rm codegen/foo.omp.gcn.bc

# ignores host-only, so the IR has a binary gfx pasted at the top
$CLANGXX -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GFX  codegen/foo.omp.cpp -S -emit-llvm --cuda-host-only -o codegen/foo.omp.gcn-x64.ll


$CLANGXX -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_50  codegen/foo.omp.cpp -c -emit-llvm -S --cuda-device-only -o codegen/foo.omp.ptx.ll

$CLANGXX -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_50  codegen/foo.omp.cpp -c -emit-llvm -S --cuda-host-only -o codegen/foo.omp.ptx-x64.ll

# OpenCL compilation model is essentially that of c++
$CLANGXX $XOPENCL -S -emit-llvm codegen/foo_cxx.cpp -S -o codegen/foo.cl.x64.ll

$CLANGXX $XOPENCL -S -nogpulib -emit-llvm -target amdgcn-amd-amdhsa -mcpu=$GFX codegen/foo_cxx.cpp -S -o codegen/foo.cl.gcn.ll

# recognises mcpu but warns that it is unused
$CLANGXX $XOPENCL -S -nogpulib -emit-llvm -target nvptx64-nvidia-cuda codegen/foo_cxx.cpp -S -o codegen/foo.cl.ptx.ll



$CLANGXX $XCUDA -std=c++14 --cuda-device-only -nocudainc -nocudalib codegen/client.cpp -emit-llvm -S -o codegen/client.cuda.ptx.ll
$CLANGXX $XCUDA -std=c++14 --cuda-host-only -nocudainc -nocudalib codegen/client.cpp -emit-llvm -S -o codegen/client.cuda.x64.ll


# HIP has excessive requirements on function annotation that cuda does not, ignore for now
# Fails to annotate CFG at O0
$CLANGXX $XHIP -std=c++14 -O1 --cuda-device-only codegen/client.cpp -S -o codegen/client.hip.gcn.ll
$CLANGXX $XHIP -std=c++14 -O1 --cuda-host-only codegen/client.cpp -S -o codegen/client.hip.x64.ll
$CLANGXX $XHIP -std=c++14 -O1 --cuda-device-only codegen/server.cpp -S -o codegen/server.hip.gcn.ll
$CLANGXX $XHIP -std=c++14 -O1 --cuda-host-only codegen/server.cpp -S -o codegen/server.hip.x64.ll

# Build as opencl/c++ too
$CLANGXX $XOPENCL -S -emit-llvm codegen/client.cpp -S -o codegen/client.ocl.x64.ll
$CLANGXX $XOPENCL -S -emit-llvm codegen/server.cpp -S -o codegen/server.ocl.x64.ll
fi

$CXX_X64 -I$HSAINC tests.cpp -c -o tests.x64.bc
$CXX_X64 -I$HSAINC x64_x64_stress.cpp -c -o x64_x64_stress.x64.bc

$CXX_GCN -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.code.bc
$CXXCL_GCN -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.kern.bc
$LINK x64_gcn_stress.gcn.code.bc x64_gcn_stress.gcn.kern.bc -o x64_gcn_stress.gcn.bc
$CXX_GCN_LD x64_gcn_stress.gcn.bc -o x64_gcn_stress.gcn.so
$CXX_X64 -DDERIVE_VAL=$DERIVE -I$HSAINC x64_gcn_stress.cpp -c -o x64_gcn_stress.x64.bc

# $CXX_GCN -D__HAVE_ROCR_HEADERS=1 -I$HSAINC -I$DEVLIBINC persistent_kernel.cpp -c -o persistent_kernel.gcn.code.bc

$CXX_GCN -D__HAVE_ROCR_HEADERS=0 persistent_kernel.cpp -c -o persistent_kernel.gcn.code.bc

$CXXCL_GCN persistent_kernel.cpp -c -o persistent_kernel.gcn.kern.bc
$LINK persistent_kernel.gcn.code.bc persistent_kernel.gcn.kern.bc $OCKL_LIBS -o persistent_kernel.gcn.bc
$CXX_GCN_LD persistent_kernel.gcn.bc -o persistent_kernel.gcn.so
$CXX_X64 -I$HSAINC persistent_kernel.cpp -c -o persistent_kernel.x64.bc

$CXX_CUDA -std=c++14 --cuda-device-only -nogpuinc -nobuiltininc $PTX_VER detail/platform.cu -c -emit-llvm -o detail/platform.ptx.bc

if (($have_amdgcn)); then
    # Tries to treat foo.so as a hip input file. Somewhat surprised, but might be right.
    # The clang driver can't handle some hip input + some bitcode input, but does have the
    # internal hook -mlink-builtin-bitcode that can be used to the same end effect
    $LINK obj/hsa_support.x64.bc obj/syscall.x64.bc -o obj/demo.hip.link.x64.bc

    # hip presently fails to build, so the library will be missing
    # $CLANGXX -I$HSAINC -std=c++11 -x hip demo.hip -o demo --offload-arch=gfx906 -Xclang -mlink-builtin-bitcode -Xclang obj/demo.hip.link.x64.bc -L$HOME/rocm/aomp/hip -L$HOME/rocm/aomp/lib -lamdhip64 -L$HSALIBDIR -lhsa-runtime64 -Wl,-rpath=$HSALIBDIR -pthread -ldl
    # ./demo hsa runtime presently segfaults in hip's library
fi

$CXX_PTX nvptx_main.cpp -ffreestanding -c -o nvptx_main.ptx.bc

if (($have_nvptx)); then
# One step at a time
    $CLANGXX $XCUDA hello.cu -o hello -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcuda -lcudart_static -ldl -lrt -pthread && ./hello

# hello.o is an executable elf, may be able to load it from cuda
$CLANGXX $XCUDA -std=c++14 hello.cu --cuda-device-only $PTX_VER -c -o hello.o  -I/usr/local/cuda/include


# ./../nvptx_loader.exe hello.o

fi

$CLANGXX -std=c++14 -Wall -Wextra -O0 -g test_storage.cpp obj/openmp_support.x64.bc obj/host_support.x64.bc $RDIR/lib/libomptarget.so -o test_storage.exe -pthread -ldl -Wl,-rpath=$RDIR/lib && valgrind ./test_storage.exe

if (($have_amdgcn)); then
    $LINK obj/openmp_support.x64.bc obj/hsa_support.x64.bc obj/syscall.x64.bc -o obj/demo_bitcode_gcn.omp.bc

# openmp is taking an excessive amount of time to compile, drop it for now
    $CLANGXX -I$HSAINC -O2 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GFX  -DDEMO_AMDGCN=1 demo_openmp.cpp -Xclang -mlink-builtin-bitcode -Xclang obj/demo_bitcode_gcn.omp.bc -o demo_openmp_gcn -pthread -ldl $HSALIB -Wl,-rpath=$HSALIBDIR && ./demo_openmp_gcn
fi

if (($have_nvptx)); then
    $LINK obj/openmp_support.x64.bc obj/cuda_support.x64.bc obj/syscall.x64.bc -o demo_bitcode_ptx.omp.bc
    
    $CLANGXX -I$HSAINC -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_50 -I/usr/local/cuda/include -DDEMO_NVPTX=1 demo_openmp.cpp -Xclang -mlink-builtin-bitcode -Xclang demo_bitcode_ptx.omp.bc -Xclang -mlink-builtin-bitcode -Xclang detail/platform.ptx.bc -o demo_openmp_ptx -L/usr/local/cuda/lib64/ -lcuda -lcudart_static -ldl -lrt -pthread && ./demo_openmp_ptx
fi




# Build the device library that calls into main()



$LINK amdgcn_main.gcn.bc amdgcn_loader_device.gcn.bc  hostcall.gcn.bc obj/hostrpc_printf.gcn.bc -o executable_device.gcn.bc

# Link the device image
$CXX_GCN_LD executable_device.gcn.bc -o a.gcn.out

if (($have_nvptx)); then

"$TRUNKBIN/llvm-link" nvptx_main.ptx.bc loader/nvptx_loader_entry.cu.ptx.bc detail/platform.ptx.bc -o executable_device.ptx.bc

$LINK nvptx_main.ptx.bc loader/nvptx_loader_entry.cu.ptx.bc detail/platform.ptx.bc -o executable_device.ptx.bc


$CLANGXX --target=nvptx64-nvidia-cuda -march=sm_50 $PTX_VER executable_device.ptx.bc -S -o executable_device.ptx.s

/usr/local/cuda/bin/ptxas -m64 -O0 --gpu-name sm_50 executable_device.ptx.s -o a.ptx.out
./../nvptx_loader.exe a.ptx.out
fi


# llc seems to need to be told what architecture it's disassembling

# for bc in `find . -type f -iname '*.x64.bc'` ; do
#     ll=`echo $bc | sed 's_.bc_.ll_g'`
#     $OPT -strip-debug $bc -S -o $ll
#     $LLC $ll
# done
# 
# for bc in `find . -type f -iname '*.gcn.bc'` ; do
#     ll=`echo $bc | sed 's_.bc_.ll_g'`
#     obj=`echo $bc | sed 's_.bc_.obj_g'`
#     $OPT -strip-debug $bc -S -o $ll
#     $LLC --mcpu=$GFX -amdgpu-fixed-function-abi $ll
#     $CXX_GCN_LD -c $ll -o $obj
# done

# $CXX_X64_LD tests.x64.bc prototype/states.x64.bc obj/catch.o obj/allocator_host_libc.x64.bc $LDFLAGS -o prototype/states.exe

$CXX_X64_LD prototype/states.x64.bc obj/catch.o $LDFLAGS -o prototype/states.exe

$CXX_X64_LD x64_x64_stress.x64.bc obj/host_support.x64.bc obj/catch.o $LDFLAGS -o x64_x64_stress.exe

$CXX_X64_LD x64_gcn_stress.x64.bc obj/hsa_support.x64.bc obj/catch.o $LDFLAGS -o x64_gcn_stress.exe

$CXX_X64_LD tests.x64.bc obj/host_support.x64.bc obj/catch.o $LDFLAGS -o tests.exe


# clang trunk is crashing on this at present
set +e
$CXX_X64_LD persistent_kernel.x64.bc obj/catch.o obj/hsa_support.x64.bc $LDFLAGS -o persistent_kernel.exe
set -e

time valgrind --leak-check=full --fair-sched=yes ./prototype/states.exe


set +e # Keep running tests after one fails

./threads.x64.exe

if (($have_amdgcn)); then
    ./x64_gcn_debug.exe
fi

# ./threads_bootstrap.x64.exe crashes the vega902 gui at present

if (($have_amdgcn)); then
time ./persistent_kernel.exe
fi

time ./tests.exe
time ./x64_x64_stress.exe

if (($have_amdgcn)); then
echo "Call hostcall/loader executable"
time ./a.gcn.out ; echo $?
fi

if (($have_amdgcn)); then
echo "Call x64_gcn_stress: Derive $DERIVE"
time ./x64_gcn_stress.exe
fi


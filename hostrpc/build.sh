#!/bin/bash
set -x
set -e
set -o pipefail

./clean.sh

DERIVE=${1:-4}
# Aomp
RDIR=$HOME/rocm/aomp

CLANGINCLUDE=$RDIR/lib/clang/11.0.0/include

# Needs to resolve to gfx906, gfx1010 or similar
GFX=`$RDIR/bin/mygpu -d gfx906` # lost the entry for gfx750 at some point

mkdir -p obj
mkdir -p lib

have_nvptx=0
if [ -e "/dev/nvidiactl" ]; then
    have_nvptx=1
fi

have_amdgcn=0
if [ -e "/dev/kfd" ]; then
    have_amdgcn=1
fi

if ((!$have_amdgcn)); then
    # Compile for a gfx906 as a credible default
    GFX=gfx906
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

HSAINC="$HOME/aomp/rocr-runtime/src/inc/"
DEVLIBINC="$HOME/aomp/rocm-device-libs/ockl/inc"
OCKL_DIR="$HOME/rocm/aomp/amdgcn/bitcode"

GFXNUM=`echo $GFX | sed 's$gfx$$'`
if (($have_amdgcn)); then
OCKL_LIBS="$OCKL_DIR/ockl.bc $OCKL_DIR/oclc_isa_version_$GFXNUM.bc $OCKL_DIR/oclc_wavefrontsize64_on.bc"
else
OCKL_LIBS=""
fi

HSALIBDIR="$HOME/rocm/aomp/lib"
HSALIB="$HSALIBDIR/libhsa-runtime64.so" # $RDIR/lib/libomptarget.rtl.hsa.so"

# Shouldn't need these, but copying across from initial for reference 
# DLIBS="$RDIR/lib/libdevice/libhostcall-amdgcn-$GFX.bc $RDIR/lib/ockl.amdgcn.bc $RDIR/lib/oclc_wavefrontsize64_on.amdgcn.bc $RDIR/lib/oclc_isa_version_$GFXNUM.amdgcn.bc"

CLANG="$RDIR/bin/clang++"
LLC="$RDIR/bin/llc"
DIS="$RDIR/bin/llvm-dis"
LINK="$RDIR/bin/llvm-link"
OPT="$RDIR/bin/opt"

#CLANG="g++"
#LINK="ld -r"

CXX="$CLANG -std=c++14 -Wall -Wextra"
LDFLAGS="-pthread $HSALIB -Wl,-rpath=$HSALIBDIR obj/hsa_support.x64.bc -lelf"


NOINC="-nostdinc -nostdinc++ -isystem $CLANGINCLUDE -DHOSTRPC_HAVE_STDIO=0"

AMDGPU="--target=amdgcn-amd-amdhsa -march=$GFX -mcpu=$GFX -mllvm -amdgpu-fixed-function-abi -nogpulib"

# Not sure why CUDACC isn't being set by clang here, probably a bad sign
PTX_VER="-Xclang -target-feature -Xclang +ptx63"

NVGPU="--target=nvptx64-nvidia-cuda -march=sm_50 $PTX_VER -D__CUDACC__"

COMMONFLAGS="-Wall -Wextra -emit-llvm " # -DNDEBUG -Wno-type-limits "
# cuda/openmp pass the host O flag through to ptxas, which crashes on debug info if > 0
X64FLAGS=" -O0 -g -pthread " # nvptx can't handle debug info on x64 for O>0
GCNFLAGS=" -O2 -ffreestanding -fno-exceptions $AMDGPU"
# atomic alignment objection seems reasonable - may want 32 wide atomics on nvptx
# clang/ptx back end is crashing in llvm::DwarfDebug::constructCallSiteEntryDIEs
NVPTXFLAGS=" -O2 -ffreestanding -fno-exceptions -Wno-atomic-alignment -emit-llvm $NVGPU "

CXX_X64="$CLANG -std=c++14 $COMMONFLAGS $X64FLAGS"
CXX_GCN="$CLANG -std=c++14 $COMMONFLAGS $GCNFLAGS"


CXXCL="$CLANG -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 -D__OPENCL__"
CXXCL_GCN="$CXXCL -emit-llvm -ffreestanding $AMDGPU"
CXXCL_PTX="$CXXCL -emit-llvm -ffreestanding $NVGPU"

TRUNKBIN="$HOME/.emacs.d/bin"
CXX_PTX="$TRUNKBIN/clang++ $NVPTXFLAGS"


XCUDA="-x cuda --cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda"
XHIP="-x hip --cuda-gpu-arch=gfx906 -nogpulib -nogpuinc"
XOPENCL="-x cl -Xclang -cl-std=clc++ -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200  -Dcl_khr_fp64 -Dcl_khr_fp16   -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics" 

CXX_CUDA="$CLANG -O2 $COMMONFLAGS $XCUDA -I/usr/local/cuda/include -nocudalib"

CXX_X64_LD="$CXX"
CXX_GCN_LD="$CXX $GCNFLAGS"


# Code running on the host can link in host, hsa or cuda support library.
# Fills in gaps in the cuda/hsa libs, implements allocators

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
$LINK  obj/host_support.x64.bc obj/msgpack.x64.bc obj/find_metadata.x64.bc obj/allocator_hsa.x64.bc -o obj/hsa_support.x64.bc
fi

# cuda support library
if (($have_nvptx)); then
 $CXX_X64 -I/usr/local/cuda/include allocator_cuda.cpp  -c -emit-llvm -o obj/allocator_cuda.x64.bc
 $LINK obj/host_support.x64.bc obj/allocator_cuda.x64.bc -o obj/cuda_support.x64.bc
fi


# loader bitcode
if (($have_amdgcn)); then
  $CXXCL_GCN loader/amdgcn_loader_entry.cl -c -o loader/amdgcn_loader_entry.gcn.bc
  $CXX_GCN loader/opencl_loader_cast.cpp -c -o loader/opencl_loader_cast.gcn.bc
  $LINK loader/amdgcn_loader_entry.gcn.bc loader/opencl_loader_cast.gcn.bc | $OPT -O2 -o amdgcn_loader_device.gcn.bc
  $CXX_X64_LD $LDFLAGS loader/amdgcn_loader.x64.bc obj/hsa_support.x64.bc hostcall.x64.bc amdgcn_main.x64.bc -o ../amdgcn_loader.exe
fi

if (($have_nvptx)); then

 # presently using the cuda entry point but may want the opencl one later
 $CXX_CUDA -std=c++14 --cuda-device-only loader/nvptx_loader_entry.cu -c -emit-llvm -o loader/nvptx_loader_entry.cu.ptx.bc   
 $CXXCL_PTX loader/nvptx_loader_entry.cl -c -o loader/nvptx_loader_entry.cl.ptx.bc
 $CXX_PTX loader/opencl_loader_cast.cpp -c -o loader/opencl_loader_cast.ptx.bc


 $CLANG nvptx_loader.cpp obj/cuda_support.x64.bc --cuda-path=/usr/local/cuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcuda -lcudart -pthread -o ../nvptx_loader.exe
fi

if [ ! -f obj/catch.o ]; then
    time $CXX -O3 catch.cpp -c -o obj/catch.o
fi

$CXX_X64 states.cpp -c -o states.x64.bc

$CXX_X64 openmp_plugins.cpp -c -o openmp_plugins.x64.bc

# Checking cross platform compilation for simple case

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
$CLANG -O2 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GFX  codegen/foo.omp.cpp -c -emit-llvm --cuda-device-only -o codegen/foo.omp.gcn.bc && $DIS codegen/foo.omp.gcn.bc && rm codegen/foo.omp.gcn.bc

# ignores host-only, so the IR has a binary gfx pasted at the top
$CLANG -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GFX  codegen/foo.omp.cpp -S -emit-llvm --cuda-host-only -o codegen/foo.omp.gcn-x64.ll


$CLANG -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_50  codegen/foo.omp.cpp -c -emit-llvm -S --cuda-device-only -o codegen/foo.omp.ptx.ll

$CLANG -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_50  codegen/foo.omp.cpp -c -emit-llvm -S --cuda-host-only -o codegen/foo.omp.ptx-x64.ll

# OpenCL compilation model is essentially that of c++
$CLANG $XOPENCL -S -emit-llvm codegen/foo_cxx.cpp -S -o codegen/foo.cl.x64.ll

$CLANG $XOPENCL -S -nogpulib -emit-llvm -target amdgcn-amd-amdhsa -mcpu=$GFX codegen/foo_cxx.cpp -S -o codegen/foo.cl.gcn.ll

# recognises mcpu but warns that it is unused
$CLANG $XOPENCL -S -nogpulib -emit-llvm -target nvptx64-nvidia-cuda codegen/foo_cxx.cpp -S -o codegen/foo.cl.ptx.ll

# Sanity check that the client and server compile successfully
# and provide an example of the generated IR
# TODO: Get these building with cuda, hip, openmp-ptx, openmp-gcn too
$CXX_X64 codegen/client.cpp -S -o codegen/client.x64.ll
$CXX_X64 codegen/server.cpp -S -o codegen/server.x64.ll
$CXX_GCN codegen/client.cpp -S -o codegen/client.gcn.ll
$CXX_GCN codegen/server.cpp -S -o codegen/server.gcn.ll
$CXX_PTX codegen/client.cpp -S -o codegen/client.ptx.ll
$CXX_PTX codegen/server.cpp -S -o codegen/server.ptx.ll


$CLANG $XCUDA -std=c++14 --cuda-device-only -nocudainc -nocudalib codegen/client.cpp -emit-llvm -S -o codegen/client.cuda.ptx.ll
$CLANG $XCUDA -std=c++14 --cuda-host-only -nocudainc -nocudalib codegen/client.cpp -emit-llvm -S -o codegen/client.cuda.x64.ll


# HIP has excessive requirements on function annotation that cuda does not, ignore for now
# Fails to annotate CFG at O0
$CLANG $XHIP -std=c++14 -O1 --cuda-device-only codegen/client.cpp -S -o codegen/client.hip.gcn.ll
$CLANG $XHIP -std=c++14 -O1 --cuda-host-only codegen/client.cpp -S -o codegen/client.hip.x64.ll
$CLANG $XHIP -std=c++14 -O1 --cuda-device-only codegen/server.cpp -S -o codegen/server.hip.gcn.ll
$CLANG $XHIP -std=c++14 -O1 --cuda-host-only codegen/server.cpp -S -o codegen/server.hip.x64.ll

# Build as opencl/c++ too
$CLANG $XOPENCL -S -emit-llvm codegen/client.cpp -S -o codegen/client.ocl.x64.ll
$CLANG $XOPENCL -S -emit-llvm codegen/server.cpp -S -o codegen/server.ocl.x64.ll


$CXX_X64 -I$RDIR/include allocator_openmp.cpp -c -o allocator_openmp.x64.bc

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
    $CLANG -I$HSAINC -std=c++11 -x hip demo.hip -o demo --offload-arch=gfx906 -Xclang -mlink-builtin-bitcode -Xclang obj/hsa_support.x64.bc -L$HOME/rocm/aomp/hip -L$HOME/rocm/aomp/lib -lamdhip64 -L$HSALIBDIR -lhsa-runtime64 -Wl,-rpath=$HSALIBDIR && ./demo
fi

$CXX_PTX nvptx_main.cpp -ffreestanding -c -o nvptx_main.ptx.bc

if (($have_nvptx)); then
# One step at a time
    $CLANG $XCUDA hello.cu -o hello -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart_static -ldl -lrt -pthread && ./hello

# hello.o is an executable elf, may be able to load it from cuda
$CLANG $XCUDA -std=c++14 hello.cu --cuda-device-only $PTX_VER -c -o hello.o  -I/usr/local/cuda/include


# ./../nvptx_loader.exe hello.o

fi

$LINK allocator_openmp.x64.bc openmp_plugins.x64.bc -o demo_bitcode.common.x64.bc

if (($have_amdgcn)); then
    $LINK demo_bitcode.common.x64.bc obj/hsa_support.x64.bc -o demo_bitcode.omp.bc
    
    $CLANG -I$HSAINC -O2 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GFX  demo_openmp.cpp -Xclang -mlink-builtin-bitcode -Xclang demo_bitcode.omp.bc -o demo_openmp.gcn -pthread -ldl $HSALIB -Wl,-rpath=$HSALIBDIR && ./demo_openmp.gcn
fi

if (($have_nvptx)); then
    $LINK demo_bitcode.common.x64.bc obj/cuda_support.x64.bc -o demo_bitcode.omp.bc
    
    $CLANG -I$HSAINC -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_50 demo_openmp.cpp -Xclang -mlink-builtin-bitcode -Xclang demo_bitcode.omp.bc -Xclang -mlink-builtin-bitcode -Xclang detail/platform.ptx.bc -o demo_openmp.ptx -L/usr/local/cuda/lib64/ -lcudart_static -ldl -lrt -pthread && ./demo_openmp.ptx
fi


$CXX_GCN hostcall.cpp -c -o hostcall.gcn.bc
$CXX_X64 -I$HSAINC hostcall.cpp -c -o hostcall.x64.bc

# Build the device code that uses said library
$CXX_X64 -I$HSAINC amdgcn_main.cpp -c -o amdgcn_main.x64.bc
$CXX_GCN amdgcn_main.cpp -c -o amdgcn_main.gcn.bc


# Build the device loader that assumes the device library is linked into the application
# TODO: Embed it directly in the loader by patching call to main, as the loader doesn't do it
$CXX_X64 -I$HSAINC amdgcn_loader.cpp -c -o amdgcn_loader.x64.bc


# Build the device library that calls into main()



$LINK amdgcn_main.gcn.bc amdgcn_loader_device.gcn.bc  hostcall.gcn.bc  -o executable_device.gcn.bc

# Link the device image
$CXX_GCN_LD executable_device.gcn.bc -o a.gcn.out

if (($have_nvptx)); then

"$TRUNKBIN/llvm-link" nvptx_main.ptx.bc loader/nvptx_loader_entry.cu.ptx.bc detail/platform.ptx.bc -o executable_device.ptx.bc

$LINK nvptx_main.ptx.bc loader/nvptx_loader_entry.cu.ptx.bc detail/platform.ptx.bc -o executable_device.ptx.bc


$CLANG --target=nvptx64-nvidia-cuda -march=sm_50 $PTX_VER executable_device.ptx.bc -S -o executable_device.ptx.s

/usr/local/cuda/bin/ptxas -m64 -O0 --gpu-name sm_50 executable_device.ptx.s -o a.ptx.out
./../nvptx_loader.exe a.ptx.out
fi

# Register amdhsa elf magic with kernel
# One off
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' > register

# Persistent
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf


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

$CXX_X64_LD tests.x64.bc x64_x64_stress.x64.bc states.x64.bc obj/catch.o obj/allocator_host_libc.x64.bc $LDFLAGS -o states.exe

$CXX_X64_LD x64_x64_stress.x64.bc obj/host_support.x64.bc obj/catch.o $LDFLAGS -o x64_x64_stress.exe

$CXX_X64_LD x64_gcn_stress.x64.bc obj/hsa_support.x64.bc obj/catch.o $LDFLAGS -o x64_gcn_stress.exe

$CXX_X64_LD tests.x64.bc obj/host_support.x64.bc obj/catch.o $LDFLAGS -o tests.exe


$CXX_X64_LD persistent_kernel.x64.bc obj/catch.o obj/hsa_support.x64.bc $LDFLAGS -o persistent_kernel.exe

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

# time valgrind --leak-check=full --fair-sched=yes ./states.exe hazard

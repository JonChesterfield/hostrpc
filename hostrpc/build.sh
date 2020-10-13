#!/bin/bash
set -x
set -e
set -o pipefail

./clean.sh

DERIVE=${1:-4}
# Aomp
RDIR=$HOME/rocm/aomp

# Needs to resolve to gfx906, gfx1010 or similar
GFX=`$RDIR/bin/mygpu -d gfx906` # lost the entry for gfx750 at some point


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

HSALIBDIR="$HOME/rocm/aomp/hsa/lib/"
HSALIB="$HSALIBDIR/libhsa-runtime64.so" # $RDIR/lib/libomptarget.rtl.hsa.so"

# Shouldn't need these, but copying across from initial for reference 
# DLIBS="$RDIR/lib/libdevice/libhostcall-amdgcn-$GFX.bc $RDIR/lib/ockl.amdgcn.bc $RDIR/lib/oclc_wavefrontsize64_on.amdgcn.bc $RDIR/lib/oclc_isa_version_$GFXNUM.amdgcn.bc"

CLANG="$RDIR/bin/clang++"
LLC="$RDIR/bin/llc"
LINK="$RDIR/bin/llvm-link"
OPT="$RDIR/bin/opt"

#CLANG="g++"
#LINK="ld -r"

CXX="$CLANG -std=c++14 -Wall -Wextra"
LDFLAGS="-pthread $HSALIB -Wl,-rpath=$HSALIBDIR hsa_support.bc -lelf"

AMDGPU="--target=amdgcn-amd-amdhsa -march=$GFX -mcpu=$GFX -mllvm -amdgpu-fixed-function-abi -nogpulib"

# Not sure why CUDACC isn't being set by clang here, probably a bad sign
NVGPU="--target=nvptx64-nvidia-cuda -march=sm_50 -D__CUDACC__"

COMMONFLAGS="-Wall -Wextra -emit-llvm " # -DNDEBUG -Wno-type-limits "
X64FLAGS=" -O2 -pthread -g"
GCNFLAGS=" -O2 -ffreestanding -fno-exceptions $AMDGPU"
# atomic alignment objection seems reasonable - may want 32 wide atomics on nvptx
NVPTXFLAGS="-g -O2 -emit-llvm -ffreestanding -fno-exceptions -Wno-atomic-alignment $NVGPU "

CXX_X64="$CLANG -std=c++14 $COMMONFLAGS $X64FLAGS"
CXX_GCN="$CLANG -std=c++14 $COMMONFLAGS $GCNFLAGS"

CXX_X64_LD="$CXX"
CXX_GCN_LD="$CXX $GCNFLAGS"

CXXCL="$CLANG -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 -emit-llvm -D__OPENCL__ -ffreestanding $AMDGPU"

CXX_PTX="$HOME/.emacs.d/bin/clang++ $NVPTXFLAGS"

CXX_CUDA="$CLANG -O2 $COMMONFLAGS -xcuda --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_50 -I/usr/local/cuda/include"

# msgpack, assumed to be checked out ../ from here
$CXX_X64 ../impl/msgpack.cpp -c -o msgpack.bc
$CXX_X64 find_metadata.cpp -c -o find_metadata.bc
$LINK msgpack.bc find_metadata.bc -o hsa_support.bc

if [ ! -f catch.o ]; then
    time $CXX -O3 catch.cpp -c -o catch.o
fi

$CXX_X64 states.cpp -c -o states.x64.bc

XCUDA="-x cuda --cuda-gpu-arch=sm_50 --cuda-path=/usr/local/cuda"
XHIP="-x hip --cuda-gpu-arch=gfx906 -nogpulib -nogpuinc"

# Sanity check that the client and server compile successfully
# and provide an example of the generated IR
# TODO: Get these building with cuda, hip, openmp-ptx, openmp-gcn too
$CXX_X64 codegen/client.cpp -S -o codegen/client.x64.ll
$CXX_X64 codegen/server.cpp -S -o codegen/server.x64.ll
$CXX_GCN codegen/client.cpp -S -o codegen/client.gcn.ll
$CXX_GCN codegen/server.cpp -S -o codegen/server.gcn.ll
$CXX_PTX codegen/client.cpp -S -o codegen/client.ptx.ll
$CXX_PTX codegen/server.cpp -S -o codegen/server.ptx.ll

$CLANG $XCUDA -std=c++14 --cuda-device-only codegen/client.cpp -S -o codegen/client.cuda.ptx.ll
$CLANG $XCUDA -std=c++14 --cuda-host-only codegen/client.cpp -S -o codegen/client.cuda.x64.ll

$CLANG $XHIP -std=c++14 --cuda-device-only codegen/client.cpp -S -o codegen/client.hip.gcn.ll
$CLANG $XHIP -std=c++14 --cuda-host-only codegen/client.cpp -S -o codegen/client.hip.x64.ll

exit

$CXX_X64 memory_host.cpp -c -o memory_host.x64.bc

$CXX_X64 -I$HSAINC memory_hsa.cpp -c -o memory_hsa.x64.bc

$CXX_X64 -I$HSAINC tests.cpp -c -o tests.x64.bc
$CXX_X64 -I$HSAINC x64_x64_stress.cpp -c -o x64_x64_stress.x64.bc


if (($have_nvptx)); then
# One step at a time
    $CLANG $XCUDA hello.cu -o hello -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart_static -ldl -lrt -pthread && ./hello

# hello.o is an executable elf, may be able to load it from cuda
$CLANG $XCUDA hello.cu --cuda-device-only -c -o hello.o  -I/usr/local/cuda/include

$CLANG hello.cpp --cuda-path=/usr/local/cuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcuda -o a.out memory_host.x64.bc && ./a.out hello.o

exit
fi


$CXX_GCN -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.code.bc
$CXXCL -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.kern.bc
$LINK x64_gcn_stress.gcn.code.bc x64_gcn_stress.gcn.kern.bc dispatch_id.ll -o x64_gcn_stress.gcn.bc
$CXX_GCN_LD x64_gcn_stress.gcn.bc -o x64_gcn_stress.gcn.so
$CXX_X64 -DDERIVE_VAL=$DERIVE -I$HSAINC x64_gcn_stress.cpp -c -o x64_gcn_stress.x64.bc

# $CXX_GCN -D__HAVE_ROCR_HEADERS=1 -I$HSAINC -I$DEVLIBINC persistent_kernel.cpp -c -o persistent_kernel.gcn.code.bc

$CXX_GCN -D__HAVE_ROCR_HEADERS=0 persistent_kernel.cpp -c -o persistent_kernel.gcn.code.bc

$CXXCL persistent_kernel.cpp -c -o persistent_kernel.gcn.kern.bc
$LINK persistent_kernel.gcn.code.bc persistent_kernel.gcn.kern.bc $OCKL_LIBS -o persistent_kernel.gcn.bc
$CXX_GCN_LD persistent_kernel.gcn.bc -o persistent_kernel.gcn.so
$CXX_X64 -I$HSAINC persistent_kernel.cpp -c -o persistent_kernel.x64.bc



# TODO: Sort out script to ignore this when there's no ptx device
if (($have_nvptx)); then
 $CXX_CUDA --cuda-device-only detail/platform.cu -c -emit-llvm -o detail/platform.ptx.bc
 $CXX_CUDA --cuda-host-only memory_cuda.cu  -c -emit-llvm -o memory_cuda.x64.bc
 $CXX_PTX x64_ptx_stress.cpp -c -o x64_ptx_stress.ptx.code.bc
 $CXX_X64 -I$HSAINC x64_ptx_stress.cpp -c -o x64_ptx_stress.x64.bc
else
 echo "Skipping ptx"
fi

$CXX_GCN hostcall.cpp -c -o hostcall.gcn.bc
$CXX_X64 -I$HSAINC hostcall.cpp -c -o hostcall.x64.bc

# Build the device code that uses said library
$CXX_X64 -I$HSAINC amdgcn_main.cpp -c -o amdgcn_main.x64.bc
$CXX_GCN amdgcn_main.cpp -c -o amdgcn_main.gcn.bc

# Build the device loader that assumes the device library is linked into the application
# TODO: Embed it directly in the loader by patching call to main, as the loader doesn't do it
$CXX_X64 -I$HSAINC amdgcn_loader.cpp -c -o amdgcn_loader.x64.bc

$CXX_X64_LD $LDFLAGS amdgcn_loader.x64.bc memory_host.x64.bc memory_hsa.x64.bc hostcall.x64.bc amdgcn_main.x64.bc -o ../amdgcn_loader.exe

# Build the device library that calls into main()
$CXXCL loader/amdgcn_loader_entry.cl -c -o loader/amdgcn_loader_entry.gcn.bc
$CXX_GCN loader/amdgcn_loader_cast.cpp -c -o loader/amdgcn_loader_cast.gcn.bc

$LINK loader/amdgcn_loader_entry.gcn.bc loader/amdgcn_loader_cast.gcn.bc | $OPT -O2 -o amdgcn_loader_device.gcn.bc



$LINK amdgcn_main.gcn.bc amdgcn_loader_device.gcn.bc  hostcall.gcn.bc  -o executable_device.gcn.bc

# Link the device image
$CXX_GCN_LD executable_device.gcn.bc -o a.out

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

$CXX_X64_LD tests.x64.bc x64_x64_stress.x64.bc states.x64.bc catch.o memory_host.x64.bc $LDFLAGS -o states.exe

$CXX_X64_LD x64_x64_stress.x64.bc catch.o memory_host.x64.bc $LDFLAGS -o x64_x64_stress.exe

$CXX_X64_LD x64_gcn_stress.x64.bc catch.o memory_host.x64.bc memory_hsa.x64.bc $LDFLAGS -o x64_gcn_stress.exe

$CXX_X64_LD tests.x64.bc catch.o memory_host.x64.bc  $LDFLAGS -o tests.exe


$CXX_X64_LD persistent_kernel.x64.bc catch.o memory_host.x64.bc memory_hsa.x64.bc $LDFLAGS -o persistent_kernel.exe

if (($have_amdgcn)); then
time ./persistent_kernel.exe
fi

time ./tests.exe
time ./x64_x64_stress.exe

if (($have_amdgcn)); then
echo "Call hostcall/loader executable"
time ./a.out ; echo $?
fi

if (($have_amdgcn)); then
echo "Call x64_gcn_stress: Derive $DERIVE"
time ./x64_gcn_stress.exe
fi

# time valgrind --leak-check=full --fair-sched=yes ./states.exe hazard

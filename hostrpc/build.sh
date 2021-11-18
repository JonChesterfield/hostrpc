#!/bin/bash
set -x
set -e
set -o pipefail

./clean.sh

DERIVE=${1:-1}

have_nvptx=0
if [ -e "/dev/nvidiactl" ]; then
    have_nvptx=1
fi

have_amdgcn=0
if [ -e "/dev/kfd" ]; then
    have_amdgcn=1
fi

echo "have_nvptx: $have_nvptx"
echo "have_amdgcn: $have_amdgcn"

RDIR=$HOME/llvm-install

if [[ -d "$RDIR" ]]
then
    echo "Using RDIR = $RDIR"
else
    RDIR=$(dirname $(dirname $(which clang)))
fi

if [[ -d "$RDIR" ]]
then
    echo "Using RDIR = $RDIR"
else
    echo "Failed to find a root toolchain directory"
    exit 1
fi

# set arch to reasonable defaults, override with those on the current system
# for the architecture that is available locally
PTXGFX=sm_50
GCNGFX=gfx906


if (($have_nvptx)); then
    if [[ -f "$RDIR/bin/llvm-omp-device-info" ]] ; then
        echo "Setting PTXGFX using llvm-omp-device-info"
        PTXGFX=`$RDIR/bin/llvm-omp-device-info | awk '/Compute Capabilities/{print "sm_"$3}'`
    else
        echo "No llvm-omp-device-info, disabling nvptx offloading"
        have_nvptx=0
    fi
fi

if (($have_amdgcn)); then
    if [[ -f "$RDIR/bin/amdgpu-arch" ]] ; then
        echo "Setting GCNGFX using amdgpu-arch"
        GCNGFX=`$RDIR/bin/amdgpu-arch | uniq`
    else
        echo "No amdgpu-arch, disabling amdgpu offloading"
        have_amdgcn=0
    fi
fi



PTXGFXNUM=$(echo "$PTXGFX" | sed 's/sm_//')
GCNGFXNUM=$(echo "$GCNGFX" | sed 's/gfx//')

PTXDEVICERTL=""
GCNDEVICERTL=""

if (($have_nvptx)); then
    PTXDEVICERTL="$RDIR/lib/libomptarget-nvptx64-$PTXGFX.bc"
fi

if (($have_amdgcn)); then
    GCNDEVICERTL="$RDIR/lib/libomptarget-amdgcn-$GCNGFX.bc"
fi

EXTRABC=

LOADPREFIX='LD_LIBRARY_PATH='$RDIR'/lib '

mkdir -p obj
mkdir -p lib
mkdir -p obj/tools
mkdir -p obj/unit_tests
mkdir -p thirdparty

if [[ -d "$HOME/relacy" ]]
then
    echo "Using existing relacy"
else
    echo "Cloning relacy"
    git clone https://github.com/dvyukov/relacy.git $HOME/relacy
fi

MSGPACKINCLUDE="thirdparty/msgpack"
if [[ -d $MSGPACKINCLUDE ]]
then
    echo "Using existing msgpack"
else
    echo "Cloning msgpack"
    git clone https://github.com/jonchesterfield/msgpack.git $MSGPACKINCLUDE
fi

EVILUNITINCLUDE="thirdparty/EvilUnit"
if [[ -d $EVILUNITINCLUDE ]]
then
    echo "Using existing evilunit"
else
    echo "Cloning evilunit"
    git clone https://github.com/jonchesterfield/evilunit.git $EVILUNITINCLUDE
fi


CXXVER='-std=c++17'
OPTLEVEL='-O2'

set +e
$RDIR/bin/clang++ -W -Wno-deprecated-copy -Wno-missing-field-initializers -Wno-inline-new-delete -Wno-unused-parameter $CXXVER -ffreestanding -I $HOME/relacy/ minimal.cpp -stdlib=libc++ -o minimal.out
set -e

echo "Using toolchain at $RDIR, GCNGFX=$GCNGFX, PTXGFX=$PTXGFX"


if (($have_nvptx)); then
    # Clang looks for this file, but cuda stopped shipping it
    if [ -e /usr/local/cuda/version.txt ]; then
        VER=`cat /usr/local/cuda/version.txt`
        echo "Found version: $VER"
    else
        VER=`/usr/local/cuda/bin/nvcc  --version | awk '/Cuda compilation/ {print $6}'`
        echo "Execute following to write to /usr/local:"
        echo 'echo "CUDA Version '$VER'" > /usr/local/cuda/version.txt'
        exit 1
    fi
fi

# A poorly named amd-stg-open, does not hang
# RDIR=$HOME/rocm-3.5-llvm-install

# Trunk, hangs
# RDIR=$HOME/llvm-install

HSAINC="$RDIR/include/hsa/"
DEVLIBINC="$HOME/aomp/rocm-device-libs/ockl/inc"
OCKL_DIR="$RDIR/amdgcn/bitcode"

OCKL_LIBS=""

HSALIBDIR="$RDIR/lib"
HSALIB="$HSALIBDIR/libhsa-runtime64.so" # $RDIR/lib/libomptarget.rtl.hsa.so"

# Shouldn't need these, but copying across from initial for reference 
# DLIBS="$RDIR/lib/libdevice/libhostcall-amdgcn-$GCNGFX.bc $RDIR/lib/ockl.amdgcn.bc $RDIR/lib/oclc_wavefrontsize64_on.amdgcn.bc $RDIR/lib/oclc_isa_version_$GCNGFXNUM.amdgcn.bc"

CLANG="$RDIR/bin/clang"
CLANGXX="$RDIR/bin/clang++"
LLC="$RDIR/bin/llc"
DIS="$RDIR/bin/llvm-dis"
LINK="$RDIR/bin/llvm-link"
OPT="$RDIR/bin/opt"

#CLANG="g++"
#LINK="ld -r"

CXX="$CLANGXX $CXXVER -Wall -Wextra "
LDFLAGS="-pthread $HSALIB -Wl,-rpath=$HSALIBDIR -lelf"

# Some languages need march and some need mcpu
# Failing to specify it means functions are emitted without target-cpu or similar
# attributes, which are subsequently miscompiled. No target-cpu implies some sort
# of gfx700, which raises errors from the compiler backend. Set march and mcpu for now.
AMDGPU="--target=amdgcn-amd-amdhsa -march=$GCNGFX -mcpu=$GCNGFX -Xclang -fconvergent-functions -nogpulib"

PTX_VER="-Xclang -target-feature -Xclang +ptx63"
NVGPU="--target=nvptx64-nvidia-cuda -march=$PTXGFX $PTX_VER -Xclang -fconvergent-functions"

CUDALINK="--cuda-path=/usr/local/cuda  -L/usr/local/cuda/lib64/ -lcuda -lcudart_static -ldl -lrt -pthread"

COMMONFLAGS="-Wall -Wextra -Werror=consumed -emit-llvm " # -DNDEBUG -Wno-type-limits "
# cuda/openmp pass the host O flag through to ptxas, which crashes on debug info if > 0
# there's a failure mode in trunk clang - 'remaining virtual register operands' - but it
# resists changing the pipeline to llvm-link + llc, will have to debug it later
X64FLAGS=" $OPTLEVEL -g -pthread " # nvptx can't handle debug info on x64 for O>0
GCNFLAGS=" $OPTLEVEL -ffreestanding -fno-exceptions $AMDGPU"
# atomic alignment objection seems reasonable - may want 32 wide atomics on nvptx
# clang/ptx back end is crashing in llvm::DwarfDebug::constructCallSiteEntryDIEs
NVPTXFLAGS=" $OPTLEVEL -ffreestanding -fno-exceptions -Wno-atomic-alignment -emit-llvm $NVGPU "

CXX_X64="$CLANGXX $CXXVER $COMMONFLAGS $X64FLAGS"
CXX_GCN="$CLANGXX $CXXVER $COMMONFLAGS $GCNFLAGS"

CXXCL="$CLANGXX -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 -D__OPENCL__ -D__OPENCL_C_VERSION__=200"
CXXCL_GCN="$CXXCL -emit-llvm -ffreestanding $AMDGPU"
CXXCL_PTX="$CXXCL -emit-llvm -ffreestanding $NVGPU"

CXX_PTX="$CLANGXX $NVPTXFLAGS"


XCUDA="-x cuda --cuda-gpu-arch=$PTXGFX --cuda-path=/usr/local/cuda"
XHIP="-x hip --cuda-gpu-arch=$GCNGFX -nogpulib -nogpuinc"
XOPENCL="-x cl -Xclang -cl-std=clc++ -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200  -Dcl_khr_fp64 -Dcl_khr_fp16   -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics" 

CXX_CUDA="$CLANGXX $OPTLEVEL $COMMONFLAGS $XCUDA -I/usr/local/cuda/include -nocudalib"

CXX_X64_LD="$CXX"
CXX_GCN_LD="$CXX $GCNFLAGS"

if [ ! -f obj/catch.o ]; then
    time $CXX -O3 thirdparty/catch.cpp -c -o obj/catch.o
fi

# Code running on the host can link in host, hsa or cuda support library.
# Fills in gaps in the cuda/hsa libs, implements allocators

if (($have_amdgcn)); then
  $CLANG $AMDGPU -ffreestanding -fno-exceptions $OPTLEVEL -emit-llvm -S conv.c -o obj/codegen_conv.ll
  $CLANG $AMDGPU -ffreestanding -fno-exceptions $OPTLEVEL -S conv.c -o obj/codegen_conv.s
fi

$CXX_X64 -I$HSAINC incprintf.cpp $OPTLEVEL -c -o obj/incprintf.x64.bc

# host support library
$CXX_X64 allocator_host_libc.cpp -c -o obj/allocator_host_libc.x64.bc
# wraps pthreads, cuda miscompiled <thread>
$CXX_X64 hostrpc_thread.cpp -c -o obj/hostrpc_thread.x64.bc 
$LINK obj/allocator_host_libc.x64.bc obj/hostrpc_thread.x64.bc -o obj/host_support.x64.bc

# hsa support library
if (($have_amdgcn)); then
$CXX_X64 $MSGPACKINCLUDE/msgpack.cpp -c -o obj/msgpack.x64.bc
$CXX_X64 find_metadata.cpp -c -o obj/find_metadata.x64.bc
$CXX_X64 -I$HSAINC allocator_hsa.cpp -c -o obj/allocator_hsa.x64.bc
$LINK  obj/host_support.x64.bc obj/msgpack.x64.bc obj/find_metadata.x64.bc obj/allocator_hsa.x64.bc  obj/incprintf.x64.bc -o obj/hsa_support.x64.bc

$CXX_X64 tools/dump_kernels.cpp -I../impl -I. -I$MSGPACKINCLUDE -c -o obj/tools/dump_kernels.x64.bc
$CXX_X64_LD obj/msgpack.x64.bc obj/tools/dump_kernels.x64.bc -lelf -o ../dump_kernels.exe

$CXX_X64 -I$HSAINC tools/query_system.cpp -c -o obj/tools/query_system.x64.bc
$CXX_X64_LD obj/tools/query_system.x64.bc obj/hsa_support.x64.bc $LDFLAGS -o ../query_system.exe
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

# currently standalone

if (($have_amdgcn)); then
    $CXX_GCN hostrpc_printf_enable_amdgpu.cpp $OPTLEVEL -c -o obj/hostrpc_printf_enable_amdgpu.gcn.bc
    $CXX_X64 hostrpc_printf_enable_amdgpu.cpp -I$HSAINC $OPTLEVEL -c -o obj/hostrpc_printf_enable_amdgpu.x64.bc
    $CXX_X64 hostrpc_printf_enable_host.cpp -I$HSAINC $OPTLEVEL -c -o obj/hostrpc_printf_enable_host.x64.bc
fi


if (($have_amdgcn)); then
    $CXX_GCN threads.cpp $OPTLEVEL -c -o threads.gcn.bc

    # $CXXCL_GCN pool_example_amdgpu.cpp $OPTLEVEL -c -o pool_example_amdgpu.ocl.gcn.bc

    $CLANGXX $XOPENCL pool_example_amdgpu.cpp $OPTLEVEL -emit-llvm -nogpulib -target amdgcn-amd-amdhsa -mcpu=$GCNGFX -c -o pool_example_amdgpu.ocl.gcn.bc
    $CXX_GCN pool_example_amdgpu.cpp $OPTLEVEL -c -o pool_example_amdgpu.cpp.gcn.bc

    $LINK threads.gcn.bc pool_example_amdgpu.ocl.gcn.bc pool_example_amdgpu.cpp.gcn.bc obj/hostrpc_printf_enable_amdgpu.gcn.bc $EXTRABC | $OPT -O2 -o obj/merged_pool_example_amdgpu.gcn.bc 
    $DIS obj/merged_pool_example_amdgpu.gcn.bc

    $CXX_GCN_LD obj/merged_pool_example_amdgpu.gcn.bc -o pool_example_amdgpu.gcn.so

    $CXX_X64 threads.cpp $OPTLEVEL -c -o threads.x64.bc
    $CXX_X64 pool_example_amdgpu.cpp -I$HSAINC $OPTLEVEL -c -o obj/pool_example_amdgpu.x64.bc
    
    $CXX_X64_LD threads.x64.bc obj/hsa_support.x64.bc obj/catch.o $LDFLAGS -o threads.x64.exe

    $CXX_X64_LD obj/pool_example_amdgpu.x64.bc obj/hostrpc_printf_enable_amdgpu.x64.bc obj/hsa_support.x64.bc $LDFLAGS -o pool_example_amdgpu.x64.exe
fi


if (($have_amdgcn)); then
    # Currently assumes hsa, but shouldn't
  $CXX_X64 -I$HSAINC pool_example_host.cpp $OPTLEVEL -c -o obj/pool_example_host.x64.bc
  $CXX_X64_LD obj/pool_example_host.x64.bc $LDFLAGS -o pool_example_host.x64.exe
fi

if (($have_amdgcn)); then
$CXX_GCN x64_gcn_debug.cpp -c -o obj/x64_gcn_debug.gcn.code.bc
$CXXCL_GCN x64_gcn_debug.cpp -c -o obj/x64_gcn_debug.gcn.kern.bc
$LINK obj/x64_gcn_debug.gcn.code.bc obj/x64_gcn_debug.gcn.kern.bc obj/hostrpc_printf_enable_amdgpu.gcn.bc -o obj/x64_gcn_debug.gcn.bc
$CXX_GCN_LD obj/x64_gcn_debug.gcn.bc -o x64_gcn_debug.gcn.so

$CXX_X64 -I$HSAINC x64_gcn_debug.cpp -c -o obj/x64_gcn_debug.x64.bc

$CXX obj/x64_gcn_debug.x64.bc obj/hsa_support.x64.bc obj/hostrpc_printf_enable_amdgpu.x64.bc $LDFLAGS -o x64_gcn_debug.exe

# Ideally would have a test case that links the x64 and gcn bitcode, but can't work out
# how to do that. I think mlink-builtin-bitcode used to be architecture aware, but
# currently it cheerfully links gcn and x64 bitcode together to make a thing that
# crashes the bitcode reader
# therefore, openmp_hostcall is presently built into the openmp deviceRTL and amdgpu plugin
# by #including it from this repo. Stand alone tests can then (optimistically) assume that
# the allocator is available, and the hostcall machinery spun up by the plugin
# The drawback is needing to rebuild llvm when the source changes.
$CXX_X64 -I$HSAINC openmp_hostcall.cpp -c -o obj/openmp_hostcall.x64.bc
$CXX_GCN openmp_hostcall.cpp -c -o obj/openmp_hostcall.gcn.bc
fi

$CXX_X64 syscall.cpp -c -o obj/syscall.x64.bc 

# amdgcn loader links these, but shouldn't. need to refactor.
if (($have_amdgcn)); then
    $CXX_GCN hostcall.cpp -c -o hostcall.gcn.bc
    $CXX_X64 -I$HSAINC hostcall.cpp -c -o hostcall.x64.bc
    # Build the device code that uses said library
    $CXX_X64 -I$HSAINC amdgcn_main.cpp -c -o amdgcn_main.x64.bc
    $CXX_GCN amdgcn_main.cpp -c -o amdgcn_main.gcn.bc
fi

# build loaders - run int main() {} on the gpu

# Register amdhsa elf magic with kernel
# One off
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::$HOME/hostrpc/amdgcn_loader.exe:' > register
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x02\x00\x00\x00\x00\x00\x00\x00::$HOME/hostrpc/amdgcn_loader.exe:' > register

# Persistent
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::$HOME/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x02\x00\x00\x00\x00\x00\x00\x00::$HOME/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf

if (($have_amdgcn)); then
  $CXXCL_GCN tools/loader/amdgcn_loader_entry.cl -c -o tools/loader/amdgcn_loader_entry.gcn.bc
  $CXX_GCN tools/loader/opencl_loader_cast.cpp -c -o tools/loader/opencl_loader_cast.gcn.bc
  $LINK tools/loader/amdgcn_loader_entry.gcn.bc tools/loader/opencl_loader_cast.gcn.bc | $OPT -O2 -o amdgcn_loader_device.gcn.bc

  $CXX_X64 -I$HSAINC -I. tools/amdgcn_loader.cpp -c -o tools/loader/amdgcn_loader.x64.bc
  $CXX_X64_LD $LDFLAGS tools/loader/amdgcn_loader.x64.bc obj/hsa_support.x64.bc obj/hostrpc_printf_enable_amdgpu.x64.bc hostcall.x64.bc amdgcn_main.x64.bc -o ../amdgcn_loader.exe
fi

if (($have_nvptx)); then
 # presently using the cuda entry point but may want the opencl one later
 $CXX_CUDA $CXXVER --cuda-device-only tools/loader/nvptx_loader_entry.cu -c -emit-llvm -o tools/loader/nvptx_loader_entry.cu.ptx.bc   
 $CXXCL_PTX tools/loader/nvptx_loader_entry.cl -c -o tools/loader/nvptx_loader_entry.cl.ptx.bc
 $CXX_PTX tools/loader/opencl_loader_cast.cpp -c -o tools/loader/opencl_loader_cast.ptx.bc

 $CXX_X64 -I/usr/local/cuda/include -I. tools/nvptx_loader.cpp -c -o tools/loader/nvptx_loader.x64.bc
 $CXX_X64_LD tools/loader/nvptx_loader.x64.bc obj/cuda_support.x64.bc $CUDALINK -o ../nvptx_loader.exe
fi

if (($have_amdgcn)); then
  $CLANG -std=c11 $COMMONFLAGS $GCNFLAGS unit_tests/test_example.c -c -o obj/unit_tests/test_example.gcn.bc
  $LINK obj/unit_tests/test_example.gcn.bc obj/hostrpc_printf_enable_amdgpu.gcn.bc amdgcn_loader_device.gcn.bc -o unit_tests/test_example.gcn.bc
  $CXX_GCN_LD unit_tests/test_example.gcn.bc -o unit_tests/test_example.gcn

  $CLANG -std=c11 -I$HSAINC $COMMONFLAGS $X64FLAGS printf_test.c -c -o obj/printf_test.x64.bc

  $CXX_X64_LD obj/host_support.x64.bc obj/hostrpc_printf_enable_host.x64.bc obj/printf_test.x64.bc obj/incprintf.x64.bc -o printf_test.x64.exe -pthread

  $CLANG -std=c11 $COMMONFLAGS $GCNFLAGS printf_test.c -c -o obj/printf_test.gcn.bc
  $LINK obj/printf_test.gcn.bc obj/hostrpc_printf_enable_amdgpu.gcn.bc amdgcn_loader_device.gcn.bc -o printf_test.gcn.bc
  $CXX_GCN_LD printf_test.gcn.bc -o printf_test.gcn
fi


$CXX_X64 unit_tests/common.cpp -c -o obj/unit_tests/common.x64.bc
$CXX_X64_LD obj/unit_tests/common.x64.bc -o unit_tests/common.x64.exe
./unit_tests/common.x64.exe

$CXX_X64 unit_tests/typed_port.cpp -c -o obj/unit_tests/typed_port.x64.bc
$CXX_X64_LD obj/unit_tests/typed_port.x64.bc -o unit_tests/typed_port.x64.exe
./unit_tests/typed_port.x64.exe


if (($have_amdgcn)); then
$CXX_GCN unit_tests/common.cpp -c -o obj/unit_tests/common.gcn.bc
$LINK obj/unit_tests/common.gcn.bc obj/hostrpc_printf_enable_amdgpu.gcn.bc amdgcn_loader_device.gcn.bc hostcall.gcn.bc -o obj/unit_tests/common.gcn.linked.bc

$CXX_GCN_LD obj/unit_tests/common.gcn.linked.bc -o unit_tests/common.gcn.exe
../amdgcn_loader.exe ./unit_tests/common.gcn.exe
fi

#if (($have_amdgcn)); then
#    $CXX_GCN devicertl_pteam_mem_barrier.cpp -c -o obj/devicertl_pteam_mem_barrier.gcn.bc
#    # todo: refer to lib from RDIR, once that lib has the function non-static    
#    $LINK obj/devicertl_pteam_mem_barrier.gcn.bc obj/hostrpc_printf_enable_amdgpu.gcn.bc amdgcn_loader_device.gcn.bc -o devicertl_pteam_mem_barrier.gcn.bc $GCNDEVICERTL
#    $CXX_GCN_LD devicertl_pteam_mem_barrier.gcn.bc -o devicertl_pteam_mem_barrier.gcn
#    set +e
#    echo "This is failing at present, HSA doesn't think the binary is valid"
#    # ./devicertl_pteam_mem_barrier.gcn
#    set -e
#fi

$CXX_X64 prototype/states.cpp -c -o prototype/states.x64.bc

if false; then
    $CXX_GCN run_on_hsa_example.cpp -c -o obj/run_on_hsa_example.cxx.gcn.bc
    $CXXCL_GCN run_on_hsa_example.cpp -c -o obj/run_on_hsa_example.ocl.gcn.bc
    $LINK obj/run_on_hsa_example.cxx.gcn.bc obj/run_on_hsa_example.ocl.gcn.bc -o obj/run_on_hsa_example.gcn.bc

    $CXX_GCN_LD obj/run_on_hsa_example.gcn.bc -o lib/run_on_hsa_example.gcn.so

    $CXX_X64 -I$HSAINC run_on_hsa_example.cpp -c -o obj/run_on_hsa_example.cxx.x64.bc
    $CXX_X64 -I$HSAINC run_on_hsa.cpp -c -o obj/run_on_hsa.x64.bc

    $CXX $LDFLAGS obj/run_on_hsa_example.cxx.x64.bc obj/run_on_hsa.x64.bc obj/hsa_support.x64.bc -o run_on_hsa.exe

    ./run_on_hsa.exe
fi

if true; then
# Sanity checks that the client and server compile successfully
# and provide an example of the generated IR
$CXX_X64 $CXXVER -DNDEBUG codegen/client.cpp -S -o codegen/client.x64.ll
$CXX_X64 $CXXVER -DNDEBUG codegen/server.cpp -S -o codegen/server.x64.ll
$CXX_GCN $CXXVER -DNDEBUG codegen/client.cpp -S -o codegen/client.gcn.ll
$CXX_GCN $CXXVER -DNDEBUG codegen/server.cpp -S -o codegen/server.gcn.ll
$CXX_PTX $CXXVER -DNDEBUG codegen/client.cpp -S -o codegen/client.ptx.ll
$CXX_PTX $CXXVER -DNDEBUG codegen/server.cpp -S -o codegen/server.ptx.ll

$CXX_X64 $CXXVER codegen/foo_cxx.cpp -S -o codegen/foo_cxx.x64.ll
$CXX_GCN $CXXVER codegen/foo_cxx.cpp -S -o codegen/foo_cxx.gcn.ll
$CXX_PTX $CXXVER codegen/foo_cxx.cpp -S -o codegen/foo_cxx.ptx.ll

$CLANGXX $XCUDA $CXXVER --cuda-device-only -nocudainc -nocudalib codegen/foo.cu -emit-llvm -S -o codegen/foo.cuda.ptx.ll

$CLANGXX $XCUDA $CXXVER --cuda-host-only -nocudainc -nocudalib codegen/foo.cu -emit-llvm -S -o codegen/foo.cuda.x64.ll

cd codegen
$CLANGXX $XCUDA $CXXVER -nocudainc -nocudalib foo.cu -emit-llvm -S
mv foo.ll foo.cuda.both_x64.ll
mv foo-cuda-nvptx64-nvidia-cuda-*.ll foo.cuda.both_ptx.ll
cd -

CODEGENOPTLEVEL='-O2'

# aomp has broken cuda-device-only
$CLANGXX -x hip --cuda-gpu-arch=$GCNGFX -nogpulib -nogpuinc $CXXVER $CODEGENOPTLEVEL --cuda-device-only codegen/foo.cu -emit-llvm -S -o codegen/foo.hip.gcn.ll
$CLANGXX -x hip --cuda-gpu-arch=$GCNGFX -nogpulib -nogpuinc $CXXVER $CODEGENOPTLEVEL --cuda-host-only codegen/foo.cu -emit-llvm -S -o codegen/foo.hip.x64.ll

# hip doesn't understand -emit-llvm (or -S, or -c) when trying to do host and device together
# so can't test that here

# This ignores -S for some reason
$CLANGXX $CODEGENOPTLEVEL -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GCNGFX codegen/foo.omp.cpp -c -emit-llvm --cuda-device-only -o codegen/foo.omp.gcn.bc && $DIS codegen/foo.omp.gcn.bc && rm codegen/foo.omp.gcn.bc

# ignores host-only, so the IR has a binary gfx pasted at the top
$CLANGXX $CODEGENOPTLEVEL  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GCNGFX  codegen/foo.omp.cpp -S -emit-llvm --cuda-host-only -o codegen/foo.omp.gcn-x64.ll


$CLANGXX $CODEGENOPTLEVEL  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=$PTXGFX  codegen/foo.omp.cpp -c -emit-llvm -S --cuda-device-only -o codegen/foo.omp.ptx.ll

$CLANGXX $CODEGENOPTLEVEL  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=$PTXGFX  codegen/foo.omp.cpp -c -emit-llvm -S --cuda-host-only -o codegen/foo.omp.ptx-x64.ll

# OpenCL compilation model is essentially that of c++
$CLANGXX $XOPENCL -S -emit-llvm codegen/foo_cxx.cpp -S -o codegen/foo.cl.x64.ll

$CLANGXX $XOPENCL -S -nogpulib -emit-llvm -target amdgcn-amd-amdhsa -mcpu=$GCNGFX codegen/foo_cxx.cpp -S -o codegen/foo.cl.gcn.ll

# recognises mcpu but warns that it is unused
$CLANGXX $XOPENCL -S -nogpulib -emit-llvm -target nvptx64-nvidia-cuda codegen/foo_cxx.cpp -S -o codegen/foo.cl.ptx.ll

$CLANGXX $XCUDA $PTX_VER $CXXVER --cuda-device-only -nocudainc -nocudalib codegen/client.cpp -emit-llvm -S -o codegen/client.cuda.ptx.ll
$CLANGXX $XCUDA $CXXVER --cuda-host-only -nocudainc -nocudalib codegen/client.cpp -emit-llvm -S -o codegen/client.cuda.x64.ll


# Fails to annotate CFG at O0
$CLANGXX $XHIP $CXXVER $CODEGENOPTLEVEL --cuda-device-only codegen/client.cpp -S -o codegen/client.hip.gcn.ll
$CLANGXX $XHIP $CXXVER $CODEGENOPTLEVEL --cuda-host-only codegen/client.cpp -S -o codegen/client.hip.x64.ll
$CLANGXX $XHIP $CXXVER $CODEGENOPTLEVEL --cuda-device-only codegen/server.cpp -S -o codegen/server.hip.gcn.ll
$CLANGXX $XHIP $CXXVER $CODEGENOPTLEVEL --cuda-host-only codegen/server.cpp -S -o codegen/server.hip.x64.ll

# Build as opencl/c++ too
$CLANGXX $XOPENCL -S -emit-llvm codegen/client.cpp -S -o codegen/client.ocl.x64.ll
$CLANGXX $XOPENCL -S -emit-llvm codegen/server.cpp -S -o codegen/server.ocl.x64.ll
fi

$CXX_X64 -I$HSAINC tests.cpp -c -o tests.x64.bc
$CXX_X64 -I$HSAINC x64_x64_stress.cpp -c -o x64_x64_stress.x64.bc

if (($have_amdgcn)); then

$CXX_GCN -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.code.bc
$CXXCL_GCN -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.kern.bc
$LINK x64_gcn_stress.gcn.code.bc x64_gcn_stress.gcn.kern.bc -o x64_gcn_stress.gcn.bc
$CXX_GCN_LD x64_gcn_stress.gcn.bc -o x64_gcn_stress.gcn.so
$CXX_X64 -DDERIVE_VAL=$DERIVE -I$HSAINC x64_gcn_stress.cpp -c -o x64_gcn_stress.x64.bc

# $CXX_GCN -D__HAVE_ROCR_HEADERS=1 -I$HSAINC -I$DEVLIBINC persistent_kernel.cpp -c -o persistent_kernel.gcn.code.bc

$CXX_GCN -D__HAVE_ROCR_HEADERS=0 persistent_kernel.cpp -c -o persistent_kernel.gcn.code.bc

$CXXCL_GCN persistent_kernel.cpp -c -o persistent_kernel.gcn.kern.bc
$LINK persistent_kernel.gcn.code.bc persistent_kernel.gcn.kern.bc $OCKL_LIBS -o persistent_kernel.gcn.bc

set +e
$CXX_GCN_LD persistent_kernel.gcn.bc -o persistent_kernel.gcn.so
$CXX_X64 -I$HSAINC persistent_kernel.cpp -c -o persistent_kernel.x64.bc
set -e
fi

$CXX_CUDA $CXXVER --cuda-device-only -nogpuinc -nobuiltininc $PTX_VER platform/nvptx.cu -c -emit-llvm -o obj/platform.ptx.bc

if (($have_amdgcn)); then
    # Tries to treat foo.so as a hip input file. Somewhat surprised, but might be right.
    # The clang driver can't handle some hip input + some bitcode input, but does have the
    # internal hook -mlink-builtin-bitcode that can be used to the same end effect
    $LINK obj/hsa_support.x64.bc obj/syscall.x64.bc -o obj/demo.hip.link.x64.bc

    # hip presently fails to build, so the library will be missing
    set +e
    $CLANGXX -I$HSAINC -std=c++11 -x hip demo.hip -o demo --offload-arch=$GCNGFX -Xclang -mlink-builtin-bitcode -Xclang obj/demo.hip.link.x64.bc -L$HOME/rocm/aomp/hip -L$HOME/rocm/aomp/lib -lamdhip64 -L$HSALIBDIR -lhsa-runtime64 -Wl,-rpath=$HSALIBDIR -pthread -ldl
    set -e
    # ./demo hsa runtime presently segfaults in hip's library
fi

$CXX_PTX $CXXVER nvptx_main.cpp -ffreestanding -c -o nvptx_main.ptx.bc

if (($have_nvptx)); then
# One step at a time
    $CLANGXX $XCUDA hello.cu -o hello -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcuda -lcudart_static -ldl -lrt -pthread && ./hello

# hello.o is an executable elf, may be able to load it from cuda
$CLANGXX $XCUDA $CXXVER hello.cu --cuda-device-only $PTX_VER -c -o hello.o  -I/usr/local/cuda/include


# ./../nvptx_loader.exe hello.o

fi

$CLANGXX $CXXVER -Wall -Wextra -O0 -g test_storage.cpp obj/openmp_support.x64.bc obj/host_support.x64.bc $RDIR/lib/libomptarget.so -o test_storage.exe -pthread -ldl -Wl,-rpath=$RDIR/lib && valgrind ./test_storage.exe

if (($have_amdgcn)); then
    $LINK obj/openmp_support.x64.bc obj/hsa_support.x64.bc obj/syscall.x64.bc -o obj/demo_bitcode_gcn.omp.bc

    # openmp was taking an excessive amount of time to compile
    # this now fails to compile (fairly quickly), looks like mlink-builtin-bitcode is mixing
    # the two ISAs in a single module
    set +e
    $CLANGXX $CXXVER -I$HSAINC $OPTLEVEL -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=$GCNGFX -mcpu=$GCNGFX -DDEMO_AMDGCN=1 demo_openmp.cpp \
         -Xclang -mlink-builtin-bitcode -Xclang obj/demo_bitcode_gcn.omp.bc -o demo_openmp_gcn -pthread -ldl $HSALIB -Wl,-rpath=$HSALIBDIR && ./demo_openmp_gcn
    set -e
fi

if (($have_nvptx)); then
    $LINK obj/openmp_support.x64.bc obj/cuda_support.x64.bc obj/syscall.x64.bc -o demo_bitcode_ptx.omp.bc

    $CLANGXX $CXXVER -I$HSAINC -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=$PTXGFX -I/usr/local/cuda/include -DDEMO_NVPTX=1 demo_openmp.cpp \
             -Xclang -mlink-builtin-bitcode -Xclang demo_bitcode_ptx.omp.bc -Xclang -mlink-builtin-bitcode -Xclang obj/platform.ptx.bc -o demo_openmp_ptx $CUDALINK -Wl,-rpath=$RDIR/lib
fi


# Build the device library that calls into main()

if (($have_amdgcn)); then
$LINK amdgcn_main.gcn.bc amdgcn_loader_device.gcn.bc hostcall.gcn.bc obj/hostrpc_printf_enable_amdgpu.gcn.bc -o executable_device.gcn.bc

# Link the device image
$CXX_GCN_LD executable_device.gcn.bc -o a.gcn.out
fi

if (($have_nvptx)); then

$LINK nvptx_main.ptx.bc tools/loader/nvptx_loader_entry.cu.ptx.bc obj/platform.ptx.bc -o executable_device.ptx.bc

$LINK nvptx_main.ptx.bc tools/loader/nvptx_loader_entry.cu.ptx.bc obj/platform.ptx.bc -o executable_device.ptx.bc


$CLANGXX --target=nvptx64-nvidia-cuda -march=$PTXGFX $PTX_VER executable_device.ptx.bc -S -o executable_device.ptx.s

/usr/local/cuda/bin/ptxas -m64 -O0 --gpu-name $PTXGFX executable_device.ptx.s -o a.ptx.out
fi


# llc seems to need to be told what architecture it's disassembling
# $LLC --mcpu=$GCNGFX $ll


# $CXX_X64_LD tests.x64.bc prototype/states.x64.bc obj/catch.o obj/allocator_host_libc.x64.bc $LDFLAGS -o prototype/states.exe

$CXX_X64_LD prototype/states.x64.bc obj/catch.o $LDFLAGS -o prototype/states.exe

$CXX_X64_LD x64_x64_stress.x64.bc obj/host_support.x64.bc obj/catch.o $LDFLAGS -o x64_x64_stress.exe

if (($have_amdgcn)); then
$CXX_X64_LD x64_gcn_stress.x64.bc obj/hsa_support.x64.bc obj/catch.o $LDFLAGS -o x64_gcn_stress.exe
fi

$CXX_X64_LD tests.x64.bc obj/host_support.x64.bc obj/catch.o $LDFLAGS -o tests.exe


# clang trunk is crashing on this at present
if (($have_amdgcn)); then
set +e
$CXX_X64_LD persistent_kernel.x64.bc obj/catch.o obj/hsa_support.x64.bc $LDFLAGS -o persistent_kernel.exe
set -e
fi

if (($have_amdgcn)); then
./pool_example_amdgpu.x64.exe
fi

time valgrind --leak-check=full --fair-sched=yes ./prototype/states.exe

if (($have_nvptx)); then
./../nvptx_loader.exe a.ptx.out
fi

# ./printf_test.x64.exe
# ./printf_test.gcn

set +e # Keep running tests after one fails

# ./threads.x64.exe

time ./x64_x64_stress.exe

if (($have_amdgcn)); then
./unit_tests/test_example.gcn
fi

# Not totally reliable, sometimes raises memory access errors
# Hanging on gfx10 at present
#if (($have_amdgcn)); then  
#    ./pool_example_amdgpu.x64.exe
#fi
if (($have_amdgcn)); then
    # slightly spuriously depends on hsa
./pool_example_host.x64.exe
fi

#if (($have_amdgcn)); then
#$RDIR/bin/amdgpu-arch
#./pool_example_amdgpu.x64.exe #works
#./unit_tests/test_example.gcn
#./pool_example_amdgpu.x64.exe #hangs then persistent memory fault
#fi

if (($have_nvptx)); then
bash -c "$LOADPREFIX ./demo_openmp_ptx"
fi

time ./tests.exe

if (($have_amdgcn)); then
echo "Call hostcall/loader executable"
time ./a.gcn.out ; echo $?
fi

if (($have_amdgcn)); then
echo "Call x64_gcn_stress: Derive $DERIVE"
time ./x64_gcn_stress.exe
fi


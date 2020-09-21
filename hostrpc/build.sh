#!/bin/bash
set -x

./clean.sh

DERIVE=${1:-4}
# Aomp
RDIR=$HOME/rocm/aomp

# Needs to resolve to gfx906, gfx1010 or similar
GFX=`$RDIR/bin/mygpu`

# A poorly named amd-stg-open, does not hang
# RDIR=$HOME/rocm-3.5-llvm-install

# Trunk, hangs
# RDIR=$HOME/llvm-install

HSAINC="$HOME/aomp/rocr-runtime/src/inc/"


HSALIBDIR="$HOME/rocm/aomp/hsa/lib/"
HSALIB="$HSALIBDIR/libhsa-runtime64.so" # $RDIR/lib/libomptarget.rtl.hsa.so"

# Shouldn't need these, but copying across from initial for reference 
# DLIBS="$RDIR/lib/libdevice/libhostcall-amdgcn-$GFX.bc $RDIR/lib/ockl.amdgcn.bc $RDIR/lib/oclc_wavefrontsize64_on.amdgcn.bc $RDIR/lib/oclc_isa_version_906.amdgcn.bc"

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
NVGPU="--target=nvptx64-nvidia-cuda -march=sm_50 --cuda-gpu-arch=sm_50 -D__CUDACC__"

COMMONFLAGS="-Wall -Wextra -emit-llvm  -DNDEBUG -Wno-type-limits "
X64FLAGS=" -O2 -pthread -g"
GCNFLAGS=" -O2 -ffreestanding -fno-exceptions $AMDGPU"
# atomic alignment objection seems reasonable - may want 32 wide atomics on nvptx
NVPTXFLAGS="-g -O2 -emit-llvm -ffreestanding -fno-exceptions -Wno-atomic-alignment $NVGPU"

CXX_X64="$CLANG -std=c++14 $COMMONFLAGS $X64FLAGS"
CXX_GCN="$CLANG -std=c++14 $COMMONFLAGS $GCNFLAGS"

CXX_X64_LD="$CXX"
CXX_GCN_LD="$CXX $GCNFLAGS"

CXXCL="$CLANG -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 -emit-llvm -D__OPENCL__ $AMDGPU"


# msgpack, assumed to be checked out ../ from here
$CXX_X64 ../impl/msgpack.cpp -c -o msgpack.bc
$CXX_X64 find_metadata.cpp -c -o find_metadata.bc
$LINK msgpack.bc find_metadata.bc -o hsa_support.bc

# time $CXX -O3 catch.cpp -c -o catch.o

$CXX_X64 states.cpp -c -o states.x64.bc

# TODO: Drop hsainc from x64 code

$CXX_X64 -I$HSAINC codegen/client.cpp -c -o codegen/client.x64.bc
$CXX_X64 -I$HSAINC codegen/server.cpp -c -o codegen/server.x64.bc
$CXX_GCN codegen/client.cpp -c -o codegen/client.gcn.bc
$CXX_GCN codegen/server.cpp -c -o codegen/server.gcn.bc


$CXX_X64 -I$HSAINC memory.cpp -c -o memory.x64.bc
$CXX_X64 -I$HSAINC x64_host_x64_client.cpp -c -o x64_host_x64_client.x64.bc
$CXX_X64 -I$HSAINC tests.cpp -c -o tests.x64.bc
$CXX_X64 -I$HSAINC x64_x64_stress.cpp -c -o x64_x64_stress.x64.bc


$CXX_GCN x64_host_gcn_client.cpp -c -o x64_host_gcn_client.gcn.bc
$CXX_X64 -I$HSAINC x64_host_gcn_client.cpp -c -o x64_host_gcn_client.x64.bc

$CXX_GCN -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.code.bc
$CXXCL -DDERIVE_VAL=$DERIVE x64_gcn_stress.cpp -c -o x64_gcn_stress.gcn.kern.bc
$LINK x64_gcn_stress.gcn.code.bc x64_gcn_stress.gcn.kern.bc -o x64_gcn_stress.gcn.bc
$CXX_GCN_LD x64_gcn_stress.gcn.bc x64_host_gcn_client.gcn.bc -o x64_gcn_stress.gcn.so
$CXX_X64 -DDERIVE_VAL=$DERIVE -I$HSAINC x64_gcn_stress.cpp -c -o x64_gcn_stress.x64.bc


$CXX_GCN gcn_host_x64_client.cpp -c -o gcn_host_x64_client.gcn.bc
$CXX_X64 -I$HSAINC gcn_host_x64_client.cpp -c -o gcn_host_x64_client.x64.bc


$CXX_GCN persistent_kernel.cpp -c -o persistent_kernel.gcn.code.bc
$CXXCL persistent_kernel.cpp -c -o persistent_kernel.gcn.kern.bc
$LINK persistent_kernel.gcn.code.bc persistent_kernel.gcn.kern.bc -o persistent_kernel.gcn.bc
$CXX_GCN_LD persistent_kernel.gcn.bc x64_host_gcn_client.gcn.bc -o persistent_kernel.gcn.so
$CXX_X64 -I$HSAINC persistent_kernel.cpp -c -o persistent_kernel.x64.bc

# $CXX $NVPTXFLAGS client.cpp -c -o client.ptx.bc

$CXX_GCN hostcall_interface.cpp -c -o hostcall_interface.gcn.bc
$CXX_X64 -I$HSAINC hostcall_interface.cpp -c -o hostcall_interface.x64.bc

$CXX_GCN hostcall.cpp -c -o hostcall.gcn.bc
$CXX_X64 -I$HSAINC hostcall.cpp -c -o hostcall.x64.bc


# Build the device code that uses said library
$CXX_X64 -I$HSAINC amdgcn_main.cpp -c -o amdgcn_main.x64.bc
$CXX_GCN amdgcn_main.cpp -c -o amdgcn_main.gcn.bc


# Build the device loader that assumes the device library is linked into the application
# TODO: Embed it directly in the loader by patching call to main, as the loader doesn't do it
$CXX_X64 -I$HSAINC amdgcn_loader.cpp -c -o amdgcn_loader.x64.bc

$CXX_X64_LD $LDFLAGS amdgcn_loader.x64.bc memory.x64.bc hostcall_interface.x64.bc hostcall.x64.bc amdgcn_main.x64.bc -o ../amdgcn_loader.exe

# Build the device library that calls into main()
$CXXCL loader/amdgcn_loader_entry.cl -c -o loader/amdgcn_loader_entry.gcn.bc
$CXX_GCN loader/amdgcn_loader_cast.cpp -c -o loader/amdgcn_loader_cast.gcn.bc

$LINK loader/amdgcn_loader_entry.gcn.bc loader/amdgcn_loader_cast.gcn.bc | $OPT -O2 -o amdgcn_loader_device.gcn.bc



$LINK amdgcn_main.gcn.bc amdgcn_loader_device.gcn.bc hostcall_interface.gcn.bc hostcall.gcn.bc  -o executable_device.gcn.bc

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

$CXX_X64_LD tests.x64.bc x64_x64_stress.x64.bc states.x64.bc catch.o memory.x64.bc x64_host_x64_client.x64.bc $LDFLAGS -o states.exe

$CXX_X64_LD x64_x64_stress.x64.bc catch.o memory.x64.bc x64_host_x64_client.x64.bc $LDFLAGS -o x64_x64_stress.exe

$CXX_X64_LD x64_gcn_stress.x64.bc catch.o memory.x64.bc x64_host_gcn_client.x64.bc $LDFLAGS -o x64_gcn_stress.exe

$CXX_X64_LD tests.x64.bc catch.o x64_host_x64_client.x64.bc memory.x64.bc  $LDFLAGS -o tests.exe


$CXX_X64_LD persistent_kernel.x64.bc catch.o memory.x64.bc gcn_host_x64_client.x64.bc $LDFLAGS -o persistent_kernel.exe

time ./persistent_kernel.exe

time ./tests.exe
time ./x64_x64_stress.exe

echo "Call hostcall/loader executable"
time ./a.out ; echo $?

echo "Call x64_gcn_stress: Derive $DERIVE"
time ./x64_gcn_stress.exe

# time valgrind --leak-check=full --fair-sched=yes ./states.exe hazard
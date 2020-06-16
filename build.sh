#!/bin/bash
set -x

RDIR=$HOME/rocm/aomp

HSAINC="$(readlink -f ~/aomp/rocr-runtime/src/inc/)"
HSALIBDIR="$(readlink -f ~/rocm/aomp/hsa/lib/)"
HSALIB="$HSALIBDIR/libhsa-runtime64.so" # $RDIR/lib/libomptarget.rtl.hsa.so"

# Shouldn't need these, but copying across from initial for reference 
# DLIBS="$RDIR/lib/libdevice/libhostcall-amdgcn-gfx906.bc $RDIR/lib/ockl.amdgcn.bc $RDIR/lib/oclc_wavefrontsize64_on.amdgcn.bc $RDIR/lib/oclc_isa_version_906.amdgcn.bc"

CC="clang -std=c99 -Wall -Wextra"
CXX="clang++ -std=c++11 -Wall -Wextra " # -DNDEBUG"
LDFLAGS="-pthread $HSALIB -Wl,-rpath=$HSALIBDIR"
LLC="llc"
LINK="llvm-link"
OPT="opt"

AMDGPU="--target=amdgcn-amd-amdhsa -march=gfx906 -mcpu=gfx906"

# Not sure why CUDACC isn't being set by clang here, probably a bad sign
NVGPU="--target=nvptx64-nvidia-cuda -march=sm_50 --cuda-gpu-arch=sm_50 -D__CUDACC__"

X64FLAGS="-g -O2 -emit-llvm -pthread"
AMDGCNFLAGS="-g -O2 -emit-llvm -ffreestanding -fno-exceptions $AMDGPU"
# atomic alignment objection seems reasonable - may want 32 wide atomics on nvptx
NVPTXFLAGS="-g -O2 -emit-llvm -ffreestanding -fno-exceptions -Wno-atomic-alignment $NVGPU"

CXXCL="clang++ -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 $AMDGPU"

# time $CXX -O3 catch.cpp -c -o catch.o
rm -rf *.s *.ll *.bc *.exe *device.o

$CXX $X64FLAGS states.cpp -c -o states.x64.bc

# TODO: Drop hsainc from x64 code

$CXX $X64FLAGS -I$HSAINC client.cpp -c -o client.x64.bc
$CXX $X64FLAGS -I$HSAINC server.cpp -c -o server.x64.bc
$CXX $X64FLAGS -I$HSAINC memory.cpp -c -o memory.x64.bc
$CXX $X64FLAGS -I$HSAINC x64_host_x64_client.cpp -c -o x64_host_x64_client.x64.bc
$CXX $X64FLAGS -I$HSAINC tests.cpp -c -o tests.x64.bc
$CXX $X64FLAGS -I$HSAINC x64_hazard_test.cpp -c -o x64_hazard_test.x64.bc


$CXX $AMDGCNFLAGS client.cpp -c -o client.gcn.bc
$CXX $AMDGCNFLAGS server.cpp -c -o server.gcn.bc

# $CXX $NVPTXFLAGS client.cpp -c -o client.ptx.bc

$CXX $AMDGCNFLAGS x64_host_amdgcn_client.cpp -c -o x64_host_amdgcn_client.gcn.bc
$CXX $X64FLAGS -I$HSAINC x64_host_amdgcn_client.cpp -c -o x64_host_amdgcn_client.x64.bc

$CXX $AMDGCNFLAGS x64_host_amdgcn_client_api.cpp -c -o x64_host_amdgcn_client_api.gcn.bc
$CXX $X64FLAGS -I$HSAINC x64_host_amdgcn_client_api.cpp -c -o x64_host_amdgcn_client_api.x64.bc


# Build the device loader that assumes the device library is linked into the application
# TODO: Embed it directly in the loader by patching call to main, as the loader doesn't do it
$CXX $X64FLAGS -I$HSAINC amdgcn_loader.cpp -c -o amdgcn_loader.x64.bc
$CXX $LDFLAGS amdgcn_loader.x64.bc memory.x64.bc x64_host_amdgcn_client.x64.bc x64_host_amdgcn_client_api.x64.bc -o amdgcn_loader.exe

# Build the device library that calls into main()
$CXXCL amdgcn_loader_entry.cl -emit-llvm -c -o amdgcn_loader_entry.gcn.bc
$CXX $AMDGCNFLAGS amdgcn_loader_cast.cpp -c -o amdgcn_loader_cast.gcn.bc
$LINK amdgcn_loader_entry.gcn.bc amdgcn_loader_cast.gcn.bc | opt -always-inline -O2 -o amdgcn_loader_device.gcn.bc

# Build the device code that uses said library
$CXX $AMDGCNFLAGS amdgcn_main.cpp -emit-llvm -c -o amdgcn_main.gcn.bc


llvm-link amdgcn_main.gcn.bc amdgcn_loader_device.gcn.bc x64_host_amdgcn_client.gcn.bc x64_host_amdgcn_client_api.gcn.bc  -o executable_device.gcn.bc

# Link the device image
$CXX $AMDGPU executable_device.gcn.bc -o a.out

# Register amdhsa elf magic with kernel
# One off
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' > register

# Persistent
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf



# llc seems to need to be told what architecture it's disassembling
for bc in *.x64.bc ; do
    ll=`echo $bc | sed 's_.bc_.ll_g'`
    opt -strip-debug $bc -S -o $ll
    llc $ll
done

for bc in *.gcn.bc ; do
    ll=`echo $bc | sed 's_.bc_.ll_g'`
    opt -strip-debug $bc -S -o $ll
    llc --mcpu=gfx906 $ll
done


rm -f states.exe
$CXX tests.x64.bc x64_hazard_test.x64.bc states.x64.bc catch.o memory.x64.bc x64_host_x64_client.x64.bc x64_host_amdgcn_client.x64.bc x64_host_amdgcn_client_api.x64.bc $LDFLAGS -o states.exe

llvm-extract server.x64.bc -func server_instance_indirect | opt -strip-debug -S -o indir.ll
llvm-extract server.x64.bc -func server_instance_direct | opt -strip-debug -S -o dir.ll
llc indir.ll
llc dir.ll

time ./states.exe hazard

echo "Call executable"
time ./a.out ; echo $?

# time valgrind --leak-check=full --fair-sched=yes ./states.exe hazard


#!/bin/bash
set -x

RDIR=$HOME/rocm/aomp

HSAINC="$(readlink -f ~/aomp/rocr-runtime/src/inc/)"
HSALIBDIR="$(readlink -f ~/rocm/aomp/hsa/lib/)"
HSALIB="$HSALIBDIR/libhsa-runtime64.so"

CC="clang -std=c99 -Wall -Wextra"
CXX="clang++ -std=c++11 -Wall -Wextra " # -DNDEBUG"
LDFLAGS="-pthread $HSALIB -Wl,-rpath=$HSALIBDIR"
LLC="llc"
LINK="llvm-link"
OPT="opt"

GPU="--target=amdgcn-amd-amdhsa -march=gfx906 -mcpu=gfx906"

X64FLAGS="-g -O2 -emit-llvm -pthread"
AMDGCNFLAGS="-O2 -emit-llvm -ffreestanding $GPU"

CXXCL="clang++ -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 $GPU"


# time $CXX -O3 catch.cpp -c -o catch.o
rm -rf *.s *.ll *.bc *.exe *device.o

# Build the device loader that assumes the device library is linked into the application
# TODO: Embed it directly in the loader by patching call to main, as the loader doesn't do it
$CXX $X64FLAGS -I$HSAINC amdgcn_loader.cpp -c -o amdgcn_loader.x64.bc
$CXX $LDFLAGS amdgcn_loader.x64.bc -o amdgcn_loader.exe

# Build the device library that calls into main()
$CXXCL amdgcn_loader_entry.cl -emit-llvm -c -o amdgcn_loader_entry.gcn.bc
$CXX $AMDGCNFLAGS amdgcn_loader_cast.cpp -c -o amdgcn_loader_cast.gcn.bc
$LINK amdgcn_loader_entry.gcn.bc amdgcn_loader_cast.gcn.bc -internalize -internalize-public-api-list="device_entry" | opt -O2 -o amdgcn_loader_device.gcn.bc
llvm-dis amdgcn_loader_device.gcn.bc
llc --mcpu=gfx906 amdgcn_loader_device.gcn.bc


# Build the device code that uses said library
$CXX $AMDGCNFLAGS amdgcn_main.cpp -emit-llvm -c -o amdgcn_main.gcn.bc


llvm-link amdgcn_main.gcn.bc amdgcn_loader_device.gcn.bc -o executable_device.gcn.bc

# Link the device image
$CXX $GPU executable_device.gcn.bc -o a.out


./a.out other arguments woo

# Register amdhsa elf magic with kernel
# One off
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' > register

# Persistent
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf



$CXX $X64FLAGS states.cpp -c -o states.x64.bc


$CXX $X64FLAGS client.cpp -c -o client.x64.bc
$CXX $X64FLAGS server.cpp -c -o server.x64.bc

$CXX $X64FLAGS x64_host_x64_client.cpp -c -o x64_host_x64_client.x64.bc

$CXX $X64FLAGS tests.cpp -c -o tests.x64.bc


$CXX $AMDGCNFLAGS x64_host_amdgcn_client.cpp -c -o x64_host_amdgcn_client.gcn.bc

$CXX $X64FLAGS -I$HSAINC x64_host_amdgcn_client.cpp -c -o x64_host_amdgcn_client.x64.bc


$CXX $AMDGCNFLAGS client.cpp -c -o client.gcn.bc
$CXX $AMDGCNFLAGS server.cpp -c -o server.gcn.bc
 
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


# $CXX catch.o x64_host_amdgcn_client.bc $LDFLAGS -o hsa.exe && time ./hsa.exe

rm -f states.exe
$CXX tests.x64.bc states.x64.bc catch.o x64_host_x64_client.x64.bc $LDFLAGS -o states.exe

time ./states.exe hazard

# time valgrind --leak-check=full --fair-sched=yes ./states.exe hazard


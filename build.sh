#!/bin/bash
set -x

RDIR=$HOME/rocm/aomp

HSAINC="$(readlink -f ~/aomp/rocr-runtime/src/inc/)"
HSALIBDIR="$(readlink -f ~/rocm/aomp/hsa/lib/)"
HSALIB="$HSALIBDIR/libhsa-runtime64.so"

CC="clang -std=c99 -Wall -Wextra"
CXX="clang++ -std=c++11 -Wall -Wextra " #-DNDEBUG"
LDFLAGS="-pthread $HSALIB -Wl,-rpath=$HSALIBDIR"
LLC="llc"
LINK="llvm-link"
OPT="opt"

X64FLAGS="-g -O0 -emit-llvm -pthread"
AMDGCNFLAGS="-O0 -emit-llvm -ffreestanding --target=amdgcn-amd-amdhsa -march=gfx906"

# time $CXX -O3 catch.cpp -c -o catch.o

rm -rf *.s *.ll *.exe


GPU="--target=amdgcn-amd-amdhsa -march=gfx906 -mcpu=gfx906"
CXXCL="clang++ -Wall -Wextra -x cl -Xclang -cl-std=CL2.0 $GPU"

$CXX $X64FLAGS -I$HSAINC amdgcn_loader.cpp -c -o amdgcn_loader.bc
$CXX $LDFLAGS amdgcn_loader.bc -o amdgcn_loader.exe

$CXXCL amdgcn_loader_entry.cl -emit-llvm -c -o amdgcn_loader_entry.bc
llvm-dis amdgcn_loader_entry.bc
llvm-link amdgcn_loader_entry.bc -o device.bc

llc device.bc

$CXX $GPU device.bc -o device.o -Wl,--dynamic-linker=$HOME/hostrpc/amdgcn_loader.exe

# One off
# cd /proc/sys/fs/binfmt_misc/ && echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' > register

# Persistent
# echo ':amdgcn:M:0:\x7f\x45\x4c\x46\x02\x01\x01\x40\x01\x00\x00\x00\x00\x00\x00\x00::/home/amd/hostrpc/amdgcn_loader.exe:' >> /etc/binfmt.d/amdgcn.conf

./amdgcn_loader.exe device.o other arguments woo

exit 0

$CXX $X64FLAGS states.cpp -c -o states.bc


$CXX $X64FLAGS client.cpp -c -o client.x64.bc
$CXX $X64FLAGS server.cpp -c -o server.x64.bc

$CXX $X64FLAGS x64_host_x64_client.cpp -c -o x64_host_x64_client.bc

$CXX $X64FLAGS hostrpc.cpp -c -o hostrpc.x64.bc

$CXX $X64FLAGS tests.cpp -c -o tests.bc

# clang++ -std=c++17 -Wall -Wextra $X64FLAGS -I$HSAINC x64_host_amdgcn_client.cpp -c -o x64_host_amdgcn_client.bc

# $CXX $AMDGCNFLAGS client.cpp -c -o client.amdgcn.bc
# $CXX $AMDGCNFLAGS server.cpp -c -o server.amdgcn.bc


for bc in *.x64.bc *.amdgcn.bc ; do
    ll=`echo $bc | sed 's_.bc_.ll_g'`
#    opt -strip-debug $bc | llvm-extract -func instantiate_try_garbage_collect_word_client | opt -S -o $ll
    llc $ll
done

# $CXX catch.o x64_host_amdgcn_client.bc $LDFLAGS -o hsa.exe && time ./hsa.exe

rm -f states.exe
$CXX tests.bc states.bc catch.o x64_host_x64_client.bc $LDFLAGS -o states.exe

time ./states.exe hazard

time valgrind --leak-check=full --fair-sched=yes ./states.exe hazard


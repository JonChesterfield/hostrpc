#!/bin/bash
set -x
CC="clang -std=c99 -Wall -Wextra"
CXX="clang++ -std=c++11 -Wall -Wextra -DNDEBUG"
LDFLAGS="-pthread"
LLC="llc"
LINK="llvm-link"
OPT="opt"

X64FLAGS="-g -O2 -emit-llvm -pthread"
AMDGCNFLAGS="-O2 -emit-llvm -ffreestanding --target=amdgcn-amd-amdhsa -march=gfx906"

# time $CXX -O3 catch.cpp -c -o catch.o

rm -rf *.ll

$CXX $X64FLAGS states.cpp -c -o states.bc

$CXX $X64FLAGS client.cpp -c -o client.x64.bc
$CXX $X64FLAGS server.cpp -c -o server.x64.bc

$CXX $X64FLAGS tests.cpp -c -o tests.bc

$CXX $AMDGCNFLAGS client.cpp -c -o client.amdgcn.bc
$CXX $AMDGCNFLAGS server.cpp -c -o server.amdgcn.bc


for bc in *.x64.bc *.amdgcn.bc ; do
    ll=`echo $bc | sed 's_.bc_.ll_g'`
    opt -strip-debug $bc | llvm-extract -func instantiate_try_garbage_collect_word_client | opt -S -o $ll
    llc $ll
done

$CXX $LDFLAGS tests.bc states.bc catch.o -o states.exe && time  ./states.exe

$CXX -O2 -ffreestanding --target=amdgcn-amd-amdhsa -march=gfx906 -c shader.cpp -o shader.o


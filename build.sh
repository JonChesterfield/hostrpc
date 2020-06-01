#!/bin/bash
set -x
CC="clang -std=c99 -Wall -Wextra"
CXX="clang++ -std=c++11 -Wall -Wextra"
LDFLAGS="-pthread"
LLC="llc"
LINK="llvm-link"
OPT="opt"

X64FLAGS="-g -O0 -emit-llvm -pthread"
AMDGCNFLAGS="-O0 -emit-llvm -ffreestanding --target=amdgcn-amd-amdhsa -march=gfx906"

# time $CXX -O3 catch.cpp -c -o catch.o

rm -rf *.ll

$CXX $X64FLAGS states.cpp -c -o states.bc

$CXX $X64FLAGS client.cpp -c -o client.bc
$CXX $X64FLAGS server.cpp -c -o server.bc

$CXX $X64FLAGS tests.cpp -c -o tests.bc

$CXX $AMDGCNFLAGS client.cpp -c -o client.amdgcn.bc
$CXX $AMDGCNFLAGS server.cpp -c -o server.amdgcn.bc

$CXX $LDFLAGS tests.bc states.bc catch.o -o states.exe && time  ./states.exe


#!/bin/bash

CC="clang -std=c99 -Wall"
CXX="clang++ -std=c++11 -Wall"
FLAGS=""
LLC="llc"
LINK="llvm-link"
OPT="opt"

# time $CXX -O3 catch.cpp -c -o catch.o

rm -rf *.ll

$CXX $FLAGS -O2 states.cpp -emit-llvm -c -o states.bc

$CXX states.bc catch.o -o states.exe

# for i in *.ll; do $LLC $i; done
time  ./states.exe


#!/bin/bash

CC="clang -std=c99 -Wall -Wextra"
CXX="clang++ -std=c++11 -Wall -Wextra"
FLAGS="-pthread"
LLC="llc"
LINK="llvm-link"
OPT="opt"

# time $CXX -O3 catch.cpp -c -o catch.o

rm -rf *.ll

$CXX $FLAGS -O0 -g states.cpp -emit-llvm -c -o states.bc

$CXX $FLAGS -O0 -g tests.cpp -emit-llvm -c -o tests.bc

$CXX $FLAGS tests.bc states.bc catch.o -o states.exe && time  ./states.exe


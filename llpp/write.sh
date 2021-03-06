#!/bin/bash
set -x
set -e
set -o pipefail

FILE=main.tex
rm -f $FILE

MINIMAL=minimal
REAL=../hostrpc

touch $FILE
cat header >> $FILE
cat paper.txt >> $FILE
cat footer >> $FILE

rm -f a.out
clang++ -std=c++14 -Wall minimal.cpp -pthread -o a.out
./a.out

for i in interface.cpp; do
    sed -e "/$i/{r $i" -e "d}" $FILE > tmp
    mv tmp $FILE
done
    
for i in header.cpp client.cpp server.cpp main.cpp; do
    sed -e "/$i/{r $MINIMAL/$i" -e "d}" $FILE > tmp
    mv tmp $FILE
done

for i in openmp_hostcall_amdgpu.cpp openmp_hostcall_host.cpp demo_kernel.hip; do
    sed -e "/$i/{r $REAL/$i" -e "d}" $FILE > tmp
    mv tmp $FILE
done

#!/bin/bash

rm -rf lib demo_openmp_gcn demo_openmp_ptx dump_kernels

for dir in "." codegen loader detail obj ".." prototype; do
    rm -rf $dir/*.ll $dir/*.bc $dir/*.s $dir/*.obj $dir/*.exe $dir/*.gcn $dir/*.ptx $dir/*.so $dir/*device.o $dir/*.out
done

exit 0

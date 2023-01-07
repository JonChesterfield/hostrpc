#!/bin/bash

rm -rf lib demo_openmp_gcn demo_openmp_ptx hello.o hello

for dir in "." codegen tools/loader detail obj unit_tests obj/unit_tests obj/libc_wip ".." prototype; do
    rm -rf $dir/*.ll $dir/*.bc $dir/*.ii $dir/*.s $dir/*.obj $dir/*.exe $dir/*.gcn $dir/*.ptx $dir/*.so $dir/*device.o $dir/*.out
done

exit 0

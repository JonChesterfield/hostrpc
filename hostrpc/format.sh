#!/bin/bash

FMT=$HOME/llvm-install/bin/clang-format

for i in `find . -type f -iname "*.?pp" -a -not -iname "catch.*" -a -not -iwholename "./platform/detect.hpp"`; do
    echo "Formatting $i"
    $FMT -i $i
done

for i in `find . -type f -iname "*.cu"`; do
    echo "Formatting $i"
    $FMT -i $i
done

for i in `find . -type f -iname "*.hip"`; do
    echo "Formatting $i"
    $FMT -i $i
done


for i in `find . -type f -iname "*.inc"`; do
    echo "Formatting $i"
    $FMT -i $i
done

exit

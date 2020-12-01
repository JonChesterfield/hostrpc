#!/bin/bash

for i in `find . -type f -iname "*.?pp" -a -not -iname "catch.*" -a -not -iwholename "./detail/platform_detect.hpp"`; do
    echo "Formatting $i"
    clang-format -i $i
done

for i in `find . -type f -iname "*.cu"`; do
    echo "Formatting $i"
    clang-format -i $i
done

for i in `find . -type f -iname "*.hip"`; do
    echo "Formatting $i"
    clang-format -i $i
done


for i in `find . -type f -iname "*.inc"`; do
    echo "Formatting $i"
    clang-format -i $i
done

exit

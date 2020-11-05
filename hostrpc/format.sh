#!/bin/bash

for i in `find . -type f -iname "*.?pp" -a -not -iname "catch.*"`; do
    echo "Formatting $i"
    clang-format -i $i
done

for i in `find . -type f -iname "*.cu"`; do
    echo "Formatting $i"
    clang-format -i $i
done

for i in `find . -type f -iname "*.inc"`; do
    echo "Formatting $i"
    clang-format -i $i
done

exit

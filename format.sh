#!/bin/bash

for i in `find . -type f -iname "*.?pp" -a -not -iname "catch.*"`; do
    echo "Formatting $i"
    clang-format -i $i
done

exit

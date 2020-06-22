#!/bin/bash

for dir in "." loader codegen; do 
    rm -rf $dir/*.s $dir/*.ll $dir/*.bc $dir/*.exe $dir/*device.o $dir/a.out
done

exit 0

#!/bin/bash

for dir in "." loader detail codegen ".."; do
    rm -rf $dir/*.ll $dir/*.bc $dir/*.s $dir/*.obj $dir/*.exe $dir/*.so $dir/*device.o $dir/a.out
done

exit 0

#!/bin/bash

cp dispatch_id.ll tmp
for dir in "." loader codegen ".."; do
    rm -rf $dir/*.ll $dir/*.bc $dir/*.s $dir/*.obj $dir/*.exe $dir/*.so $dir/*device.o $dir/a.out
done
mv tmp dispatch_id.ll

exit 0

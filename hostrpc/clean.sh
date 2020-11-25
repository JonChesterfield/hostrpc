#!/bin/bash

for dir in "." loader detail codegen obj lib ".."; do
    rm -rf $dir/*.ll $dir/*.bc $dir/*.s $dir/*.obj $dir/*.exe $dir/*.gcn $dir/*.ptx $dir/*.so $dir/*device.o $dir/a.out $dir/a.gcn.out
done

exit 0

#!/bin/bash
set -x
set -e
set -o pipefail

MSGPACKINCLUDE="thirdparty/msgpack"
if [[ -d $MSGPACKINCLUDE ]]
then
    echo "Using existing msgpack"
    cd $MSGPACKINCLUDE && git pull && cd -
else
    echo "Cloning msgpack"
    git clone https://github.com/jonchesterfield/msgpack.git $MSGPACKINCLUDE
fi

clang++ -std=c++17 tools/dump_kernels.cpp $MSGPACKINCLUDE/msgpack.cpp -I../impl -I. -I$MSGPACKINCLUDE -lelf -o dump_kernels

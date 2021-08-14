#!/bin/bash

set -x
set -e
set -o pipefail

TREE=$HOME/llvm-project
SRC=$HOME/hostrpc/hostrpc

# openmp_hostcall compiles as x64 and gcn, includes enough files to be self contained
# add it to the corresponding runtime libraries
echo '#include "'$SRC'/openmp_hostcall.cpp"' > $TREE/openmp/libomptarget/plugins/amdgpu/src/openmp_hostcall.cpp

echo '#include "'$SRC'/openmp_hostcall.cpp"' > $TREE/openmp/libomptarget/deviceRTLs/amdgcn/src/openmp_hostcall.cpp

CMAKE=$TREE/openmp/libomptarget/deviceRTLs/amdgcn/CMakeLists.txt
grep -v openmp_hostcall.cpp $CMAKE > tmp

cat tmp |  sed  's#${CMAKE_CURRENT_SOURCE_DIR}/src/target_impl.hip#${CMAKE_CURRENT_SOURCE_DIR}/src/target_impl.hip\n  ${CMAKE_CURRENT_SOURCE_DIR}/src/openmp_hostcall.cpp#g' > $CMAKE

CMAKE=$TREE/openmp/libomptarget/plugins/amdgpu/CMakeLists.txt
grep -v openmp_hostcall.cpp $CMAKE > tmp

cat tmp | sed 's#src/rtl.cpp#src/rtl.cpp\n      src/openmp_hostcall.cpp#g' > $CMAKE


# Create a stdio.h wrapper that defines a printf macro for openmp

cat << EOF > $TREE/clang/lib/Headers/openmp_wrappers/stdio.h
#include_next <stdio.h>

#ifndef __CLANG_OPENMP_STDIO_H__
#define __CLANG_OPENMP_STDIO_H__

#ifndef _OPENMP
#error "This file is for OpenMP compilation only."
#endif

#include "$SRC/hostrpc_printf.h"
#define printf(...) __hostrpc_printf(__VA_ARGS__)

#endif

EOF

CMAKE=$TREE/clang/lib/Headers/CMakeLists.txt
grep -v stdio.h $CMAKE > tmp

cat tmp | sed 's#openmp_wrappers/new#openmp_wrappers/new\n  openmp_wrappers/stdio.h#g' > $CMAKE

rm tmp

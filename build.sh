#!/usr/bin/env bash

## Usage: ./build.sh [toolchain]

if [ -z $1 ]; then
    TOOLCHAIN=clang
    BUILD_DIR=build
else
    TOOLCHAIN=$1
    BUILD_DIR=build_${TOOLCHAIN}
fi
echo "Using toolchain: ${TOOLCHAIN}"   

rm -rf ${BUILD_DIR}
mdkir ${BUILD_DIR}
cmake -DCMAKE_TOOLCHAIN_FILE=toolchains/${TOOLCHAIN}.cmake -H. -B${BUILD_DIR}
make -j -C ${BUILD_DIR}
cd ${BUILD_DIR}; ctest --output-on-failure




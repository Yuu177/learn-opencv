#!/bin/bash
set -e
set -x

mkdir -p build 2>/dev/null
cd build

conan install ../conanfile.txt -s compiler.libcxx=libstdc++11
cmake ..
make -j16

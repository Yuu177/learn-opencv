#!/bin/bash
set -e
set -x

# LD_LIBRARY_PATH 告诉 loader 在哪些目录中可以找到共享库。可以设置多个搜索目录，这些目录之间用冒号分隔开
export LD_LIBRARY_PATH=../build/runtime:${LD_LIBRARY_PATH}
../build/bin/head-pose-estimation

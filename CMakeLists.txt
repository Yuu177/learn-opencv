cmake_minimum_required(VERSION 3.2)
project(learn-opencv)
add_definitions(-std=c++11)

# 完成 conan 的基本配置
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# 生成 clangd 依赖的 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(face-detection)
add_subdirectory(face-landmark-detection)
add_subdirectory(head-pose-estimation)

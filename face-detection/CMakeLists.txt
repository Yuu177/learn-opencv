cmake_minimum_required(VERSION 3.2)
project(face-detection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

ADD_EXECUTABLE(face-detection main.cpp)
TARGET_LINK_LIBRARIES(face-detection ${CONAN_LIBS})

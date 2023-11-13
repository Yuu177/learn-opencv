cmake_minimum_required(VERSION 3.2)
project(face-landmark-detection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

ADD_EXECUTABLE(face-landmark-detection main.cpp)
TARGET_LINK_LIBRARIES(face-landmark-detection ${CONAN_LIBS})

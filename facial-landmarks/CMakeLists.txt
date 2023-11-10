cmake_minimum_required(VERSION 3.2)
project(facial-landmarks)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

ADD_EXECUTABLE(facial-landmarks main.cpp)
TARGET_LINK_LIBRARIES(facial-landmarks ${CONAN_LIBS})

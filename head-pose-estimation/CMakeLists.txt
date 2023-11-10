cmake_minimum_required(VERSION 3.2)
project(head-pose-estimation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

ADD_EXECUTABLE(head-pose-estimation main.cpp)
# 变量 CONAN_LIBS 中包含了你用到的所有 conan 包中含有的 libs 的名称，
# 你可以直接使用这个变量，若你知道你该 link 哪些 libs，当然你也可以自己写 libs 名称。
# 一般推荐自己写 libs 名称，这样会避免链接了不必要的 libs
TARGET_LINK_LIBRARIES(head-pose-estimation ${CONAN_LIBS})

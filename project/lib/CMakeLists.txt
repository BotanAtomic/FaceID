cmake_minimum_required(VERSION 3.16)
project(ML-framework)
project(ML-test)

set(CMAKE_CXX_STANDARD 14)

add_library(ML-framework SHARED library.cpp library.h)

add_executable(ML-test library.cpp library.h test.cpp)
cmake_minimum_required(VERSION 3.10.2)
project(ML-framework)
project(ML-test)

set(CMAKE_CXX_STANDARD 14)

add_library(ML-framework SHARED library.cpp)

add_executable(ML-test library.cpp test.cpp)
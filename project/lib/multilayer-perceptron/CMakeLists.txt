cmake_minimum_required(VERSION 3.10.2)
project(ML-framework)
project(ML-test)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Oi")

macro(get_WIN32_WINNT version)
    if(WIN32)
        enable_language(CUDA)
        find_package(CUDA 10.0 REQUIRED)
        message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
        set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
        cuda_select_nvcc_arch_flags(ARCH_FLAGS)
        include_directories("${CUDA_INCLUDE_DIRS}")
    endif()
endmacro()

get_WIN32_WINNT(ver)
add_definitions(-D_WIN32_WINNT=${ver})

file(GLOB_RECURSE PROJECT_SRC src/*.Macpp src/*.h src/*.cpp ./src/*.cu)

add_library(ML-framework SHARED ${PROJECT_SRC})
add_executable(ML-test ${PROJECT_SRC})

target_link_libraries(ML-framework ${CUDA_LIBRARIES})
target_link_libraries(ML-test ${CUDA_LIBRARIES})


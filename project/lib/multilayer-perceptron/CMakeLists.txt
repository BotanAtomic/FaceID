cmake_minimum_required(VERSION 3.10.2)
project(ML-framework)
project(ML-test)

set(CMAKE_CXX_STANDARD 14)

add_library(ML-framework SHARED
        src/library.cpp
        src/layer/Layer.cpp src/layer/Layer.h
        src/network/MultiLayerNetwork.cpp src/network/MultiLayerNetwork.h
        src/matrix/Matrix.h src/matrix/Matrix.cpp
        src/utils/utils.h src/utils/utils.cpp
        src/activation/ActivationFunction.h src/activation/implementation/Hyperbolic.h
        src/activation/implementation/Sigmoid.h src/activation/implementation/Softmax.h
        src/activation/implementation/Relu.h src/activation/implementation/Linear.h
        src/initializer/Initializer.h src/initializer/implementation/RandomUniform.h src/file/FileWriter.cpp src/file/FileWriter.h src/file/FileReader.cpp src/file/FileReader.h)

add_executable(ML-test
        src/test.cpp
        src/library.cpp
        src/layer/Layer.cpp src/layer/Layer.h
        src/network/MultiLayerNetwork.cpp src/network/MultiLayerNetwork.h
        src/matrix/Matrix.h src/matrix/Matrix.cpp
        src/utils/utils.h src/utils/utils.cpp
        src/activation/ActivationFunction.h src/activation/implementation/Hyperbolic.h
        src/activation/implementation/Sigmoid.h src/activation/implementation/Softmax.h
        src/activation/implementation/Relu.h src/activation/implementation/Linear.h
        src/initializer/Initializer.h src/initializer/implementation/RandomUniform.h src/file/FileWriter.cpp src/file/FileWriter.h src/file/FileReader.cpp src/file/FileReader.h)

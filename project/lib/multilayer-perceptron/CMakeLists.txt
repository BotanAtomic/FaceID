cmake_minimum_required(VERSION 3.10.2)
project(ML-framework)
project(ML-test)

set(CMAKE_CXX_STANDARD 14)

add_library(ML-framework SHARED
        library.cpp
        layer/Layer.cpp layer/Layer.h
        network/MultiLayerNetwork.cpp network/MultiLayerNetwork.h
        matrix/Matrix.h matrix/Matrix.cpp
        utils/utils.h utils/utils.cpp
        activation/ActivationFunction.h activation/implementation/Hyperbolic.h
        activation/implementation/Sigmoid.h activation/implementation/Softmax.h
        activation/implementation/Relu.h)

add_executable(ML-test
        test.cpp
        layer/Layer.cpp layer/Layer.h
        network/MultiLayerNetwork.cpp network/MultiLayerNetwork.h
        matrix/Matrix.h matrix/Matrix.cpp
        utils/utils.h utils/utils.cpp
        activation/ActivationFunction.h activation/implementation/Hyperbolic.h
        activation/implementation/Sigmoid.h activation/implementation/Softmax.h
        activation/implementation/Relu.h)

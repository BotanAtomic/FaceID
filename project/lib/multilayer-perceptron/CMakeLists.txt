cmake_minimum_required(VERSION 3.16)
project(ML-framework)
project(ML-test)

set(CMAKE_CXX_STANDARD 14)

add_library(ML-framework SHARED library.cpp network/neuron/Neuron.cpp network/neuron/Neuron.h network/layer/Layer.cpp
        network/layer/Layer.h network/MultiLayerNetwork.cpp network/MultiLayerNetwork.h matrix/Matrix.h utils/maths.h)

add_executable(ML-test test.cpp network/neuron/Neuron.cpp network/neuron/Neuron.h network/layer/Layer.cpp
        network/layer/Layer.h network/MultiLayerNetwork.cpp network/MultiLayerNetwork.h matrix/Matrix.h utils/maths.h)
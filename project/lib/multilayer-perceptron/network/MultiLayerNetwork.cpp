//
// Created by botan on 5/31/2020.
//

#include <iostream>
#include "MultiLayerNetwork.h"
#include "../utils/maths.h"

MultiLayerNetwork::MultiLayerNetwork(long inputSize, int hiddenLayer, long hiddenLayerSize, long outputSize,
                                     double learningRate) {
    this->inputSize = inputSize;
    this->hiddenLayer = hiddenLayer;
    this->outputSize = outputSize;
    this->learningRate = learningRate;

    auto *previousLayer = new Layer(hiddenLayerSize, inputSize);

    for (long i = 0; i < hiddenLayer; i++) {
        auto *layer = new Layer(hiddenLayerSize, previousLayer->getSize());
        this->layers.push_back(layer);
        previousLayer = layer;
    }

    this->layers.push_back(new Layer(outputSize, previousLayer->getSize())); //output layer
}

void MultiLayerNetwork::train(double *inputs, double *labels, int size, int epochs) {
    Matrix inputsMatrix(inputs, this->inputSize, size);
    Matrix labelsMatrix(labels, 1, size);

    inputsMatrix.dump("inputs matrix");
    labelsMatrix.dump("labels matrix");

    Matrix z(0, 0);
    for (int _ = 0; _ < epochs; _++) {
        Layer *layer = this->layers[0];
        Layer *outputLayer = this->layers[1];

        Matrix product = inputsMatrix * layer->getWeights();
        product.apply(sigmoid);

        z = (product * outputLayer->getWeights());

        Matrix error = (labelsMatrix - z) * learningRate;
        outputLayer->updateWeights(product.transpose() * error);

        product.apply(sigmoid_);
        layer->updateWeights(inputsMatrix.transpose() * (error * outputLayer->getWeights().transpose()).multiply(product));
    }
    z.dump("result");
}



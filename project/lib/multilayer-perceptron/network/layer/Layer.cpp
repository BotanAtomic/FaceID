//
// Created by botan on 5/31/2020.
//

#include <iostream>
#include "Layer.h"

Layer::Layer(long size, long previousSize) {
    this->size = size;
    this->previousSize = previousSize;
    for (long i = 0; i < size; i++) {
        this->neurons.push_back(new Neuron(previousSize));
    }
}

long Layer::getSize() {
    return this->neurons.size();
}

Neuron *Layer::getNeurons(long index) {
    return this->neurons[index];
}

Matrix Layer::getWeights() {
    Matrix matrix(this->previousSize, this->size);

    for (long i = 0; i < this->size; i++) {
        Neuron *neuron = this->getNeurons(i);
        for (int j = 0; j < this->previousSize; j++) {
            matrix[i][j] = neuron->getWeights(j);
        }
    }

    return matrix;
}

void Layer::updateWeights(Matrix matrix) {
    for (long layerIndex = 0; layerIndex < this->size; layerIndex++) {
        Neuron *toUpdate = this->neurons[layerIndex];

        for (int j = 0; j < toUpdate->getSize(); j++) {
            toUpdate->updateWeights(j, matrix[layerIndex][j]);
        }
    }
}

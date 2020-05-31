//
// Created by botan on 5/31/2020.
//

#include "Neuron.h"

Neuron::Neuron(int size) {
    this->size = size;
    this->bias = 0;
    this->output = 0;

    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0.0, 1.0);

    for (int i = 0; i < size; i++) {
        this->weights.push_back(dist(e2));
    }
}

int Neuron::getSize() const {
    return this->size;
}

double Neuron::getOutput() const {
    return this->output;
}

double Neuron::getWeights(int index) {
    return this->weights[index];
}

double Neuron::getError() const {
    return this->error;
}

void Neuron::updateWeights(int index, double value) {
    this->weights[index] += value;
}

void Neuron::updateBias(double value) {
    this->bias += value;
}

//
// Created by botan on 5/31/2020.
//

#include "Layer.h"

Layer::Layer(const string &name, int neurons, ActivationFunction * activation) {
    this->name = name;
    this->neurons = neurons;
    this->outputs = new Matrix(neurons);
    this->errors = new Matrix(neurons);
    this->activation = activation;
}

void Layer::initialize(int inputSize) {
    this->weights = new Matrix(this->neurons, inputSize);

    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(-1.0f, 1.0f);

    for (int i = 0; i < this->neurons * inputSize; i++) {
        (*this->weights).set(i, dist(e2));
    }
}

int Layer::getSize() const {
    return this->neurons;
}

string Layer::getName() {
    return this->name;
}

Matrix * Layer::getWeights() {
    return this->weights;
}

Matrix * Layer::getOutputs() {
    return this->outputs;
}

Matrix * Layer::getErrors() {
    return this->errors;
}

ActivationFunction * Layer::getActivation() {
    return this->activation;
}

void Layer::setOutputs(Matrix & o) {
    *outputs = o;
}

void Layer::computeErrors(const vector<double> &currentErrors) {
    for (int i = 0; i < neurons; i++) {
        errors->set(i, currentErrors[i] * activation->getDerivation(outputs->get(i)));
    }
}

void Layer::updateWeights(Matrix * correction) {
    weights->add(*correction);
}
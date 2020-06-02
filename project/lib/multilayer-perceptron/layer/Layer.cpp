//
// Created by botan on 5/31/2020.
//

#include "Layer.h"

Layer::Layer(const string &name, int neurons, ActivationFunction * activation) {
    this->name = name;
    this->neurons = neurons;
    this->outputs.assign(this->neurons, 0.0f);
    this->errors.assign(this->neurons, 0.0f);
    this->activation = activation;
}

void Layer::initialize(int inputSize) {
    this->weights = Matrix(this->neurons, inputSize);

    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0.0f, 1.0f);

    for (int i = 0; i < this->neurons * inputSize; i++) {
        this->weights.set(i, dist(e2));
    }
}

int Layer::getSize() const {
    return this->neurons;
}

Matrix Layer::getWeights() {
    return this->weights;
}

string Layer::getName() {
    return this->name;
}

vector<double> Layer::getOutputs() {
    return this->outputs;
}

vector<double> Layer::getErrors() {
    return this->errors;
}

ActivationFunction * Layer::getActivation() {
    return this->activation;
}

void Layer::setOutputs(const vector<double> &o) {
    for (int i = 0; i < o.size(); i++) {
        outputs[i] = o[i];
    }
}

void Layer::computeErrors(const vector<double> &currentErrors) {
    for (int i = 0; i < neurons; i++) {
        errors[i] = currentErrors[i] * activation->getDerivation(outputs[i]);
    }
}

void Layer::updateWeights(Matrix correction) {
    weights.add(std::move(correction));
}
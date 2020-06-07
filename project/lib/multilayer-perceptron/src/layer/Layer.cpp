//
// Created by botan on 5/31/2020.
//

#include "Layer.h"

Layer::Layer(const string &name, int neurons, ActivationFunction *activation, Initializer *initializer) {
    this->name = name;
    this->neurons = neurons;
    this->outputs = new Matrix(neurons);
    this->errors = new Matrix(neurons);
    this->activation = activation;
    this->initializer = initializer;
    this->weights = nullptr;
}

void Layer::initialize(int inputSize) {
    cout << "initialize layer " << name << endl;
    this->weights = new Matrix(this->neurons, inputSize);
    cout << "initialize layer [1] " << name << endl;

    this->initializer->fill(*this->weights);
    cout << "layer " << name << " initialized" << endl;
}

void Layer::initialize(Matrix *loadedWeights) {
    this->weights = loadedWeights;
}

int Layer::getSize() const {
    return this->neurons;
}

string Layer::getName() {
    return this->name;
}

Matrix *Layer::getWeights() {
    return this->weights;
}

Matrix *Layer::getOutputs() {
    return this->outputs;
}

Matrix *Layer::getErrors() {
    return this->errors;
}

ActivationFunction *Layer::getActivation() {
    return this->activation;
}

void Layer::setOutputs(Matrix &o) {
    *outputs = o;
}

void Layer::computeErrors(const vector<double> &currentErrors) {
    for (int i = 0; i < neurons; i++) {
        errors->set(i, currentErrors[i] * activation->getDerivation(outputs->get(i)));
    }
}

void Layer::updateWeights(Matrix *correction) {
    weights->add(*correction);
}
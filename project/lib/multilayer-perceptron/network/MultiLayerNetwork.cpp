//
// Created by botan on 5/31/2020.
//

#include "MultiLayerNetwork.h"

static int o = 0;

MultiLayerNetwork::MultiLayerNetwork(int inputSize) {
    this->inputSize = inputSize;
}

void MultiLayerNetwork::addLayer(const string &name, int size) {
    layers.push_back(new Layer(name, size, new Sigmoid()));
}

void MultiLayerNetwork::addLayer(const string &name, int size, ActivationFunction *activationFunction) {
    layers.push_back(new Layer(name, size, activationFunction));
}

void MultiLayerNetwork::initialize() {
    int inputNeurons = inputSize;
    for (int i = 0; i < layers.size(); i++) {
        if (i != 0)
            inputNeurons = layers[i - 1]->getSize();

        layers[i]->initialize(inputNeurons);
    }
}


void MultiLayerNetwork::train(double *inputs, double *labels, int size, int epochs, double alpha) {
    cout << "Start training model: size=" << size << ", epochs=" << epochs << ", alpha=" << alpha << endl;

    Matrix inputsMatrix(inputs, inputSize, size);
    inputsMatrix.dump("input matrix");

    cout << "labels: " << vectorToString(vector<double>(labels, labels + size)) << endl;

    for (int _ = 1; _ <= epochs; _++) {
        double error = 0;
        for (int i = 0; i < size; i++) {
            vector<double> networkPredictions = predict(inputsMatrix[i]);


            vector<double> expectedPredictions(networkPredictions.size(), 0.0f);

            if (networkPredictions.size() > 1) {
                expectedPredictions[labels[i]] = 1;
            } else {
                expectedPredictions[0] = labels[i];
            }

            for (int j = 0; j < networkPredictions.size(); j++)
                error += pow((expectedPredictions[j] - networkPredictions[j]), 2);

            backPropagation(expectedPredictions);
            updateWeights(inputsMatrix[i], alpha);
        }

        if (_ % 100 == 0)
            cout << "> epochs " << _ << " - error=" << error << endl;
    }
}

vector<double> MultiLayerNetwork::predict(double *inputs) {
    Matrix matrix(inputs, 1, this->inputSize);

    for (auto &layer : layers) {
        matrix = layer->getWeights()->dot(matrix);
        layer->getActivation()->activate(matrix);
        layer->setOutputs(matrix);
    }

    return matrix.toVector();
}

void MultiLayerNetwork::backPropagation(const vector<double> &expectedPredictions) {
    for (int i = layers.size(); i > 0; i--) {
        Layer *layer = layers[i - 1];
        vector<double> errors;

        if (i != layers.size()) {
            Layer *nextLayer = layers[i];
            Matrix error = nextLayer->getErrors()->T();
            errors = nextLayer->getWeights()->T().dot(error).toVector();
        } else {
            errors.assign(layer->getSize(), 0.0f);
            for (int j = 0; j < layer->getSize(); j++)
                errors[j] = expectedPredictions[j] - layer->getOutputs()->get(j);
        }
        layer->computeErrors(errors);
    }
}

void MultiLayerNetwork::updateWeights(double *inputs, double alpha) {
    vector<double> currentInputs(inputs, inputs + inputSize);
    for (int _ = 0; _ < layers.size(); _++) {
        Layer *layer = layers[_];
        if (_ != 0)
            currentInputs = layers[_ - 1]->getOutputs()->toVector();

        Matrix transversedInputs = Matrix(currentInputs).T();
        Matrix delta = layer->getErrors()->T().dot(transversedInputs);

        layer->updateWeights(delta * alpha);

    }
}


void MultiLayerNetwork::dump() {
    for (auto layer : layers) {
        cout << layer->getName() << " (" << layer->getSize() << " neurons, " << layer->getWeights()->getColumns()
             << " input neurons): " << endl << layer->getWeights()->toString() << endl;
    }
}




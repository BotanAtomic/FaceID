//
// Created by botan on 5/31/2020.
//

#include "MultiLayerNetwork.h"

#include <utility>

using namespace std::chrono;

MultiLayerNetwork::MultiLayerNetwork(int inputSize) {
    this->inputSize = inputSize;
}

MultiLayerNetwork::MultiLayerNetwork(int inputSize, vector<Layer *> layers) {
    this->inputSize = inputSize;
    this->layers = std::move(layers);
}


void MultiLayerNetwork::addLayer(const string &name, int size, ActivationFunction *activationFunction,
                                 Initializer *initializer) {
    layers.push_back(new Layer(name, size, activationFunction, initializer));
}

void MultiLayerNetwork::initialize() {
    int inputNeurons = inputSize;
    for (int i = 0; i < layers.size(); i++) {
        if (i != 0)
            inputNeurons = layers[i - 1]->getSize();

        layers[i]->initialize(inputNeurons);
    }
}


void MultiLayerNetwork::train(double *inputs, double *labels, int samples, int epochs, double alpha) {
    cout << "Start training network: inputsSize=" << inputSize << ", outputSize="
         << layers[layers.size() - 1]->getSize()
         << ", samples=" << samples << ", epochs=" << epochs << ", alpha=" << alpha << endl;

    Matrix inputsMatrix(inputs, samples, inputSize);
    int64_t timestamp, total, remaining;

    for (int _ = 1; _ <= epochs; _++) {
        double error = 0;
        timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        for (int i = 0; i < samples; i++) {
            vector<double> networkPredictions = predict(inputsMatrix[i]);
            vector<double> expectedPredictions(networkPredictions.size(), 0.0f);

            if (isClassification()) {
                expectedPredictions[labels[i]] = 1;
            } else {
                expectedPredictions[0] = labels[i];
            }

            for (int j = 0; j < networkPredictions.size(); j++)
                error += pow((expectedPredictions[j] - networkPredictions[j]), 2);

            backPropagation(expectedPredictions);
            updateWeights(inputsMatrix[i], alpha);
        }

        if (epochs < 20 || _ % (epochs / 20) == 0) {
            total = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - timestamp;
            remaining = ((epochs - _)) * total;
            printf("> epoch %d/%d - %llds - loss: %lf - ETA: %llds\n", _, epochs, total / 1000, error,
                   remaining / 1000);
        }
    }
}

vector<double> MultiLayerNetwork::predict(double *inputs) {
    Matrix matrix(inputs, this->inputSize, 1);

    int i = 0;
    for (auto &layer : layers) {
        i++;
        matrix = layer->getWeights()->dot(matrix);

        if (i < layers.size())
            matrix.add(*layer->getBias());

        if (i < layers.size() || isClassification())
            layer->getActivation()->activate(matrix);

        Matrix output = matrix.T();
        layer->setOutputs(output);
    }

    return matrix.toVector();
}

void MultiLayerNetwork::backPropagation(const vector<double> &expectedPredictions) {
    for (int i = layers.size(); i > 0; i--) {
        Layer *layer = layers[i - 1];
        vector<double> errors;

        if (i != layers.size()) {
            Layer *nextLayer = layers[i];
            errors = nextLayer->getWeights()->T().dot(*nextLayer->getErrors()).toVector();
        } else {
            errors = vector<double>(layer->getSize());
            for (int j = 0; j < layer->getSize(); j++) {
                errors[j] = expectedPredictions[j] - layer->getOutputs()->get(j);
            }
        }
        layer->computeErrors(errors, i != layers.size() || isClassification());
    }
}

void MultiLayerNetwork::updateWeights(double *inputs, double alpha) {
    Matrix currentInputs(inputs, 1, inputSize);

    for (int i = 0; i < layers.size(); i++) {
        Layer *layer = layers[i];
        if (i != 0)
            currentInputs = *layers[i - 1]->getOutputs();


        Matrix delta = layer->getErrors()->dot(currentInputs);

        layer->getBias()->add(*(*layer->getErrors() * alpha));
        layer->updateWeights(delta * alpha);
    }
}


void MultiLayerNetwork::dump() {
    for (auto layer : layers) {
        cout << layer->getName() << " [" << layer->getSize() << " neurons, " << layer->getWeights()->getColumns()
             << " inputs, activation=" << layer->getActivation()->getName() << "]: " << endl
             << layer->getWeights()->toString() << endl;
    }
}

void MultiLayerNetwork::save(const string &path) {
    FileWriter fileWriter(path);

    if (!fileWriter.isOpen())
        return;

    fileWriter.writeInt(inputSize);
    fileWriter.writeInt(layers.size());

    for (Layer *layer: layers) {
        fileWriter.writeString(layer->getName());
        fileWriter.writeInt(layer->getSize());
        fileWriter.writeString(layer->getActivation()->getName());
        fileWriter.writeMatrix(layer->getWeights());
        fileWriter.writeMatrix(layer->getBias());
    }

    fileWriter.close();
}

MultiLayerNetwork *MultiLayerNetwork::load(const string &path) {
    FileReader fileReader(path);

    if (!fileReader.isOpen())
        return nullptr;

    int inputSize = fileReader.readInt();
    int layersSize = fileReader.readInt();

    vector<Layer *> layers;

    for (int i = 0; i < layersSize; i++) {
        string name = fileReader.readString();
        int neurons = fileReader.readInt();
        string activationName = fileReader.readString();
        Matrix *weights = fileReader.readMatrix();
        Matrix *bias = fileReader.readMatrix();

        auto *layer = new Layer(name, neurons, getActivation(activationName), nullptr);
        layer->initialize(weights, bias);
        layers.push_back(layer);
    }

    return new MultiLayerNetwork(inputSize, layers);
}

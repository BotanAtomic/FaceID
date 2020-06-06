//
// Created by botan on 5/31/2020.
//

#include "MultiLayerNetwork.h"

MultiLayerNetwork::MultiLayerNetwork(int inputSize) {
    this->inputSize = inputSize;
}

MultiLayerNetwork::MultiLayerNetwork(vector<Layer *> layers) {
    this->inputSize = layers[0]->getSize();
    this->layers = layers;
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


void MultiLayerNetwork::train(double *inputs, double *labels, int size, int epochs, double alpha) {
    cout << "Start training model: size=" << size << ", epochs=" << epochs << ", alpha=" << alpha << endl;

    Matrix inputsMatrix(inputs, size, inputSize);

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

            if (_ % 1000 == 0)
                for (int j = 0; j < networkPredictions.size(); j++)
                    error += pow((expectedPredictions[j] - networkPredictions[j]), 2);

            backPropagation(expectedPredictions);
            updateWeights(inputsMatrix[i], alpha);
        }

        if (_ % 1000 == 0)
            cout << "> epochs " << _ << " - error=" << error << endl;
    }
}

vector<double> MultiLayerNetwork::predict(double *inputs) {
    Matrix matrix(inputs, this->inputSize, 1);

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
        cout << layer->getName() << " [" << layer->getSize() << " neurons, " << layer->getWeights()->getColumns()
             << " inputs, activation=" << layer->getActivation()->getName() << "]: " << endl
             << layer->getWeights()->toString() << endl;
    }
}

void MultiLayerNetwork::save(char *path) {
    FileWriter fileWriter(path);

    if (!fileWriter.isOpen())
        return;

    fileWriter.writeInt(layers.size());

    for (Layer *layer: layers) {
        fileWriter.writeString(layer->getName());
        fileWriter.writeInt(layer->getSize());
        fileWriter.writeString(layer->getActivation()->getName());
        fileWriter.writeMatrix(layer->getWeights());
    }

    fileWriter.close();
}

MultiLayerNetwork *MultiLayerNetwork::load(char *path) {
    FileReader fileReader(path);

    if (!fileReader.isOpen())
        return nullptr;

    int layersSize = fileReader.readInt();

    vector<Layer *> layers;

    for (int i = 0; i < layersSize; i++) {
        string name = fileReader.readString();
        int neurons = fileReader.readInt();
        string activationName = fileReader.readString();
        Matrix *weights = fileReader.readMatrix();

        auto *layer = new Layer(name, neurons, getActivation(activationName), nullptr);
        layer->initialize(weights);
        layers.push_back(layer);
    }

    return new MultiLayerNetwork(layers);
}

#include <cstring>
#include "network/MultiLayerNetwork.h"
#include "activation/implementation/Softmax.h"
#include "activation/implementation/Hyperbolic.h"
#include "activation/implementation/Relu.h"
#include "activation/implementation/Linear.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT

#endif

struct Parameters {
    ActivationFunction *activationFunction = new Sigmoid();
    Initializer *initializer = new RandomUniform(-1.0, 1.0);
};

Parameters parseParameters(char *input) {
    Parameters parameters;

    if (input != nullptr && strlen(input) > 0) {
        std::for_each(input, input + (strlen(input)), [](char &c) {
            c = ::tolower(c);
        });
        for (string s: split(input, ";")) {
            vector<string> pair = split(s, "=");
            if (pair.size() < 2) continue;
            string key = pair[0], value = pair[1];

            if (key.compare("activation") == 0) {
                parameters.activationFunction = MultiLayerNetwork::getActivation(value);
            } else if (key.compare("initializer") == 0) {
                parameters.initializer = MultiLayerNetwork::getInitializer(split(value, ","));
            }
        }
    }

    return parameters;
}

extern "C" {

EXPORT void *createModel(int inputSize) {
    return new MultiLayerNetwork(inputSize);
}

EXPORT void addLayer(MultiLayerNetwork *network, int size, char *params) {
    Parameters p = parseParameters(params);
    network->addLayer("unnamed-layer", size, p.activationFunction, p.initializer);
}

EXPORT void
trainModel(MultiLayerNetwork *network, double *inputs, double *labels, int size, int epochs, double alpha) {
    network->initialize();
    network->train(inputs, labels, size, epochs, alpha);
}

EXPORT void summary(MultiLayerNetwork *model) {
    model->dump();
}

EXPORT double *predict(MultiLayerNetwork *network, double *inputs) {
    return vectorToArray(network->predict(inputs));
}

EXPORT void deleteModel(MultiLayerNetwork *network) {
    delete network;
}
}

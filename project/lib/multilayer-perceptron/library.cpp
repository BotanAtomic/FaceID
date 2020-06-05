#include <cstring>
#include "network/MultiLayerNetwork.h"
#include "activation/implementation/Softmax.h"
#include "activation/implementation/Sigmoid.h"
#include "activation/implementation/Hyperbolic.h"
#include "activation/implementation/Relu.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT

#endif

ActivationFunction *getActivationByName(char *activation) {

    if (strcmp("sigmoid", activation) == 0)
        return new Sigmoid();
    else if (strcmp("tanh", activation) == 0)
        return new Hyperbolic();
    else if (strcmp("softmax", activation) == 0)
        return new Softmax();
    else if (strcmp("relu", activation) == 0)
        return new Relu();
    return new Sigmoid();
}

extern "C" {

EXPORT void *createModel(int inputSize) {
    return new MultiLayerNetwork(inputSize);
}

EXPORT void addLayer(MultiLayerNetwork *network, int size, char *activation) {
    network->addLayer("unnamed-layer", size, getActivationByName(activation));
}

EXPORT void
trainModel(MultiLayerNetwork *network, double *inputs, double *labels, int size, int epochs, double alpha) {
    network->initialize();
    network->dump();
    network->train(inputs, labels, size, epochs, alpha);
}

EXPORT double *predict(MultiLayerNetwork *network, double *inputs) {
    return vectorToArray(network->predict(inputs));
}


EXPORT void deleteModel(MultiLayerNetwork *network) {
    delete network;
}
}

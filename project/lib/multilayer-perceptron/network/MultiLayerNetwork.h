//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_MULTILAYERNETWORK_H
#define ML_TEST_MULTILAYERNETWORK_H

#include "../utils/utils.h"
#include "../layer/Layer.h"
#include "../activation/implementation/Sigmoid.h"

using namespace std;

class MultiLayerNetwork {

private:
    int inputSize;
    vector<Layer *> layers;

public:
    explicit MultiLayerNetwork(int inputSize);

    void addLayer(const string &name, int size);

    void addLayer(const string &name, int size, ActivationFunction *activationFunction);

    void initialize();

    vector<double> predict(double *inputs);

    void train(double *inputs, double *labels, int size, int epochs, double alpha);

    void backPropagation(const vector<double> &errors);

    void updateWeights(double *inputs, double alpha);

    void dump();
};


#endif //ML_TEST_MULTILAYERNETWORK_H

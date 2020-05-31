//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_MULTILAYERNETWORK_H
#define ML_TEST_MULTILAYERNETWORK_H

#include "layer/Layer.h"

using namespace std;

class MultiLayerNetwork {

private:
    long inputSize;
    int hiddenLayer;
    long outputSize;
    double learningRate;
    vector<Layer*> layers;

public:
    MultiLayerNetwork(long inputSize, int hiddenLayer, long hiddenLayerSize, long outputSize, double learningRate);

    void train(double * inputs, double * labels, int size, int epochs);
};


#endif //ML_TEST_MULTILAYERNETWORK_H

//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_LAYER_H
#define ML_TEST_LAYER_H

#include "../neuron/Neuron.h"
#include "../../matrix/Matrix.h"

using namespace std;

class Layer {

private:
    long size, previousSize;
    vector<Neuron *> neurons;

public:
    Layer(long size, long previousSize);

    long getSize();

    Matrix getWeights();

    Neuron * getNeurons(long i);

    void updateWeights(Matrix matrix);
};


#endif //ML_TEST_LAYER_H

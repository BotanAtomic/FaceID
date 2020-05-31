//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_NEURON_H
#define ML_TEST_NEURON_H

#include "random"

using namespace std;

class Neuron {

private:
    double bias;
    double output;
    double error;
    int size;
    vector<double> weights;

public:
    explicit Neuron(int size);

    int getSize() const;

    double getOutput() const;

    double getWeights(int index);

    double getError() const;

    void updateWeights(int index, double value);

    void updateBias(double value);
};


#endif //ML_TEST_NEURON_H

//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_LAYER_H
#define ML_TEST_LAYER_H


#include "../matrix/Matrix.h"
#include "vector"
#include "../activation/ActivationFunction.h"

using namespace std;

class Layer {

private:
    string name = "unnamed-layer";
    int neurons;
    Matrix * weights{};
    Matrix * outputs;
    Matrix * errors;
    ActivationFunction * activation;
public:
    explicit Layer(const string& name, int neurons, ActivationFunction * activation);

    void initialize(int weights);

    int getSize() const;

    string getName();

    Matrix * getWeights();

    Matrix * getOutputs();

    Matrix * getErrors();

    ActivationFunction * getActivation();

    void setOutputs(Matrix & o);

    void computeErrors(const vector<double>& currentErrors);

    void updateWeights(Matrix * correction);
};


#endif //ML_TEST_LAYER_H

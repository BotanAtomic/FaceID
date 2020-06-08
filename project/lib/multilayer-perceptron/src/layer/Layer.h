//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_LAYER_H
#define ML_TEST_LAYER_H


#include "../matrix/Matrix.h"
#include "vector"
#include "../activation/ActivationFunction.h"
#include "../initializer/Initializer.h"

using namespace std;

class Layer {

private:
    string name;
    int neurons;
    Matrix *weights;
    Matrix *outputs;
    Matrix *errors;
    Matrix *bias;
    ActivationFunction *activation;
    Initializer *initializer;
public:
    explicit Layer(const string &name, int neurons, ActivationFunction *activation, Initializer * initializer);

    void initialize(int weights);

    void initialize(Matrix * loadedWeights);

    int getSize() const;

    string getName();

    Matrix *getWeights();

    Matrix *getOutputs();

    Matrix *getErrors();

    Matrix *getBias();

    ActivationFunction *getActivation();

    void setOutputs(Matrix &o);

    void computeErrors(const vector<double> &currentErrors, bool useDerivation);

    void updateWeights(Matrix *correction);
};


#endif //ML_TEST_LAYER_H

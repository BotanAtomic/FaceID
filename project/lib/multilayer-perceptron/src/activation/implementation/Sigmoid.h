//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_SIGMOID_CPP
#define ML_TEST_SIGMOID_CPP

#include "../ActivationFunction.h"

class Sigmoid : public ActivationFunction {

    static double sigmoidFunction(double x) {
        return 1.0f / (1.0f + exp(-x));
    }

    void activate(Matrix & matrix) override {
        matrix.apply(sigmoidFunction);
    }

    double getDerivation(double x) override {
        return x * (1.0f - x);
    }

};


#endif //ML_TEST_SIGMOID_CPP

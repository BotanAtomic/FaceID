//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_SIGMOID_CPP
#define ML_TEST_SIGMOID_CPP

#include "../ActivationFunction.h"

class Sigmoid : public ActivationFunction {

    static double sigmoid(double x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static double sigmoid_(double x) {
        return x * (1.0f - x);
    }

    void activate(Matrix &matrix) override {
        matrix.apply(sigmoid);
    }

    virtual double getDerivation(double x) override {
        return x * (1.0f - x);
    }

};


#endif //ML_TEST_SIGMOID_CPP

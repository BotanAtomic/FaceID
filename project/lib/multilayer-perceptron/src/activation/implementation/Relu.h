//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_RELU_CPP
#define ML_TEST_RELU_CPP

#include "../ActivationFunction.h"

class Relu : public ActivationFunction {

    static double reluFunction(double x) {
        return x <= 0 ? 0 : x;
    }

    void activate(Matrix &matrix) override {
        matrix.apply(reluFunction);
    }

    double getDerivation(double x) override {
        return x > 0 ? 1 : 0;
    }

};


#endif //ML_TEST_RELU_CPP

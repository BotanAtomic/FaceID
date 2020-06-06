//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_LINEAR_CPP
#define ML_TEST_LINEAR_CPP

#include "../ActivationFunction.h"

class Linear : public ActivationFunction {

    static double linearFunction(double x) {
        return x;
    }

    string getName() override {
        return "linear";
    }

    void activate(Matrix &matrix) override {
        matrix.apply(linearFunction);
    }

    double getDerivation(double x) override {
        return 1;
    }

};


#endif //ML_TEST_LINEAR_CPP

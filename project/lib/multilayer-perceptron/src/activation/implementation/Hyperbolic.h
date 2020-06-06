//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_HYPERBOLIC_CPP
#define ML_TEST_HYPERBOLIC_CPP

#include "../ActivationFunction.h"

class Hyperbolic : public ActivationFunction {

    static double hyperbolicFunction(double x) {
        return tanh(x);
    }

    string getName() override {
        return "tanh";
    }

    void activate(Matrix &matrix) override {
        matrix.apply(hyperbolicFunction);
    }

    double getDerivation(double x) override {
        double th = tanh(x);
        return 1.0 - (th * th);
    }

};


#endif //ML_TEST_HYPERBOLIC_CPP

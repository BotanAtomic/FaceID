//
// Created by botan on 6/15/2020.
//

#ifndef ML_TEST_RADIALBASISFUNCTION_H
#define ML_TEST_RADIALBASISFUNCTION_H


#include "../Kernel.h"

class RadialBasisFunction : public Kernel {

private:
    double gamma;

public:

    explicit RadialBasisFunction(double gamma) {
        this->gamma = gamma;
    }

    double get(Matrix &x1, Matrix &x2) override {
        double norm = pow(-((x1 - x2).vectorNorm()), 2);
        double result = exp(-gamma * norm);
        return result;
    }

};


#endif //ML_TEST_RADIALBASISFUNCTION_H

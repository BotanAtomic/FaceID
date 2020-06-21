//
// Created by botan on 6/15/2020.
//

#ifndef ML_TEST_RBFKERNEL_H
#define ML_TEST_RBFKERNEL_H


#include "../Kernel.h"

class RBFKernel : public Kernel {

private:
    double gamma;

public:

    explicit RBFKernel(double gamma) {
        this->gamma = gamma;
    }

    Matrix build(Matrix &inputs) override {
        Matrix kernel(inputs.getRows(), inputs.getRows());
        for (int i = 0; i < kernel.getRows(); i++) {
            for (int j = 0; j < kernel.getColumns(); j++) {
                Matrix xi = inputs.sub(i);
                Matrix xn = inputs.sub(j);
                kernel.set(i, j, compute(xi, xn));
            }
        }
        return kernel;
    }

    double compute(Matrix &x1, Matrix &x2) override {
        double norm = pow(-(x1 - x2).vectorNorm(), 2);
        return exp(-gamma * norm);
    }


};


#endif //ML_TEST_RBFKERNEL_H

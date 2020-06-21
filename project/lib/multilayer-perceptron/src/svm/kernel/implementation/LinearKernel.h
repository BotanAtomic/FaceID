//
// Created by botan on 6/15/2020.
//

#ifndef ML_TEST_LINEARKERNEL_H
#define ML_TEST_LINEARKERNEL_H


#include "../Kernel.h"

class LinearKernel : public Kernel {

public:

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
        Matrix tr = x2.T();
        return x1.dot(tr).sum();
    }


};


#endif //ML_TEST_LINEARKERNEL_H

//
// Created by botan on 6/5/2020.
//

#ifndef ML_TEST_XAVIERUNIFORM_H
#define ML_TEST_XAVIERUNIFORM_H

#include "../Initializer.h"

class XavierUniform : public Initializer {

private:
    uniform_real_distribution<> distribution;
    mt19937 e2;
public:
    XavierUniform(int input, int output) {
        double r =  sqrt(6 / (input + output));
        random_device rd;
        this->distribution = uniform_real_distribution<>(-r, r);
        this->e2 = mt19937(rd());
    }

    void fill(Matrix &matrix) override {
        for (long i = 0; i < matrix.getColumns() * matrix.getRows(); i++)
            matrix.set(i, distribution(e2));
    }

};


#endif //ML_TEST_XAVIERUNIFORM_H

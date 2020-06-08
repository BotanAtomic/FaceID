//
// Created by botan on 6/5/2020.
//

#ifndef ML_TEST_XAVIERUNIFORM_H
#define ML_TEST_XAVIERUNIFORM_H

#include "../Initializer.h"

class XavierUniform : public Initializer {

private:
    int input, output;
public:
    XavierUniform(int input, int output) {
        this->input = input;
        this->output = output;
    }

    void fill(Matrix &matrix) override {
        for (long i = 0; i < matrix.getColumns() * matrix.getRows(); i++)
            matrix.set(i, sqrt(6 / (this->input + this->output)));
    }

};


#endif //ML_TEST_XAVIERUNIFORM_H

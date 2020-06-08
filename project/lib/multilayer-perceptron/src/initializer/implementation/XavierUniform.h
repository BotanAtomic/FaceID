//
// Created by botan on 6/5/2020.
//

#ifndef ML_TEST_XAVIERUNIFORM_H
#define ML_TEST_XAVIERUNIFORM_H

#include "../Initializer.h"

class XavierUniform : public Initializer {

private:
    RandomUniform *randomUniform = nullptr;
public:

    void fill(Matrix &matrix) override {
        if (randomUniform == nullptr) {
            double r = sqrt(6 / (matrix.getColumns() + matrix.getRows()));
            randomUniform = new RandomUniform(-r, r);
        }
        randomUniform->fill(matrix);
    }

};


#endif //ML_TEST_XAVIERUNIFORM_H

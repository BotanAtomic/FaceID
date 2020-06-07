//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_SOFTMAX_CPP
#define ML_TEST_SOFTMAX_CPP

#include <algorithm>
#include "../ActivationFunction.h"

class Softmax : public ActivationFunction {
    //todo: psq j ai rien capté la

    string getName() override {
        return "softmax";
    }

    void activate(Matrix & matrix) override {

    }

    double getDerivation(double x) override {
        return 0;
    }

};


#endif //ML_TEST_SOFTMAX_CPP

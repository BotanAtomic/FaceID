//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_SOFTMAX_CPP
#define ML_TEST_SOFTMAX_CPP

#include <algorithm>
#include "../ActivationFunction.h"

class Softmax : public ActivationFunction {

    string getName() override {
        return "softmax";
    }

    void activate(Matrix & matrix) override {
        std::vector<double> vector = matrix.toVector();

        double max = *std::max_element(vector.begin(), vector.end());

        double sum = 0.0f;
        for (double i : vector)
            sum += exp(i - max);

        double offset = max + log(sum);

        for (double &i : vector) {
            i = exp(i - offset);
        }
    }

    double getDerivation(double x) override {
        //todo: psq j ai rien capté la donc sigmoid pr l instant
        return x * (1.0f - x);
    }

};


#endif //ML_TEST_SOFTMAX_CPP

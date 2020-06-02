//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_SOFTMAX_CPP
#define ML_TEST_SOFTMAX_CPP

#include "../ActivationFunction.h"

class Softmax : public ActivationFunction {

    void activate(Matrix &matrix) override {
        std::vector<double> vector = matrix.toVector();

        double max = *std::max_element(vector.begin(), vector.end());

        double sum = 0.0f;
        for (double i : vector)
            sum += exp(i - max);

        double offset = max + log(sum);

        for (double &i : vector) {
            i = exp(i - offset);
        }
        matrix = Matrix(vector);
    }

    double getDerivation(double input) override {
        //todo: psq j ai rien capté la
        return input;
    }

};


#endif //ML_TEST_SOFTMAX_CPP

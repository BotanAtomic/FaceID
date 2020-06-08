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

        for(int i = 0; i < (matrix.getRows() * matrix.getColumns()); i++) {
            matrix.set(i, exp(matrix.get(i) - offset));
        }
    }

    double getDerivation(double x) override {
        return 0;
    }

};


#endif //ML_TEST_SOFTMAX_CPP

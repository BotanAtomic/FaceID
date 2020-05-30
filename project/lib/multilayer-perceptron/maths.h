//
// Created by botan on 5/29/2020.
//

#ifndef ML_TEST_MATHS_H
#define ML_TEST_MATHS_H

#include <cmath>
#include "multilayer-perceptron.h"

double sigmoid(double x) {
    return 1.0f / (1.0f + std::exp(-x));
}

double f_theta(double x, Layer *layer) {
    double result = layer->b;
    for (int i = 0; i < layer->size; i++) {
        result += layer->V[i] * sigmoid(layer->c[i] + layer->W[i] * x);
    }
    return result;
}

#endif //ML_TEST_MATHS_H

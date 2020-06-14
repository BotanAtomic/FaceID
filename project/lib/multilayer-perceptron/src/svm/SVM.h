//
// Created by botan on 6/14/2020.
//

#ifndef ML_TEST_SVM_H
#define ML_TEST_SVM_H


#include "../matrix/Matrix.h"
#include <algorithm>
#include "unordered_map"

class SVM {

private:
    Matrix weights{0};
    double bias;
    int inputSize;

public:
    explicit SVM(int inputSize);

    void train(double * inputs, double * labels, int samples);

    double predict(double * inputs);

};


#endif //ML_TEST_SVM_H

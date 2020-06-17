//
// Created by botan on 6/14/2020.
//

#ifndef ML_TEST_SVM_H
#define ML_TEST_SVM_H


#include "../matrix/Matrix.h"
#include <algorithm>
#include "unordered_map"
#include "kernel/implementation/RadialBasisFunction.h"
#include "../alglib/stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../alglib/optimization.h"

using namespace alglib;

class SVM {

private:
    Matrix weights{0};
    double bias;
    int inputSize;
    Kernel *kernel;

public:
    SVM(int inputSize, Kernel *kernel = nullptr);

    void train(double *inputs, double *labels, int samples);

    double predict(double *inputs);

};


#endif //ML_TEST_SVM_H

//
// Created by botan on 6/14/2020.
//

#ifndef ML_TEST_SVM_H
#define ML_TEST_SVM_H


#include "../matrix/Matrix.h"
#include <algorithm>
#include "unordered_map"
#include "kernel/implementation/RBFKernel.h"
#include "../alglib/stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../alglib/optimization.h"

using namespace alglib;

class SVM {

private:
    double bias;
    vector<double> supportVectors;
    int inputSize;
    Kernel *kernel;
    double lambda = 0;
    Matrix * trainInputs;
    double * trainLabels = nullptr;
    double * lagrangians = nullptr;

public:
    explicit SVM(int inputSize, Kernel *kernel = nullptr);

    void train(double *inputs, double *labels, int samples);

    double predict(double *inputs);

};


#endif //ML_TEST_SVM_H

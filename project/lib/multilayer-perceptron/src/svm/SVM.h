//
// Created by botan on 6/14/2020.
//

#ifndef ML_TEST_SVM_H
#define ML_TEST_SVM_H


#include "../matrix/Matrix.h"
#include "kernel/implementation/LinearKernel.h"

class SVM {

private:
    int inputSize;

    double bias = 0;
    double lambda = 0;

    Matrix *trainInputs;
    vector<double> supportVectors;
    vector<double> trainLabels;
    vector<double> solution;

    Kernel *kernel;

public:
    explicit SVM(int inputSize, Kernel *kernel = nullptr);

    void train(double *inputs, double *labels, int samples);

    double predict(double *inputs);

};


#endif //ML_TEST_SVM_H

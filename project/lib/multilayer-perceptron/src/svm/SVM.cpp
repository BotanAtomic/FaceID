//
// Created by botan on 6/14/2020.
//

#include "SVM.h"
#include "solver/QuadraticSolver.h"

SVM::SVM(int inputSize, Kernel *kernel) {
    this->bias = 0;
    this->inputSize = inputSize;
    this->kernel = kernel;
    this->trainInputs = new Matrix(0, 0);
}

void SVM::train(double *inputs, double *labels, int samples) {
    if (this->kernel == nullptr)
        this->kernel = new LinearKernel();

    this->trainLabels = vector<double>(labels, labels + samples);
    this->trainInputs = new Matrix(inputs, samples, inputSize);


    Matrix kernelMatrix = kernel->build(*trainInputs);

    this->solution = solveQP(kernelMatrix, samples, labels);

    for (int i = 0; i < solution.size(); i++) {
        if (solution[i] > 0)
            supportVectors.push_back(i);
    }

    for (int i: supportVectors)
        for (int j:supportVectors)
            lambda += labels[i] * labels[j] * solution[i] * solution[j] * kernelMatrix[i][j];

    int n = supportVectors.front();
    bias = labels[n] * lambda;
    for (int i: supportVectors)
        bias -= solution[i] * labels[i] * kernelMatrix[n][i];

}

double SVM::predict(double *inputs) {
    double result = 0;
    Matrix inputMatrix(inputs, 1, this->inputSize);
    for (int i: supportVectors) {
        Matrix sub = trainInputs->sub(i);
        result += solution[i] * trainLabels[i] * kernel->compute(sub, inputMatrix);
    }
    result += bias;
    return result >= 0 ? 1 : -1;
}

//
// Created by botan on 6/14/2020.
//

#include "SVM.h"

SVM::SVM(int inputSize, Kernel *kernel) {
    this->bias = 0;
    this->inputSize = inputSize;
    this->kernel = kernel;
    this->trainInputs = new Matrix(0, 0);
}

void SVM::train(double *inputs, double *labels, int samples) {
    if (this->kernel == nullptr)
        this->kernel = new RBFKernel(0.01);

    this->trainLabels = labels;

    auto *inputMatrix = new Matrix(inputs, samples, inputSize);
    this->trainInputs = inputMatrix;

    Matrix K = kernel->build(*inputMatrix);

    Matrix L(samples, 1, 0);

    real_2d_array q;
    q.setcontent(K.getRows(), K.getColumns(), K.toVector().data());

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < samples; j++) {
            q[i][j] = labels[i] * labels[j] * K.get(i, j);
        }
    }

    real_1d_array l;
    l.setcontent(L.getRows(), L.toVector().data());

    real_1d_array a;

    real_2d_array c;
    real_1d_array lbnd;
    real_1d_array ubnd;
    c.setlength(2, samples + 1);
    lbnd.setlength(samples);
    ubnd.setlength(samples);

    double maxValue = numeric_limits<float>::max();
    for (int i = 0; i < samples; i++) {
        lbnd[i] = 0;
        ubnd[i] = maxValue;
        c[0][i] = labels[i]; // Y
        c[1][i] = 1;// a
    }
    c[1][samples] = 1;// >=
    c[0][samples] = 0; // 0

    minqpstate state;
    minqpreport rep;

    minqpcreate(samples, state);
    minqpsetquadraticterm(state, q);
    minqpsetlinearterm(state, l);
    minqpsetbc(state, lbnd, ubnd);
    minqpsetlc(state, c, "[0, 0]");


    minqpsetalgobleic(state, 0, 0, 0, 0);
    minqpoptimize(state);
    minqpresults(state, a, rep);

    this->lagrangians = a.getcontent();

    for (int i = 0; i < a.length(); i++) {
        if (a[i] > 0)
            supportVectors.push_back(i);
    }

    for (int i: supportVectors) {
        for (int j:supportVectors) {
            lambda += labels[i] * labels[j] * a[i] * a[j] * K.get(i, j);
        }
    }

    int n = supportVectors.front();
    bias = labels[n] * lambda;
    for (int i: supportVectors) {
        bias -= a[i] * labels[i] * K.get(n, i);
    }
}

double SVM::predict(double *inputs) {
    double result = 0;
    Matrix inputMatrix(inputs, 1, this->inputSize);
    for (int i: supportVectors) {
        Matrix sub = trainInputs->sub(i);
        result += lagrangians[i] * trainLabels[i] * kernel->compute(sub, inputMatrix);
    }
    result += bias;
    return result > 0 ? 1 : -1;
}

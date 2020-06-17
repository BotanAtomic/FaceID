//
// Created by botan on 6/14/2020.
//

#include "SVM.h"

SVM::SVM(int inputSize, Kernel *kernel) {
    this->weights = Matrix(inputSize, 1, 0.0);
    this->bias = 0;
    this->inputSize = inputSize;
    this->kernel = kernel;
}

void SVM::train(double *inputs, double *labels, int samples) {
    if (this->kernel == nullptr)
        this->kernel = new RadialBasisFunction(0.01);

    Matrix inputMatrix(inputs, samples, inputSize);
    Matrix labelMatrix(labels, 1, samples);

    Matrix K(samples, samples);

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < samples; j++) {
            double result = labels[i] * labels[j];
            Matrix xi = inputMatrix.sub(i);
            Matrix xn = inputMatrix.sub(j).T();
            K.set(i, j, xi.dot(xn).sum() * result);
        }
    }

    Matrix Q(samples, 1, -1);

    real_2d_array q;
    q.setcontent(K.getRows(), K.getColumns(), K.toVector().data());

    real_1d_array a;

    real_2d_array c;
    real_1d_array lbnd;
    real_1d_array ubnd;
    c.setlength(2, samples + 1);
    lbnd.setlength(samples);
    ubnd.setlength(samples);

    for (int i = 0; i < samples; i++) {
        lbnd[i] = 0;
        ubnd[i] = MAXFLOAT;
        c[0][i] = labels[i]; // Y
        c[1][i] = 1;// a
    }
    c[1][samples] = 1;// >=
    c[0][samples] = 0; // 0

    minqpstate state;
    minqpreport rep;

    minqpcreate(samples, state);
    minqpsetquadraticterm(state, q);
    minqpsetbc(state, lbnd, ubnd);
    minqpsetlc(state, c, "[0, 0]");


    minqpsetalgobleic(state, 0, 0, 0, 0);
    minqpoptimize(state);
    minqpresults(state, a, rep);

    vector<double> supportVector;
    for (int i = 0; i < samples; i++) {
        Matrix result = (inputMatrix.sub(i) * (a[i] * labels[i]))->T();

        weights.add(result);
        if (a[i] > 0)
            supportVector.push_back(a[i]);
    }

    int n = 0;
    for (int i = 0; i < samples; i++) if (a[i] > 0) n = i;

    bias = (1 / labels[n]);

    for (int i = 0; i < supportVector.size(); i++) {
        bias -= weights.toVector()[i] * inputMatrix[n][i];
    }
}

double SVM::predict(double *inputs) {
    Matrix inputMatrix(inputs, inputSize, 1);
    return this->weights.T().dot(inputMatrix).sum() + bias < 0 ? -1 : 1;
}

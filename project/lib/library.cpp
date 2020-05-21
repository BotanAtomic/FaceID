#include "library.h"

#include <iostream>
#include <algorithm>
#include "random"

using namespace std;

extern "C" {

__declspec(dllexport) double *createModel(int size) {
    double * W = new double[size + 1];

    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < size + 1; i++) {
        W[i] = dist(e2);
    }

    return W;
}

void printModel(double *model, int size) {
    cout << "Linear Model { ";
    for (int i = 0; i < size; i++) {
        if (i) {
            cout << ", ";
        }
        cout << model[i];
    }
    cout << " }\n";
}

__declspec(dllexport) void
trainModel(double *model, double *X, const double *Y, int exampleCount, int inputSize, int iteration, double alpha) {
    for (int i = 0; i < iteration; i++) {
        int k = rand() % exampleCount;
        double gxk = predictClassificationModel(model, X + (k * inputSize), inputSize);
        for (int j = 0; j < inputSize; j++) {
            model[j + 1] += alpha * (Y[k] - gxk) * (X[(inputSize * k) + j]);
        }
        model[0] += alpha * (Y[k] - gxk);
    }
}

__declspec(dllexport) double predictRegressionModel(const double *model, const double *inputs, int size) {
    double sum = model[0];
    for (int i = 0; i < size; i++) {
        sum += model[i + 1] * inputs[i];
    }
    return sum;
}

__declspec(dllexport) double predictClassificationModel(const double *model, const double *inputs, int size) {
    return predictRegressionModel(model, inputs, size) >= 0 ? 1.0 : -1.0;
}

__declspec(dllexport) void deleteModel(const double *model) {
    delete[] model;
}

}

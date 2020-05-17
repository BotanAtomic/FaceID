#include "library.h"

#include <iostream>
#include <algorithm>

using namespace std;

extern "C" {

float random(int min, int max) {
    return (float) min + (float) rand() / ((float) RAND_MAX / (float) (max - min));
}

__declspec(dllexport) double *createModel(int size, int seed) {
    srand(seed);

    auto W = new double[size + 1];

    for (int i = 0; i < size; i++) {
        W[i] = random(-1, 1);
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
trainModel(double *model, double *X[], double *Y, int *shape, int iteration, double alpha) {
    for (int i = 0; i < iteration; i++) {
        int k = (int) random(0, shape[0]);
        double gxk = predictClassificationModel(model, X[k], shape[1]);
        for (int j = 0; j < shape[1]; j++) {
            model[j + 1] += alpha * (Y[k] - gxk) * X[k][j];
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

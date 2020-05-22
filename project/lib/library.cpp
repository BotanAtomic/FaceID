#include <fstream>
#include <iostream>
#include <algorithm>
#include "random"

using namespace std;

extern "C" {

__declspec(dllexport) double *createModel(int size) {
    double *W = new double[size + 1];

    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < size + 1; i++) {
        W[i] = dist(e2);
    }

    return W;
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

__declspec(dllexport) void trainModel(double *model, double *X, const double *Y, int exampleCount,
                                      int inputSize, int iteration, double alpha) {
    for (int i = 0; i < iteration; i++) {
        int k = rand() % exampleCount;
        double gxk = predictClassificationModel(model, X + (k * inputSize), inputSize);
        for (int j = 0; j < inputSize; j++) {
            model[j + 1] += alpha * (Y[k] - gxk) * (X[(inputSize * k) + j]);
        }
        model[0] += alpha * (Y[k] - gxk);
    }
}


__declspec(dllexport) void deleteModel(const double *model) {
    delete[] model;
}

__declspec(dllexport) bool saveModel(const double *model, int size, char *path) {
    ofstream modelFile;
    modelFile.open(path, ofstream::binary | ofstream::ate);

    if (!modelFile.is_open())
        return false;


    modelFile.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (int i = 0; i < size + 1; i++) {
        double value = model[i];
        modelFile.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    modelFile.close();
    return true;
}

__declspec(dllexport) double *loadModel(char *path) {
    ifstream modelFile;
    modelFile.open(path, ofstream::binary);

    if (!modelFile.is_open())
        return nullptr;

    int size = 0;
    modelFile.read(reinterpret_cast<char *>(&(size)), sizeof(size));
    
    double * model = createModel(size);

    for (int i = 0; i < size + 1; i++) {
        modelFile.read((char *) (model + i), sizeof(double));
    }

    modelFile.close();

    return model;
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

}

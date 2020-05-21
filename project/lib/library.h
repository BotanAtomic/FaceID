#ifndef UNTITLED_LIBRARY_H
#define UNTITLED_LIBRARY_H

extern "C" {
__declspec(dllexport) double *createModel(int size);

__declspec(dllexport) void
trainModel(double *model, double *X, const double *Y, int exampleCount, int inputSize, int iteration, double alpha);

__declspec(dllexport) double predictRegressionModel(const double *model, const double *inputs, int size);

__declspec(dllexport) double predictClassificationModel(const double *model, const double *inputs, int size);

__declspec(dllexport) void deleteModel(const double *model);

}

#endif //UNTITLED_LIBRARY_H

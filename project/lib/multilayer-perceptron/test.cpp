//
// Created by botan on 5/4/2020.
//

#include <iostream>
#include "library.cpp"

using namespace std;

int main(int size, char **args) {
    int shape[2] = {3, 2};
    double values[3][2] = {{1, 1},
                           {2,  3},
                           {3,  3}};

    double Y[3] = {1, -1, -1};


    double *model = createModel(shape[1]);
    printModel(model, shape[1] + 1);

    auto *X = static_cast<double *>(malloc(1));
    int index = 0;
    for (int i = 0; i < shape[0]; i++) {
        double * v = values[i];
        for(int j = 0; j < shape[1]; j++) {
            X[index] = v[j];
            index++;
        }
    }

    trainModel(model, X, Y, shape[0], shape[1], 100, 0.1);

    int correct = 0;
    for (int i = 0; i < shape[0]; i++) {
        double prediction = predictClassificationModel(model, X + i, shape[1]);
        std::cout << "[" << (X + i)[0] << ", " << (X + i)[1] << "] ->  predicted: " << prediction << " | value: "
                  << Y[i]
                  << "\n";
        if ((int) prediction == (int) Y[i]) {
            correct++;
        }
    }


    std::cout << "Accuracy: " << ((float) correct / (float) shape[0] * 100) << "%";

    deleteModel(model);
}

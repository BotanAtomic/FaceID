//
// Created by botan on 5/4/2020.
//

#include <iostream>
#include "library.h"

using namespace std;

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

int main(int size, char **args) {
    int seed = 1;

    int shape[2] = {8, 2};
    double values[8][2] = {{0.15, 0.2},
                           {0.2,  0.3},
                           {0.3,  0.4},
                           {0.5,  0.1},
                           {0.4,  0.7},
                           {0.5,  0.6},
                           {0.55, 0.4},
                           {0.8,  0.2}};

    double Y[8] = {-1, -1, -1, -1, 1, 1, 1, 1};


    double *model = createModel(shape[1], seed);
    printModel(model, shape[1] + 1);

    double *X[shape[0]];
    for (int i = 0; i < shape[0]; i++)
        X[i] = values[i];

    trainModel(model, X, Y, shape, 100, 0.01);

    int correct = 0;
    for (int i = 0; i < shape[0]; i++) {
        double prediction = predictClassificationModel(model, X[i], shape[1]);
        std::cout << "[" << X[i][0] << ", " << X[i][1] << "] ->  predicted: " << prediction << " | value: " << Y[i]
                  << "\n";
        if ((int) prediction == (int) Y[i]) {
            correct++;
        }
    }


    std::cout << "Accuracy: " <<( (float) correct / (float) shape[0] * 100) << "%";
}
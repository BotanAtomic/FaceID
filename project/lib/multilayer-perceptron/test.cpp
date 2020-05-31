//
// Created by botan on 5/30/2020.
//

#include <iostream>
#include "network/MultiLayerNetwork.h"

using namespace std;

int main(int size, char **args) {
    int epochs = 2000;
    int inputSize = 2;
    int outputSize = 1;
    int hiddenLayer = 1;
    int hiddenLayerSize = 2;

    double inputs[] = {0, 0,
                       0, 1,
                       1, 0,
                       1, 1};

    double labels[] = {0, 1, 1, 0};

    MultiLayerNetwork network(inputSize, hiddenLayer, hiddenLayerSize, outputSize, 0.1);

    network.train(inputs, labels, 4, 20000);
}

//
// Created by botan on 5/30/2020.
//


#include "network/MultiLayerNetwork.h"
#include "activation/implementation/Softmax.h"

using namespace std;

int main(int size, char **args) {
    int epochs = 20000;
    double alpha = 0.05;
    int inputSize = 3;
    int outputSize = 2;

    double test1[] = {0, 0, 1,
                      0, 1, 1,
                      1, 0, 1,
                      0, 1, 0,
                      1, 0, 0,
                      1, 1, 1,
                      0, 0, 0};

    double test1_label[] = {0, 1, 1, 1, 1, 0, 0};

    double test2[] = {1, 1,
                      1, 0,
                      0, 1,
                      0, 0};
    double test2_label[] = {1, 0, 0, 0};

    MultiLayerNetwork network(inputSize);
    network.addLayer("hidden-layer1", 4);
    network.addLayer("output-layer", outputSize);

    network.initialize();

    network.dump();
    network.train(test1, test1_label, 7, epochs, alpha);
    network.dump();

    cout << "Prediction: " << vectorToString(network.predict(new double[3]{1, 1, 0}));

}

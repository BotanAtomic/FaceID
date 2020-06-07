//
// Created by botan on 5/30/2020.
//


#include "network/MultiLayerNetwork.h"
#include "activation/implementation/Hyperbolic.h"
#include "activation/implementation/Softmax.h"
#include "activation/implementation/Relu.h"
#include <chrono>

using namespace std;
using namespace std::chrono;


int main(int size, char **args) {

    cout << "Hello, World" << endl;
    double * test2 = new double[6912 * 100];

    for(int i = 0; i < 691200; i++) {
        test2[i] = 0.546;
    }

    double test2_label[6912];

    for(int i = 0; i < 6912; i++) {
        test2_label[i] = i % 3;
    }

    int epochs = 5;
    double alpha = 0.1;
    int inputSize = 6912;
    int outputSize = 3;


    string modelPath = "C:\\Users\\botan\\Work\\FaceID\\project\\lib\\multilayer-perceptron\\model.mlp";

    MultiLayerNetwork *network = nullptr;

    bool loaded = false;
    if (ifstream(modelPath).good()) {
        network = MultiLayerNetwork::load(modelPath);
        loaded = true;
    } else {
        network = new MultiLayerNetwork(inputSize);
    }

    network->dump();

    if (!loaded) {
        int64_t timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        network->addLayer("hidden-layer", inputSize + 1, new Hyperbolic(), new RandomUniform(0.0, 1.0));
        network->addLayer("output-layer", outputSize, new Relu(), new RandomUniform(0.0, 1.0));
        network->initialize();
        network->train(test2, test2_label, 100, epochs, alpha);
        int64_t total = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - timestamp;
        cout << total << " ms" << endl;
    }


    for (int i = 0; i < 2; i++) {
        double response = test2_label[i];
        vector<double> predictions = network->predict(new double[1]{test2[i]});
        cout << "response=" << response << " | prediction=" << vectorToString(predictions) << endl;
    }

    if (!loaded && false)
        network->save("C:\\Users\\botan\\Work\\FaceID\\project\\lib\\multilayer-perceptron\\model.mlp");

    network->dump();
}

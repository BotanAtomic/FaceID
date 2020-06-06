//
// Created by botan on 5/30/2020.
//


#include "network/MultiLayerNetwork.h"
#include "activation/implementation/Hyperbolic.h"
#include "activation/implementation/Softmax.h"
#include "activation/implementation/Relu.h"
#include <chrono>
#include "stdio.h"

using namespace std;


int main(int size, char **args) {


    double test2[] = {1, 2};
    double test2_label[] = {2, 3};

    int epochs = 5;
    double alpha = 0.1;
    int inputSize = 1;
    int outputSize = 1;


    string modelPath = "C:\\Users\\botan\\Work\\FaceID\\project\\lib\\multilayer-perceptron\\model.mlp";

    MultiLayerNetwork *network = nullptr;

    bool loaded = false;
    if (ifstream(modelPath).good()) {
        network = MultiLayerNetwork::load(modelPath);
        loaded = true;
    } else {
        network = new MultiLayerNetwork(inputSize);
    }

    if (!loaded) {
        network->addLayer("output-layer", outputSize, new Relu(), new RandomUniform(0.0, 1.0));
        network->initialize();
    }
    network->dump();
    using namespace std::chrono;
    int64_t timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    network->train(test2, test2_label, 2, epochs, alpha);
    int64_t total = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - timestamp;

    cout << total << " ms" << endl;

    int correct = 0;
    for (int i = 0; i < 2; i++) {
        double response = test2_label[i];
        vector<double> predictions = network->predict(new double[1]{test2[i]});

        cout << "response=" << response << " | prediction=" << vectorToString(predictions) << endl;
    }

    cout << "correct = " << correct << endl;

    if (!loaded)
        network->save("C:\\Users\\botan\\Work\\FaceID\\project\\lib\\multilayer-perceptron\\model.mlp");
    network->dump();
}

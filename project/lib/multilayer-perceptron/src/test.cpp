//
// Created by botan on 5/30/2020.
//


#include "network/MultiLayerNetwork.h"
#include "activation/implementation/Hyperbolic.h"
#include "activation/implementation/Softmax.h"
#include "activation/implementation/Relu.h"
#include "initializer/implementation/XavierUniform.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(int size, char **args) {
    double test2[] = {1, 1,
                      0, 0,
                      0, 1,
                      1, 0};
    double test2_label[] = {0, 0, 1, 1};

    int epochs = 20000;
    double alpha = 0.01;
    int inputSize = 2;
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
        int64_t timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        network->addLayer("hidden-layer", 2, new Sigmoid(), new RandomUniform(0.0, 1.0));
        network->addLayer("output-layer", outputSize, new Sigmoid(), new XavierUniform());
        network->initialize();
        network->dump();
        network->train(test2, test2_label, 4, epochs, alpha);
        int64_t total = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - timestamp;
        cout << total << " ms" << endl;
    }


    for (int i = 0; i < 8; i += 2) {
        double response = test2_label[i / 2];
        vector<double> predictions = network->predict(new double[2]{test2[i], test2[i + 1]});
        cout << "response=" << response << " | prediction=" << vectorToString(predictions) << endl;
    }

    //if (!loaded)
        //network->save(modelPath);

}

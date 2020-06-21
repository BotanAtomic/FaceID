//
// Created by botan on 5/30/2020.
//


#include "network/MultiLayerNetwork.h"
#include "activation/implementation/Hyperbolic.h"
#include "activation/implementation/Softmax.h"
#include "activation/implementation/Relu.h"
#include "initializer/implementation/XavierUniform.h"
#include <chrono>
#include "backend/cuda/CudaBackend.h"
#include "svm/SVM.h"

using namespace std;
using namespace std::chrono;

void testMLP() {
    double test2[] = {1, 1,
                      0, 1,
                      1, 0,
                      0, 0};
    double test2_label[] = {-1, 1, 1, -1};

    int epochs = 20000;
    double alpha = 0.1;
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
        network->addLayer("hidden-layer", 5, new Hyperbolic(), new RandomUniform(-1, 1));
        network->addLayer("output-layer", outputSize, new Hyperbolic(), new RandomUniform(-1, 1));
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

    if (!loaded)
        network->save(modelPath);
}

void testSVM() {
    vector<double> test = vector<double>({
                                                 4, 2,
                                                 3, 3,
                                                 4, 4,
                                                 5, 3,

                                                 3.5, 3,
                                                 4.5, 3,
                                                 4, 3,
                                                 4, 3.5
                                         });
    vector<double> labels = vector<double>({-1, -1, -1, -1, 1, 1, 1, 1});


    SVM svm(2, new RBFKernel(0.01));

    svm.train(test.data(), labels.data(), labels.size());

    for (int i = 0; i < test.size(); i += 2) {
        double prediction = svm.predict(new double[2]{test[i], test[i + 1]});
        cout << "Prediction:" << prediction << " for " << test[i] << "," << test[i + 1] << endl;
    }

}

int main(int argc, char **argv) {
    testSVM();
}

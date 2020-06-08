//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_MULTILAYERNETWORK_H
#define ML_TEST_MULTILAYERNETWORK_H

#include "chrono"
#include "../utils/utils.h"
#include "../layer/Layer.h"
#include "../activation/implementation/Sigmoid.h"
#include "../activation/implementation/Hyperbolic.h"
#include "../activation/implementation/Linear.h"
#include "../activation/implementation/Relu.h"
#include "../activation/implementation/Softmax.h"

#include "../initializer/implementation/RandomUniform.h"
#include "../initializer/implementation/XavierUniform.h"

#include "../file/FileWriter.h"
#include "../file/FileReader.h"

using namespace std;

class MultiLayerNetwork {

public:
    static ActivationFunction *getActivation(const string &activation) {
        if (activation == "sigmoid")
            return new Sigmoid();
        else if (activation == "tanh")
            return new Hyperbolic();
        else if (activation == "softmax")
            return new Softmax();
        else if (activation == "relu")
            return new Relu();
        else if (activation == "linear")
            return new Linear();
        return new Sigmoid();
    }

    static Initializer *getInitializer(vector<string> params) {
        if (params[0] == "random_uniform") {
            if (params.size() < 3)
                cout << "Invalid parameters for initializer: required 3, got " << params.size() << endl;
            else
                return new RandomUniform(stod(params[1]), stod(params[2]));
        } else if(params[0] == "xavier" || params[0] == "xavier_uniform") {
            return new XavierUniform();
        }

        return new RandomUniform(-1.0, 1.0);
    }


private:
    int inputSize;
    vector<Layer *> layers;

public:
    explicit MultiLayerNetwork(int inputSize);

    explicit MultiLayerNetwork(int inputSize, vector<Layer *> layers);

    void addLayer(const string &name, int size, ActivationFunction *activationFunction, Initializer *initializer);

    void initialize();

    vector<double> predict(double *inputs);

    void train(double *inputs, double *labels, int size, int epochs, double alpha);

    void backPropagation(const vector<double> &errors);

    void updateWeights(double *inputs, double alpha);

    void dump();

    void save(const string &path);

    inline bool isClassification() const {
        return layers.back()->getSize() > 1;
    }

    static MultiLayerNetwork *load(const string &path);
};


#endif //ML_TEST_MULTILAYERNETWORK_H

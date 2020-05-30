//
// Created by botan on 5/29/2020.
//

#ifndef ML_TEST_MULTILAYER_PERCEPTRON_H
#define ML_TEST_MULTILAYER_PERCEPTRON_H

typedef struct Layer {
    double *c{};
    double *W{};
    double *V{};
    double b = 0;
    int size = 0;
} Layer;

typedef struct MultiLayerPerceptron {
    int inputNeurons;
    int outputNeurons;
    int hiddenLayers;
    Layer ** layers;
    double alpha;
} MultiLayerPerceptron;

#endif //ML_TEST_MULTILAYER_PERCEPTRON_H

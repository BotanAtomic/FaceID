//
// Created by botan on 6/14/2020.
//

#include "SVM.h"

SVM::SVM(int inputSize) {
    this->weights = Matrix(inputSize, 1, 0.0);
    this->bias = 0;
    this->inputSize = inputSize;
}

void SVM::train(double *inputs, double *labels, int samples) {
    Matrix inputMatrix(inputs, samples, inputSize);

    //minimiser 1/2 ||w||^2
    //contraintes: Yk( T(W) * Xk + b ) >= 1

    unordered_map<double, pair<Matrix, double>> supportVectors;

    Matrix transforms(vector<double>{
            1, 1,
            -1, 1,
            -1, -1,
            1, -1
    }.data(), 4, 2);

    double maximumValue = *max_element(inputs, inputs + (samples * inputSize));

    vector<double> stepSizes = {maximumValue * 0.1, maximumValue * 0.01, maximumValue * 0.001};
    double biasRangeMultiple = 5.0;
    double biasMultiple = 5.0;

    double latestOptimum = maximumValue * 10;

    int X = 0;
    for (double step: stepSizes) {
        Matrix currentWeights = Matrix(inputSize, 1, latestOptimum);
        bool optimized = false;

        while (!optimized) {
            double biasStep = step * biasMultiple;
            double currentBias = -1 * (maximumValue * biasRangeMultiple);
            double maximumBias = (maximumValue * biasRangeMultiple) - biasStep;
            while (currentBias <= maximumBias + biasStep) {
                for (int t = 0; t < transforms.getRows(); t++) {
                    Matrix transformation = transforms.sub(t).T();
                    Matrix tempWeights = currentWeights * transformation;
                    bool foundOption = true;

                    for (int i = 0; i < samples; i++) {
                        Matrix xi = inputMatrix.sub(i).T();
                        double yi = labels[i];
                        double r = ((tempWeights.T().dot(xi)).get(0) + currentBias) * yi;
                        if (r < 1) {
                            foundOption = false;
                        }
                    }

                    if (foundOption) {
                        supportVectors.insert(
                                std::make_pair(tempWeights.vectorNorm(), std::make_pair(tempWeights, currentBias)));
                    }
                }
                currentBias += biasStep;
            }

            if (currentWeights.get(0) < 0)
                optimized = true;
            else
                currentWeights + (-step);
        }

        double lowestNorm = numeric_limits<double>::max();
        pair<Matrix, double> bestSupport = supportVectors.end()->second;

        for (const auto &support : supportVectors) {
            if (support.first < lowestNorm) {
                lowestNorm = support.first;
                bestSupport = support.second;
            }
        }

        this->weights = bestSupport.first;
        this->bias = bestSupport.second;
        latestOptimum = this->weights.get(0) + (step * 2);
    }

    supportVectors.clear();
}

double SVM::predict(double *inputs) {
    Matrix inputMatrix(inputs, inputSize, 1);
    double res = inputMatrix.T().dot(this->weights).get(0) + bias;
    return res < 0 ? -1 : 1;
}

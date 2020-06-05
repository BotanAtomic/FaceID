//
// Created by botan on 6/5/2020.
//

#ifndef ML_TEST_RANDOMUNIFORM_H
#define ML_TEST_RANDOMUNIFORM_H

#include "../Initializer.h"

class RandomUniform : public Initializer {

private:
    uniform_real_distribution<> distribution;
    mt19937 e2;
public:
    RandomUniform(double min, double max) {
        random_device rd;
        this->e2 = mt19937(rd());
        this->distribution = uniform_real_distribution<>(min, max);
    }

    double get() override {
        return distribution(e2);
    }

};


#endif //ML_TEST_RANDOMUNIFORM_H

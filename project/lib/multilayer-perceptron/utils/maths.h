//
// Created by botan on 5/29/2020.
//

#ifndef ML_TEST_MATHS_H
#define ML_TEST_MATHS_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "random"

double sigmoid(double x) {
    return 1.0f / (1.0f + std::exp(-x));
}

double sigmoid_(double x) {
    return x * (1.0f - x) ;
}

#endif //ML_TEST_MATHS_H

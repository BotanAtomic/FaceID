//
// Created by botan on 6/1/2020.
//

#ifndef ML_TEST_UTILS_H
#define ML_TEST_UTILS_H

#include <cmath>
#include <string>
#include <sstream>
#include "vector"
#include "random"

using namespace std;

string vectorToString(vector<double> vec);

double *vectorToArray(vector<double> vec);

#endif //ML_TEST_UTILS_H

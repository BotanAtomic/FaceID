//
// Created by botan on 6/1/2020.
//

#ifndef ML_TEST_UTILS_H
#define ML_TEST_UTILS_H

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include "fstream"

using namespace std;

string vectorToString(vector<double> vec);

double *vectorToArray(vector<double> vec);

vector<string> split(const std::string& string, const std::string& delimiter);

#endif //ML_TEST_UTILS_H

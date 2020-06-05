//
// Created by botan on 6/1/2020.
//

#include "utils.h"

string vectorToString(vector<double> vec) {
    std::ostringstream vts;
    vts << "[";
    for (int i = 0; i < vec.size(); i++) {
        vts << vec[i];
        if (i < vec.size() - 1)
            vts << ", ";
    }
    vts << "]";
    return vts.str();
}

double *vectorToArray(vector<double> vec) {
    auto *array = new double[vec.size()];
    for (int i = 0; i < vec.size(); i++) {
        array[i] = vec[i];
    }
    return array;
}

vector<std::string> split(string str, string delimiter) {
    std::vector<std::string> tokens;
    char *str_c = strdup(str.c_str());
    char *token = nullptr;

    token = strtok(str_c, delimiter.c_str());
    while (token != nullptr) {
        tokens.emplace_back(token);
        token = strtok(nullptr, delimiter.c_str());
    }

    delete[] str_c;

    return tokens;
}

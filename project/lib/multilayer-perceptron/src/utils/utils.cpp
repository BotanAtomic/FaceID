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

vector<string> split(const string& str, const string& delimiter) {
    std::vector<string> tokens;
    char *str_c = _strdup(str.c_str());
    char *token = nullptr;

    char * nextToken;
    token = strtok_s(str_c, delimiter.c_str(), &nextToken);
    while (token != nullptr) {
        tokens.emplace_back(token);
        token = strtok_s(nullptr, delimiter.c_str(), &nextToken);
    }

    delete[] str_c;

    return tokens;
}

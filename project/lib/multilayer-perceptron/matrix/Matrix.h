//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_MATRIX_H
#define ML_TEST_MATRIX_H

#include "iostream"
#include "string"
#include "../utils/utils.h"
#include <sstream>
#include <vector>

using namespace std;

class Matrix {

private:

    int rows, columns;
    vector<double> data;

public:

    Matrix(int rows, int columns);

    Matrix(double * inputs, int inputSize, int size);

    explicit Matrix(const vector<double>& inputs);

    double * operator[](int i);

    Matrix dot(Matrix other);

    Matrix operator*(double number);

    Matrix operator-(Matrix other);

    Matrix T();

    Matrix apply(double transformer(double));

    double get(int i, int j);

    double *get(int i);

    void set(int i, int j, double value);

    void set(int i, double value);

    void add(Matrix other);

    int getRows() const;

    int getColumns() const;

    void dump(string name);

    string toString();

    vector<double> toVector();
};

#endif //ML_TEST_MATRIX_H

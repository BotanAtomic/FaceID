//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_MATRIX_H
#define ML_TEST_MATRIX_H

#include <iostream>
#include "string"
#include <sstream>
#include <vector>
#include "../utils/utils.h"

using namespace std;

class Matrix {

private:

    int rows, columns;
    vector<double> data;

public:

    explicit Matrix(int size);

    Matrix(int rows, int columns, double defaultValue = -1.0);

    Matrix(double *inputs, int rows, int columns);

    Matrix(Matrix * matrix, int inputSize, int size);

    explicit Matrix(const vector<double> &inputs);

    double *operator[](int i);

    Matrix operator*(Matrix & other);

    Matrix dot(Matrix &other);

    Matrix *operator*(double number);

    Matrix *operator+(double number);

    Matrix T();

    Matrix apply(double transformer(double));

    Matrix sub(int row);

    double get(int i, int j);

    double get(int i);

    void set(int i, int j, double value);

    void set(int i, double value);

    void add(Matrix other);

    int getRows() const;

    int getColumns() const;

    double sum();

    double vectorNorm();

    void dump(const string &name);

    string toString();

    vector<double> &toVector();
};

#endif //ML_TEST_MATRIX_H

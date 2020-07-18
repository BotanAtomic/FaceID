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

    static Matrix diag(int size, double value = 1.0);

    explicit Matrix(int size);

    Matrix(int rows, int columns, double defaultValue);

    Matrix(int rows, int columns);

    Matrix(double *inputs, int rows, int columns);

    Matrix(Matrix *matrix, int inputSize, int size);

    explicit Matrix(const vector<double> &inputs);

    double *operator[](int i);

    Matrix operator*(Matrix &other);

    Matrix dot(Matrix &other);

    Matrix *operator*(double number);

    Matrix *operator+(double number);

    Matrix operator-(Matrix &other);

    Matrix T();

    Matrix apply(double transformer(double));

    Matrix sub(int row);

    Matrix outer(Matrix &other);

    double get(int i, int j);

    double get(int i);

    void set(int i, int j, double value);

    void set(int i, double value);

    void add(Matrix &other);

    int getRows() const;

    int getColumns() const;

    double sum();

    double vectorNorm();

    void dump(const string &name);

    bool isVector() const;

    string toString();

    vector<double> &toVector();
};

#endif //ML_TEST_MATRIX_H

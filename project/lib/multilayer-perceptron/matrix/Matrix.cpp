//
// Created by botan on 6/2/2020.
//

#include "Matrix.h"

#include <utility>

Matrix::Matrix(int rows, int columns) {
    this->rows = rows;
    this->columns = columns;
    this->data.assign(rows * columns, 0.0f);
}

Matrix::Matrix(double *inputs, int inputSize, int size) {
    this->rows = size;
    this->columns = inputSize;
    for (int i = 0; i < rows * rows; i++) {
        this->data.push_back(inputs[i]);
    }
}

Matrix::Matrix(const vector<double> &inputs) {
    this->rows = inputs.size();
    this->columns = 1;
    this->data = inputs;
}

double *Matrix::operator[](int i) {
    return get(i);
}

Matrix Matrix::T() {
    Matrix newMatrix(columns, rows);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            newMatrix[j][i] = get(i, j);

    return newMatrix;
}

Matrix Matrix::dot(Matrix other) {
    if (this->columns == other.rows) {
        Matrix product(this->rows, other.columns);
        for (int i = 0; i < this->rows; ++i)
            for (int j = 0; j < other.columns; ++j) {
                product[i][j] = 0;
                for (int k = 0; k < this->columns; ++k) {
                    product[i][j] += get(i, k) * other[k][j];
                }
            }
        return product;
    } else {
        printf("cannot multiply matrix(%d, %d) (%d, %d)\n", rows, columns, other.rows, other.columns);
        exit(-1);
    }
}

Matrix Matrix::operator*(double number) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            set(i, j, get(i, j) * number);
        }
    }
    return *this;
}

Matrix Matrix::operator-(Matrix other) {
    if (this->columns == other.columns && this->rows == other.rows) {
        Matrix newMatrix(this->rows, other.columns);
        for (int i = 0; i < this->rows; ++i)
            for (int j = 0; j < this->columns; ++j) {
                newMatrix[i][j] = get(i, j) - other[i][j];
            }
        return newMatrix;
    } else {
        printf("cannot compute (-) matrix (%d, %d) with (%d, %d)\n", rows, columns, other.rows, other.columns);
        exit(-1);
    }
}

Matrix Matrix::apply(double transformer(double)) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            this->set(i, j, transformer(get(i, j)));
        }
    }
    return *this;
}

double Matrix::get(int i, int j) {
    return data[i * columns + j];
}

double *Matrix::get(int i) {
    return &data[i * columns];
}

void Matrix::set(int i, int j, double value) {
    data[i * columns + j] = value;
}

void Matrix::set(int i, double value) {
    data[i] = value;
}

void Matrix::add(Matrix other) {
    if (this->columns == other.columns && this->rows == other.rows) {
        for (int i = 0; i < this->rows; ++i)
            for (int j = 0; j < this->columns; ++j) {
                set(i, j, get(i, j) + other[i][j]);
            }
    } else {
        printf("cannot compute (+) matrix (%d, %d) with (%d, %d)\n", rows, columns, other.rows, other.columns);
        exit(0);
    }
}

int Matrix::getRows() const { return rows; }

int Matrix::getColumns() const { return columns; }

void Matrix::dump(string name) {
    cout << "Matrix(" << rows << "," << columns << ") " << name << endl;
    cout << toString() << endl;
}

string Matrix::toString() {
    stringstream stream;
    for (int i = 0; i < rows && columns > 0; i++) {
        stream << "[";
        for (int j = 0; j < columns && rows > 0; ++j) {
            double value = get(i, j);
            if (j > 0)
                stream << ", ";

            char buffer[1024];
            sprintf(buffer, "%s%0.8f", value >= 0 ? " " : "", value);

            stream << string(buffer);
        }
        stream << "]" << endl;
    }
    return stream.str();
}

vector<double> Matrix::toVector() {
    vector<double> vector;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            vector.push_back(get(i, j));
        }
    }
    return vector;
}

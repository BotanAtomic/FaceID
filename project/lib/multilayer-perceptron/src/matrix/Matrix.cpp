//
// Created by botan on 6/2/2020.
//

#include "Matrix.h"

Matrix::Matrix(int size) {
    this->rows = 1;
    this->columns = size;
    this->data = vector<double>(size);
}


Matrix::Matrix(int rows, int columns, double defaultValue) {
    this->rows = rows;
    this->columns = columns;
    if (defaultValue > 0)
        this->data.assign(rows * columns, defaultValue);
    else
        this->data = vector<double>(rows * columns);
}

Matrix::Matrix(double *inputs, int rows, int columns) {
    this->rows = rows;
    this->columns = columns;
    this->data = vector<double>(inputs, inputs + (rows * columns));
}

Matrix::Matrix(Matrix *matrix, int inputSize, int size) {
    this->rows = size;
    this->columns = inputSize;
    this->data = matrix->data;
}

Matrix::Matrix(const vector<double> &inputs) {
    this->rows = inputs.size();
    this->columns = 1;
    this->data = inputs;
}

double *Matrix::operator[](int i) {
    return &data[i * columns];
}

Matrix Matrix::T() {
    return Matrix(this, rows, columns);
}

Matrix Matrix::dot(Matrix &other) {
    if (this->columns == other.rows) {
        Matrix product(this->rows, other.columns);
        for (int i = 0; i < this->rows; ++i)
            for (int j = 0; j < other.columns; ++j) {
                product.set(i, j, 0);
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

Matrix *Matrix::operator*(double number) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            set(i, j, get(i, j) * number);
        }
    }
    return this;
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

double Matrix::get(int i) {
    return data[i];
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

void Matrix::dump(const string &name) {
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

vector<double> &Matrix::toVector() {
    return data;
}

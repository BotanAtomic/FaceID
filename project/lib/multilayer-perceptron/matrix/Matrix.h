//
// Created by botan on 5/31/2020.
//

#ifndef ML_TEST_MATRIX_H
#define ML_TEST_MATRIX_H

class Matrix {

private:
    unsigned int rows, columns;
    mutable double *data;

public:

    Matrix(unsigned int rows, unsigned int cols) {
        this->rows = rows;
        this->columns = cols;
        this->data = new double[rows * cols];
    }

    Matrix(double *inputs, long inputSize, long size) {
        this->rows = size;
        this->columns = inputSize;
        this->data = inputs;
    }

    double get(int i, int j) {
        return (data + (columns * i))[j];
    }

    int size() const { return rows * columns; }

    int getRows() const { return rows; }

    int getColumns() const { return columns; }

    double *operator[](unsigned int row) const {
        return data + (columns * row);
    }

    Matrix operator*(Matrix other) {
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
            return *this;
        }
    }

    Matrix operator*(double number) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this->operator[](i)[j] *= number;
            }
        }
        return *this;
    }

    Matrix operator-(Matrix other) {
        if (this->columns == other.columns && this->rows == other.rows) {
            Matrix newMatrix(this->rows, other.columns);
            for (int i = 0; i < this->rows; ++i)
                for (int j = 0; j < this->columns; ++j) {
                    newMatrix[i][j] = get(i, j) - other[i][j];
                }
            return newMatrix;
        } else {
            return *this;
        }
    }

    void dump(string name) {
        cout << "Matrix(" << rows << "," << columns << ") [" << name << "]" << endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; ++j) {
                if (j > 0)
                    cout << " | ";
                cout << get(i, j);
            }
            cout << endl;
        }
        cout << "____________" << endl << endl;
    }

    void apply(double transformer(double)) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this->operator[](i)[j] = transformer(get(i, j));
            }
        }
    }

    Matrix transpose() {
        Matrix newMatrix(columns, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                newMatrix[j][i] = get(i, j);
            }
        }

        return newMatrix;
    }

    Matrix multiply(Matrix other) {
        if (this->columns == other.columns && this->rows == other.rows) {
            Matrix newMatrix(this->rows, other.columns);
            for (int i = 0; i < this->rows; ++i)
                for (int j = 0; j < this->columns; ++j) {
                    newMatrix[i][j] = get(i, j) * other[i][j];
                }
            return newMatrix;
        } else {
            return *this;
        }
    }
};

#endif //ML_TEST_MATRIX_H

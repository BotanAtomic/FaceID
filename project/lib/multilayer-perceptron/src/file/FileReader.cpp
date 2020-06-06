//
// Created by botan on 6/6/2020.
//

#include "FileReader.h"

FileReader::FileReader(const string& path) {
    stream.open(path, ofstream::binary);
}

bool FileReader::isOpen() {
    return stream.is_open();
}

int FileReader::readInt() {
    int value = 0;
    stream.read(reinterpret_cast<char *>(&(value)), sizeof(int));
    return value;
}

string FileReader::readString() {
    int size = readInt();
    char * value = new char[size + 1];
    stream.read(value, sizeof(char) * size);
    value[size] = 0;
    return value;
}

Matrix *FileReader::readMatrix() {
    int rows = readInt();
    int columns = readInt();
    auto *values = new double[rows * columns];
    stream.read(reinterpret_cast<char *>(values), sizeof(double) * (rows * columns));
    return new Matrix(values, rows, columns);
}

void FileReader::close() {
    stream.close();
}

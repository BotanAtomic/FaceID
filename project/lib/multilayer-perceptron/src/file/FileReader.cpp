//
// Created by botan on 6/6/2020.
//

#include "FileReader.h"

FileReader::FileReader(char *path) {
    stream.open(path, ofstream::binary);
}

bool FileReader::isOpen() {
    return stream.is_open();
}

int FileReader::readInt() {
    int value = 0;
    stream.read(reinterpret_cast<char *>(&(value)), sizeof(value));
    return value;
}

string FileReader::readString() {
    int size = readInt();
    string value;
    stream.read(reinterpret_cast<char *>(&(value)), size);
    return value;
}

Matrix *FileReader::readMatrix() {
    int rows = readInt();
    int columns = readInt();
    auto *values = new double[rows * columns];
    stream.read(reinterpret_cast<char *>(&(values)), rows * columns);
    return new Matrix(values, rows, columns);
}

void FileReader::close() {
    stream.close();
}

//
// Created by botan on 6/6/2020.
//

#include "FileWriter.h"

FileWriter::FileWriter(char *path) {
    this->stream.open(path, ofstream::binary | ofstream::ate);
}

bool FileWriter::isOpen() {
    return stream.is_open();
}

void FileWriter::writeInt(int value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void FileWriter::writeString(const string &value) {
    writeInt(value.length());
    stream.write(reinterpret_cast<const char *>(value.c_str()), sizeof(value.length()));
}

void FileWriter::writeMatrix(Matrix *value) {
    writeInt(value->getRows());
    writeInt(value->getColumns());
    vector<double> vector = value->toVector();
    stream.write((char *) &vector[0], vector.size() * sizeof(double));
}

void FileWriter::close() {
    stream.close();
}

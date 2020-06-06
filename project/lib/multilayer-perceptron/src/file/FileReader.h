//
// Created by botan on 6/6/2020.
//

#ifndef ML_TEST_FILEREADER_H
#define ML_TEST_FILEREADER_H


#include <string>
#include "../matrix/Matrix.h"
#include "fstream"

using namespace std;

class FileReader {

private:
    ifstream stream;

public:
    explicit FileReader(const string& path);

    bool isOpen();

    int readInt();

    string readString();

    Matrix * readMatrix();

    void close();
};


#endif //ML_TEST_FILEREADER_H

//
// Created by botan on 6/6/2020.
//

#ifndef ML_TEST_FILEWRITER_H
#define ML_TEST_FILEWRITER_H


#include "fstream"
#include "../matrix/Matrix.h"

using namespace std;

class FileWriter {
private:
    ofstream stream;
public:
    explicit FileWriter(char *path);

    bool isOpen();

    void writeInt(int value);

    void writeString(const string& value);

    void writeMatrix(Matrix *value);

    void close();
};


#endif //ML_TEST_FILEWRITER_H

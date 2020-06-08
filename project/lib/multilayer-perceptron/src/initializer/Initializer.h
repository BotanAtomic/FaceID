//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_INITIALIZER_H
#define ML_TEST_INITIALIZER_H


#include "../matrix/Matrix.h"

class Initializer {

public:

    virtual void fill(Matrix &matrix) = 0;

};

#endif //ML_TEST_INITIALIZER_H

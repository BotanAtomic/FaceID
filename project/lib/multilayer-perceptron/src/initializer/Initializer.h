//
// Created by botan on 6/2/2020.
//

#ifndef ML_TEST_INITIALIZER_H
#define ML_TEST_INITIALIZER_H


#include "../matrix/Matrix.h"
#include <functional> // bind

class Initializer {

public:

    virtual void fill(Matrix &input) = 0;

};

#endif //ML_TEST_INITIALIZER_H

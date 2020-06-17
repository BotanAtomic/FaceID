//
// Created by botan on 6/15/2020.
//

#ifndef ML_TEST_KERNEL_H
#define ML_TEST_KERNEL_H

#include "../../matrix/Matrix.h"

class Kernel {

public:
    virtual double get(Matrix &x1, Matrix &x2) = 0;

};

#endif //ML_TEST_KERNEL_H

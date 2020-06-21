//
// Created by botan on 6/21/2020.
//

#ifndef ML_TEST_QUADRATICSOLVER_H
#define ML_TEST_QUADRATICSOLVER_H

#include "../../matrix/Matrix.h"
#include "../../alglib/optimization.h"
#include "../../alglib/stdafx.h"

using namespace alglib;

vector<double> resolve(Matrix &kernel, int samples, const double *labels) {
    Matrix L(samples, 1, -1); //terme lineaire

    real_2d_array q;
    q.setcontent(kernel.getRows(), kernel.getColumns(), kernel.toVector().data()); //terme quadratic

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < samples; j++) {
            q[i][j] = labels[i] * labels[j] * kernel.get(i, j);
        }
    }

    real_1d_array l;
    l.setcontent(L.getRows(), L.toVector().data());

    real_1d_array a;

    real_2d_array c;
    real_1d_array lbnd;
    real_1d_array ubnd;
    c.setlength(2, samples + 1);
    lbnd.setlength(samples);
    ubnd.setlength(samples);

    double maxValue = numeric_limits<float>::max();
    for (int i = 0; i < samples; i++) {
        lbnd[i] = 0;
        ubnd[i] = maxValue;
        c[0][i] = labels[i]; // Y
        c[1][i] = 1;// a
    }
    c[1][samples] = 1;// >=
    c[0][samples] = 0; // 0

    minqpstate state;
    minqpreport rep;

    minqpcreate(samples, state);
    minqpsetquadraticterm(state, q);
    minqpsetlinearterm(state, l);
    minqpsetbc(state, lbnd, ubnd);
    minqpsetlc(state, c, "[0, 0]");


    minqpsetalgobleic(state, 0, 0, 0, 0);
    minqpoptimize(state);
    minqpresults(state, a, rep);

    return vector<double>(a.getcontent(), a.getcontent() + a.length());
}

#endif //ML_TEST_QUADRATICSOLVER_H

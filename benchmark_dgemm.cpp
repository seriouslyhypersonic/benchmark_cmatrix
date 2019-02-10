/*
 * Copyright (c) Nuno Alves de Sousa 2019
 *
 * Use, modification and distribution is subject to the Boost Software License,
 * Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 * Benchmark: dgemm using alpha * A^T * B + beta * C
 */

#include <iostream>

#include <celero/Celero.h>

#include <mkl.h>

#include <CMatrix.h>

#include <fixture.hpp>

CELERO_MAIN

const int numSamples = 30;
const int numIterations = 100;

namespace constants
{
    const double alpha = 1.0;
    const double beta = 2.0;
}

BASELINE_F(Dgemm, MKL, MKLFixture<3>, numSamples, numIterations)
{
    const std::size_t dim = this->matrixDim;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans
               ,dim ,dim, dim
               ,constants::alpha
               ,mats[0], dim
               ,mats[1], dim
               ,constants::beta
               ,mats[2], dim);
}

BENCHMARK_F(Dgemm, CMatrixAssign, CMatrixFixture<3>, numSamples, numIterations)
{
    mats[2] = constants::alpha * T(mats[0]) * T(mats[1]) +
        constants::beta * mats[2];
}
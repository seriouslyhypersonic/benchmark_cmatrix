/*
 * Copyright (c) Nuno Alves de Sousa 2019
 *
 * Use, modification and distribution is subject to the Boost Software License,
 * Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef FIXTURE_H
#define FIXTURE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <new>

#include <celero/Celero.h>

#include <mkl.h>

#include <CMatrix.h>

#include <random_vector.hpp>
#include <debug.hpp>

using MKLMatrix = double*;
using CMatrixLib::FMatrix;

namespace settings
{
constexpr int numTests = 35;
}

/**
 * Creates a set of matrix dimensions to create a semilog plot
 * @param numTests Total number of tests
 * @return A problemSpace (i.e. the set of matrix dimensions)
 */
inline std::vector<celero::TestFixture::ExperimentValue>
semilogProgression(int numTests)
{
    std::vector<celero::TestFixture::ExperimentValue> problemSpace;

    std::int64_t matrixDim = 1;
    for (int i = 1; i <= numTests; ++i) {
        auto orderMag = static_cast<std::int64_t>
        (std::floor(std::log10(matrixDim)));
        matrixDim += static_cast<std::int64_t>(std::pow(10, orderMag));

        // Adjust iterations of the problemSpace according to matrix dimensions
        static std::int64_t iterations;
        switch (matrixDim) {
            case      2: iterations = 100; break;
            case    100: iterations = 25;  break;
            case   1000: iterations = 5;   break;
            case 10'000: iterations = 3;   break;
        }
        problemSpace.push_back({matrixDim, iterations});
    }
    return problemSpace;
}

class MatrixFixture: public celero::TestFixture
{
public:
    MatrixFixture() = default;

    /// The problem space is a set of matrix dimensions
    std::vector<celero::TestFixture::ExperimentValue> getExperimentValues() const override
    {
        return semilogProgression(numTests);
    }

protected:
    // Helper setter
    void updateMatrixDim(std::int64_t newDim)
    {
        matrixDim = newDim;
        matrixSize = matrixDim * matrixDim;
    }

    // Helper generator according to current matrix dimensions
    std::vector<double> makeRandomMatrixData()
    {
        return makeRandomVector(matrixSize, dataMin, dataMax);
    }

    std::int64_t matrixDim;
    std::int64_t matrixSize;

private:
    static constexpr int numTests = settings::numTests;

    static constexpr double dataMin = 0.0;
    static constexpr double dataMax = 1.0;
};

/// A class that allocates and initializes n MKL matrices with random data
template<std::size_t numMatrices>
class MKLFixture: public MatrixFixture
{
public:
    MKLFixture(): mats(numMatrices, nullptr)
    { }

    /// Before each run, initialize matrices with random numbers
    void setUp(const celero::TestFixture::ExperimentValue& experimentValue) override
    {
        this->updateMatrixDim(experimentValue.Value);

        auto randomData = this->makeRandomMatrixData();

        // Allocate and initialize MKL matrices with random data
        for(auto& m : mats) {
            m = allocate_dmatrix(this->matrixSize);
            std::copy(randomData.begin(), randomData.end(), m);
            MKL_DEBUG(m, this->matrixDim, this->matrixDim);
        }
    }

    // Deallocate MKL matrices
    void tearDown() override
    {
        for (auto& m : mats) {
            mkl_free(m);
        }
    };

protected:
    std::vector<MKLMatrix> mats;

private:
    // Helper allocation function
    virtual double* allocate_dmatrix(std::int64_t matrixSize)
    {
        auto ptr =
             static_cast<double*>
             (mkl_malloc(matrixSize*sizeof(double), this->align));
        if (!ptr) {
            std::cerr << "error: cannot allocate matrices\n";
            throw std::bad_alloc{};
        }
        return ptr;
    }

    static constexpr int align = 64; // default alignment on 64-byte boundary
};

/// A class that allocates and initializes n FMatrix matrices with random data
template<std::size_t numMatrices>
class CMatrixFixture: public MatrixFixture
{
public:
    CMatrixFixture(): mats(numMatrices, FMatrix{})
    { }

    /// Before each run, initialize matrices with random numbers
    void setUp(const celero::TestFixture::ExperimentValue& experimentValue) override
    {
        this->updateMatrixDim(experimentValue.Value);

        auto randomData = this->makeRandomMatrixData();

        // Initialize matrices with random data
        for(auto& m : mats) {
            m.Initialize(this->matrixDim, this->matrixDim, randomData.data());
            CMATRIX_DEBUG(m);
        }
    }

protected:
    std::vector<FMatrix> mats;
};

#endif //FIXTURE_H

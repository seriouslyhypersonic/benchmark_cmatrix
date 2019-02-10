/*
 * Copyright (c) André Carvalho
 */
#pragma once

#ifdef _WIN32
#ifdef CMATRIX_EXPORTS
#define CMATRIXLIB_API __declspec(dllexport)
#else
#define CMATRIXLIB_API __declspec(dllimport)
#endif
#else
#ifdef CMATRIX_EXPORTS
#define CMATRIXLIB_API __attribute__ ((visibility ("default")))
#else
#define CMATRIXLIB_API
#endif
#endif

#include "CMatrix.h"
#include "CTensor.h"

namespace CMatrixLib
{
	class CMATRIXLIB_API FVector
	{
#ifndef __GNUC__
	public:
		_declspec(property(get = GetLength)) unsigned INT Length;					//!< Total length of the vector
#endif
	protected:
		unsigned INT _length;

		double* _data;
	public:
		FVector();
		FVector(unsigned INT Length);
		FVector(const double* Data, unsigned INT Length);
		FVector(unsigned INT Length, double val1, ...);
		FVector(const FVector& Orig);
		FVector(const FMatrix& Orig);
		FVector(const FTensor& Orig);

		~FVector();

		void Clear();

		void Initialize(unsigned INT Length);
		void Initialize(const double* Data, unsigned INT Length);
		void Initialize(unsigned INT Length, double val1, ...);

		/*!\brief Return the total length of the vector.
		*/
		unsigned INT GetLength() const;

		FMatrix ToFMatrix(bool Column = true) const;

	protected:

		void PrivateInitialize(const double* Data, unsigned INT Length);
		void PrivateCopy(const FVector& Orig);
		void PrivateCopy(const FMatrix& Orig);
		void PrivateCopy(const FTensor& Orig);

	public:
		FVector & operator=(const FVector& Orig);
		FVector & operator=(const FMatrix& Orig);
		FVector & operator=(const FTensor& Orig);

		double operator()(unsigned INT i) const;
		double& operator()(unsigned INT i);

		FVector operator-(double Val) const;

		FTensor operator*(const FTensor& B) const;
		FMatrix operator*(const FMatrix& B) const;
		double operator*(const FVector& B) const;

		FVector operator,(const FVector& B) const;


		operator const double*() const;

		operator FTensor() const;

		friend CMATRIXLIB_API double Sum(const FVector& vect);
	};
}


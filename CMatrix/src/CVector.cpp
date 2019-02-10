/*
 * Copyright (c) André Carvalho
 */
#include <stdafx.h>
#include <CVector.h>

#ifndef __GNUC__
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#ifdef _DEBUG
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif
#endif

//Standard function replacements
#ifdef __GNUC__
#define memcpy_s(Dest,SDest,Source,SSource) memcpy((Dest),(Source),(SDest))
#define fscanf_s fscanf
#define sscanf_s sscanf
#define sprintf_s(Dest,SDest,Frmt,...) sprintf((Dest),(Frmt),__VA_ARGS__)
#define swprintf_s swprintf
#define strncpy_s(Dest,SDest,Source,N) strncpy((Dest),(Source),(N))

#define max(x,y) (((x)>(y))?(x):(y))
#endif // __GNUC__
//

#define FREE(p) \
{\
	if((p) != 0)\
		free((p));\
	(p) = 0;\
}

namespace CMatrixLib
{
	FVector::FVector()
	{
		_length = 0;
		_data = 0;
	}
	FVector::FVector(unsigned INT Length)
	{
		PrivateInitialize(0, Length);
	}
	FVector::FVector(const double* Data, unsigned INT Length)
	{
		PrivateInitialize(Data, Length);
	}
	FVector::FVector(unsigned INT Length, double val1, ...)
	{
		if (Length == 0)
		{
			PrivateInitialize(0, 0);
			return;
		}

		double* data = (double*)malloc(Length * sizeof(double));

		data[0] = val1;
		va_list args;
		va_start(args, val1);
		for (unsigned int i = 1; i < Length; i++)
		{
			data[i] = (double)va_arg(args, double);
		}
		va_end(args);

		PrivateInitialize(data, Length);

		FREE(data);
	}
	FVector::FVector(const FVector& Orig)
	{
		PrivateCopy(Orig);
	}

	FVector::FVector(const FMatrix& Orig)
	{
		PrivateCopy(Orig);
	}
	FVector::FVector(const FTensor& Orig)
	{
		PrivateCopy(Orig);
	}

	FVector::~FVector()
	{
		Clear();
	}

	void FVector::Clear()
	{
		FREE(_data);
		_length = 0;
	}

	void FVector::Initialize(unsigned INT Length)
	{
		Clear();
		PrivateInitialize(0, Length);
	}
	void FVector::Initialize(const double* Data, unsigned INT Length)
	{
		Clear();
		PrivateInitialize(Data, Length);
	}
	void FVector::Initialize(unsigned INT Length, double val1, ...)
	{
		Clear();

		if (Length == 0)
		{
			PrivateInitialize(0, 0);
			return;
		}

		double* data = (double*)malloc(Length * sizeof(double));

		data[0] = val1;
		va_list args;
		va_start(args, val1);
		for (unsigned int i = 1; i < Length; i++)
		{
			data[i] = va_arg(args, double);
		}
		va_end(args);

		PrivateInitialize(data, Length);

		FREE(data);
	}

	/*!\brief Return the total length of the vector.
	*/
	unsigned INT FVector::GetLength() const
	{
		return _length;
	}

	FMatrix FVector::ToFMatrix(bool Column) const
	{
		if (Column)
		{
			return FMatrix(_length, 1, _data);
		}
		else
		{
			return FMatrix(1, _length, _data);
		}
	}

	void FVector::PrivateInitialize(const double* Data, unsigned INT Length)
	{
		if (Length == 0)
		{
			_length = 0;
			_data = 0;
			return;
		}

		_length = Length;
		_data = (double*)calloc((size_t)_length, sizeof(double));
		if (errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FVector: Not enough memory\r\n");
		}

		if (Data != 0)
		{
			memcpy_s(_data, _length * sizeof(double), Data, _length * sizeof(double));
		}
	}
	void FVector::PrivateCopy(const FVector& Orig)
	{
		_length = Orig._length;
		if (_length == 0)
		{
			_data = 0;
			return;
		}

		_data = (double*)malloc((size_t)_length * sizeof(double));
		if (errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FVector: Not enough memory\r\n");
		}

		if (Orig._data != 0)
		{
			memcpy_s(_data, _length * sizeof(double), Orig._data, _length * sizeof(double));
		}
	}

	void FVector::PrivateCopy(const FMatrix& Orig)
	{
		if (Orig.GetLength() == 0)
		{
			_length = 0;
			_data = 0;
			return;
		}

		if (Orig.GetNRows() > 1 && Orig.GetNColumns() > 1)
		{
			THROWERROR(L"FVector: The FMatrix matrix must be a column or row vector");
		}

		_length = Orig.GetLength();
		_data = (double*)malloc((size_t)_length * sizeof(double));
		if (errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FVector: Not enough memory\r\n");
		}

		if (Orig.ToDoublePtr() != 0)
		{
			memcpy_s(_data, _length * sizeof(double), Orig.ToDoublePtr(), _length * sizeof(double));
		}
	}
	void FVector::PrivateCopy(const FTensor& Orig)
	{
		if (Orig.GetLength() == 0)
		{
			_length = 0;
			_data = 0;
			return;
		}

		if (Orig.GetRank() == 1)
		{
			THROWERROR(L"FVector: The FTensor must be a rank-1 tensor");
		}

		_length = Orig.GetLength();
		_data = (double*)malloc((size_t)_length * sizeof(double));
		if (errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FVector: Not enough memory\r\n");
		}

		memcpy_s(_data, _length * sizeof(double), Orig.ToPtr(), _length * sizeof(double));
	}

	FVector & FVector::operator=(const FVector& Orig)
	{
		Clear();
		PrivateCopy(Orig);

		return *this;
	}

	FVector & FVector::operator=(const FMatrix& Orig)
	{
		Clear();
		PrivateCopy(Orig);

		return *this;
	}
	FVector & FVector::operator=(const FTensor& Orig)
	{
		Clear();
		PrivateCopy(Orig);

		return *this;
	}

	double FVector::operator()(unsigned INT i) const
	{
		if (i >= _length)
			THROWERROR(L"FVector::operator(): Index out of bounds.");

		return _data[i];
	}

	double& FVector::operator()(unsigned INT i)
	{
		if (i >= _length)
			THROWERROR(L"FVector::operator(): Index out of bounds.");

		return _data[i];
	}

	FVector FVector::operator-(double Val) const
	{
		FVector res(_length);
		for (unsigned int i = 0; i < _length; i++)
		{
			res._data[i] = _data[i] - Val;
		}

		return res;
	}

	FTensor FVector::operator*(const FTensor& B) const
	{
		if (_length != B.GetDim(0))
			THROWERROR(L"FVector::operator*: Vector and tensor must have compatible dimensions.");

		FTensor res;

		res = _data[0] * B.GetSubTensor(0, 0);
		for (unsigned int i = 1; i < _length; i++)
		{
			res = res + _data[i] * B.GetSubTensor(0, i);
		}

		return res;
	}

	FMatrix FVector::operator*(const FMatrix& B) const
	{
		return this->ToFMatrix(false)*B;
	}

	double FVector::operator*(const FVector& B) const
	{
		if (_length != B._length)
			THROWERROR(L"FVector::operator,: vectors must have the same length.");

		return cblas_ddot(_length, _data, 1, B._data, 1);
	}

	FVector FVector::operator,(const FVector& B) const
	{
		if (_length != B._length)
			THROWERROR(L"FVector::operator,: vectors must have the same length.");

		FVector res(_length);

		vdMul(_length, _data, B._data, res._data);

		return res;
	}

	FVector::operator const double*() const
	{
		return _data;
	}

	FVector::operator FTensor() const
	{
		return FTensor(_data, 1, _length);
	}

	double Sum(const FVector& vect)
	{
		double res = 0;
		for (unsigned int i = 0; i < vect._length; i++)
		{
			res += vect._data[i];
		}

		return res;
	}
}

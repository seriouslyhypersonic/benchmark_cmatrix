/*
 * Copyright (c) André Carvalho
 */
#include <stdafx.h>
#include <CTensor.h>

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

namespace CMatrixLib
{
#define FREE(p) \
{\
	if((p) != 0)\
		free((p));\
	(p) = 0;\
}

	FTensor::FTensor()
	{
		_rank = 0;
		_dims = 0;

		_length = 0;
		_data = 0;
	}
	FTensor::FTensor(unsigned int Rank, ...)
	{
		if (Rank == 0)
		{
			_rank = 0;
			_dims = 0;

			_length = 0;
			_data = 0;

			return;
		}

		unsigned INT* dims = (unsigned INT*)malloc(Rank * sizeof(unsigned INT));

		va_list args;
		va_start(args, Rank);
		for (unsigned int r = 0; r < Rank; r++)
		{
			dims[r] = va_arg(args, int);
		}
		va_end(args);

		PrivateInitialize(0, Rank, dims);

		FREE(dims);
	}
	FTensor::FTensor(CMatrix<unsigned INT> indices)
	{
		PrivateInitialize(0, (unsigned int)indices.GetLength(), indices.GetDataPtr());
	}

	FTensor::FTensor(const double* Data, unsigned int Rank, ...)
	{
		if (Rank == 0)
		{
			_rank = 0;
			_dims = 0;

			_length = 0;
			_data = 0;

			return;
		}

		unsigned INT* dims = (unsigned INT*)malloc(Rank * sizeof(unsigned INT));

		va_list args;
		va_start(args, Rank);
		for (unsigned int r = 0; r < Rank; r++)
		{
			dims[r] = (unsigned INT)va_arg(args, int);
		}
		va_end(args);

		PrivateInitialize(Data, Rank, dims);

		FREE(dims);
	}

	FTensor::FTensor(const double* data, CMatrix<unsigned INT> indices)
	{
		PrivateInitialize(data, (unsigned int)indices.GetLength(), indices.GetDataPtr());
	}

	FTensor::FTensor(const FTensor& Orig)
	{
		PrivateCopy(Orig);
	}

	FTensor::FTensor(const FMatrix& Orig)
	{
		unsigned INT dims[] = { Orig.GetNRows(), Orig.GetNColumns() };

		PrivateInitialize(Orig.ToDoublePtr(), 2, dims);

		if (Orig.GetType() == Diagonal)
			_type = TT_Diagonal;
	}

	FTensor::~FTensor()
	{
		Clear();
	}

	void FTensor::Clear()
	{
		_rank = 0;
		FREE(_dims);

		_length = 0;
		FREE(_data);

		_type = TT_None;
	}

	void FTensor::Initialize(unsigned int Rank, ...)
	{
		Clear();

		if (Rank == 0)
			return;

		unsigned INT* dims = (unsigned INT*)malloc(Rank * sizeof(unsigned INT));

		va_list args;
		va_start(args, Rank);
		for (unsigned int r = 0; r < Rank; r++)
		{
			dims[r] = va_arg(args, int);
		}
		va_end(args);

		PrivateInitialize(0, Rank, dims);

		FREE(dims);
	}
	void FTensor::Initialize(const double* Data, unsigned int Rank, ...)
	{
		Clear();

		if (Rank == 0)
			return;

		unsigned INT* dims = (unsigned INT*)malloc(Rank * sizeof(unsigned INT));

		va_list args;
		va_start(args, Rank);
		for (unsigned int r = 0; r < Rank; r++)
		{
			dims[r] = va_arg(args, int);
		}
		va_end(args);

		PrivateInitialize(Data, Rank, dims);

		FREE(dims);
	}

	unsigned INT FTensor::GetDim(unsigned int rank) const
	{
		if (rank > _rank)
			THROWERROR(L"GetDim: given rank is larger than the tensor rank\r\n");

		return _dims[rank];
	}

	CMatrix<unsigned INT> FTensor::GetDimensions() const
	{
		return CMatrix<unsigned INT>(_rank, 1, _dims);
	}

	unsigned int FTensor::GetRank() const
	{
		return _rank;
	}

	unsigned INT FTensor::GetLength() const
	{
		return _length;
	}

	FMatrix FTensor::GetSubMatrix(unsigned INT Dim3, ...) const
	{
		if (_rank == 0 || _length == 0)
			THROWERROR(L"GetSubMatrix: Cannot get a submatrix out of an empty tensor\r\n");

		unsigned INT* indices = (unsigned INT*)calloc(_rank, sizeof(unsigned INT));

		va_list args;

		indices[2] = Dim3;
		va_start(args, Dim3);
		for (unsigned int r = 3; r < _rank; r++)
		{
			indices[r] = va_arg(args, int);
		}
		va_end(args);

		unsigned INT Index = indices2Index(indices);

		double* pointer = &_data[indices2Index(indices)];

		FREE(indices);

		return FMatrix(_dims[0], _dims[1], pointer);
	}
	void FTensor::SetSubMatrix(const FMatrix& Mat, unsigned INT Dim3, ...)
	{
		if (_rank == 0 || _length == 0)
			THROWERROR(L"GetSubMatrix: Cannot set a submatrix out of an empty tensor\r\n");

		if (Mat.GetNRows() > _dims[0] || Mat.GetNColumns() > _dims[1])
			THROWERROR(L"SetSubMatrix: Given matrix does not fit the tensor\r\n");

		unsigned INT* indices = (unsigned INT*)calloc(_rank, sizeof(unsigned INT));

		va_list args;

		indices[2] = Dim3;
		va_start(args, Dim3);
		for (unsigned int r = 3; r < _rank; r++)
		{
			indices[r] = va_arg(args, int);
		}
		va_end(args);

		unsigned INT Index = indices2Index(indices);

		if (Index >= _length)
			THROWERROR(L"SetSubMatrix: index outside of tensor dimensions\r\n");

		double* pointer = &_data[indices2Index(indices)];

		FREE(indices);

		memcpy_s(pointer, Mat.GetNRows()*Mat.GetNColumns() * sizeof(double), Mat.ToDoublePtr(), Mat.GetNRows()*Mat.GetNColumns() * sizeof(double));
	}

	FTensor FTensor::GetSubTensor(unsigned int SelectIndex, unsigned int cycle) const
	{
		if (_rank == 0)
			return *this;

		CMatrix<unsigned INT> dims(_rank - 1, 1);
		unsigned int iter = 0;
		for (unsigned int i = 0; i < _rank; i++)
		{
			if (i == SelectIndex)
				continue;

			dims(iter) = _dims[i];

			iter++;
		}

		FTensor res(dims);

		double* data = _data;
		
		#pragma omp parallel for shared(SelectIndex, cycle,data,res)
		for (INT index = 0; index < (INT)_length; index++)
		{
			CMatrix<unsigned INT> indices = Index2indices(index), indicesRES(indices.GetLength() - 1, 1);
			if (indices(SelectIndex) == cycle)
			{
				unsigned int skip = 0;
				for (unsigned int i = 0; i < indices.GetLength(); i++)
				{
					if (i != SelectIndex)
					{
						indicesRES(skip++) = indices(i);
					}
				}


				unsigned INT IndexRes = res.indices2Index(indicesRES);
				res(IndexRes) = _data[index];
			}
		}

		return res;
	}

	RTensor FTensor::RefSubTensor(unsigned int SelectIndex, unsigned int cycle)
	{
		if (_rank == 0)
			return *this;

		CMatrix<unsigned INT> dims(_rank, 1);
		CMatrix<unsigned INT> start(_rank, 1);
		for (unsigned int i = 0; i < dims.GetLength(); i++)
		{
			if (i == SelectIndex)
			{
				dims(i) = 1;
				start(i) = cycle;
			}
			else
			{
				dims(i) = _dims[i];
				start(i) = 0;
			}
		}

		RTensor res(*this, dims, start);

		return res;
	}

	void FTensor::PrivateInitialize(const double* Data, unsigned int Rank, const unsigned INT *Dims, TensorType type)
	{
		_rank = Rank;

		if (_rank == 0)
		{
			_data = 0;
			_dims = 0;
			_length = 0;
			return;
		}

		_dims = (unsigned INT*)malloc(_rank * sizeof(unsigned INT));
		memcpy_s(_dims, _rank * sizeof(unsigned INT), Dims, _rank * sizeof(unsigned INT));
		if (errno == ENOMEM || _dims == 0)
		{
			THROWERROR(L"CTensor: Not enough memory\r\n");
		}

		_length = 1;
		for (unsigned int r = 0; r < _rank; r++)
			_length *= _dims[r];

		if (_length == 0)
		{
			_data = 0;
			return;
		}

		_data = (double*)calloc((size_t)_length, sizeof(double));
		if (errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"CTensor: Not enough memory\r\n");
		}

		if (Data != 0)
		{
			memcpy_s(_data, _length * sizeof(double), Data, _length * sizeof(double));
		}


		_type = type;
	}

	void FTensor::PrivateCopy(const FTensor& Orig)
	{
		_rank = Orig._rank;

		if (_rank == 0)
		{
			_data = 0;
			_dims = 0;
			return;
		}

		_dims = (unsigned INT*)malloc(_rank * sizeof(unsigned INT));
		memcpy_s(_dims, _rank * sizeof(unsigned INT), Orig._dims, _rank * sizeof(unsigned INT));
		if (errno == ENOMEM || _dims == 0)
		{
			THROWERROR(L"CTensor: Not enough memory\r\n");
		}

		_length = Orig._length;
		if (_length == 0)
		{
			_data = 0;
			return;
		}

		_data = (double*)calloc((size_t)_length, sizeof(double));
		if (errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"CTensor: Not enough memory\r\n");
		}

		if (Orig._data != 0)
		{
			memcpy_s(_data, _length * sizeof(double), Orig._data, _length * sizeof(double));
		}

		_type = Orig._type;
	}

	unsigned INT FTensor::indices2Index(unsigned INT* indices) const
	{
		unsigned INT index = indices[0];
		unsigned INT jump = 1;
		for (unsigned INT r = 1; r < _rank; r++)
		{
			jump *= _dims[r - 1];
			index += jump*indices[r];
		}

		return index;
	}
	unsigned INT FTensor::indices2Index(const CMatrix<unsigned INT>& indices) const
	{
		unsigned INT index = indices(0);
		unsigned INT jump = 1;
		for (unsigned INT r = 1; r < _rank; r++)
		{
			jump *= _dims[r - 1];
			index += jump*indices(r);
		}

		return index;
	}

	CMatrix<unsigned INT> FTensor::Index2indices(unsigned INT Index) const
	{
		CMatrix<unsigned INT> indices(_rank, 1);

		unsigned INT temp_Index = Index;
		for (unsigned int r = 0; r < _rank; r++)
		{
			indices(r) = temp_Index%_dims[r];

			temp_Index = (temp_Index - indices(r)) / _dims[r];
		}

		return indices;
	}

	TensorType FTensor::GetTensorType() const
	{
		return _type;
	}
	void FTensor::SetTensorType(TensorType Type)
	{
		_type = Type;
	}

	double& FTensor::operator() (unsigned INT Index)
	{
		if (Index >= _length)
			THROWERROR(L"operator(): index outside of tensor dimensions\r\n");

		if (_type == Diagonal)
		{
			CMatrix<unsigned INT> indices = Index2indices(Index);
			for (unsigned int i = 0; i < indices.GetLength(); i++)
			{
				if (indices(i) != indices(0))
				{
					_type = TT_General;
					break;
				}
			}
		}

		return _data[Index];
	}
	double FTensor::operator() (unsigned INT Index) const
	{
		if (Index >= _length)
			THROWERROR(L"operator(): index outside of tensor dimensions\r\n");

		return _data[Index];
	}


	double& FTensor::operator() (unsigned INT Dim1, unsigned INT Dim2, ...) 
	{
		CMatrix<unsigned INT> indices(_rank, 1);
		va_list args;

		indices(0) = Dim1;
		indices(1) = Dim2;
		va_start(args, Dim2);
		for (unsigned int r = 2; r < _rank; r++)
		{
			indices(r) = va_arg(args, int);
		}
		va_end(args);

		if (_type == TT_Diagonal)
		{

			for (unsigned int i = 0; i < indices.GetLength(); i++)
			{
				if (indices(i) != indices(0))
				{
					_type = TT_General;
					break;
				}
			}
		}

		return (*this)(indices2Index(indices));
	}
	double FTensor::operator() (unsigned INT Dim1, unsigned INT Dim2, ...) const
	{
		unsigned INT* indices = (unsigned INT*)malloc(_rank * sizeof(unsigned INT));

		va_list args;

		indices[0] = Dim1;
		indices[1] = Dim2;
		va_start(args, Dim2);
		for (unsigned int r = 2; r < _rank; r++)
		{
			indices[r] = (unsigned INT)va_arg(args, int);
		}
		va_end(args);

		double res = (*this)(indices2Index(indices));

		FREE(indices);

		return res;
	}

	double& FTensor::operator() (CMatrix<unsigned INT> indices) noexcept(false)
	{
		return (*this)(indices2Index(indices));
	}

	double FTensor::operator() (CMatrix<unsigned INT> indices) const noexcept(false)
	{
		return (*this)(indices2Index(indices));
	}

	FTensor FTensor::operator*(const FTensor& B) const
	{
		if (_dims[_rank - 1] != B._dims[0])
			THROWERROR(L"operator*: the last index of left tensor must be equal to the first index of the tensor on the right\r\n");

		if (_type == TT_Diagonal && B._type == TT_Diagonal)
		{
			CMatrix<unsigned INT> final_index(_rank - 1 + B._rank - 1, 1);
			for (unsigned int r = 0; r < final_index.GetLength(); r++)
				final_index(r) = _dims[0];

			FTensor C(final_index);

			CMatrix<unsigned INT> indA(_rank, 1), indB(B._rank, 1), indC(C._rank, 1);

			for (int k = 0; k < C.GetDim(0); k++)
			{
				for (unsigned int r = 0; r < _rank; r++)
					indA(r) = k;
				for (unsigned int r = 0; r < B._rank; r++)
					indB(r) = k;
				for (unsigned int r = 0; r < C._rank; r++)
					indC(r) = k;

				C(indC) = (*this)(indA)*B(indB);
			}

			C._type = TT_Diagonal;

			return C;
		}
		else if (_type == TT_Diagonal && B._type == TT_General)
		{
			CMatrix<unsigned INT> final_index(_rank - 1 + B._rank - 1, 1);
			for (unsigned int r = 0; r < final_index.GetLength(); r++)
				final_index(r) = _dims[0];

			FTensor C(final_index);

			CMatrix<unsigned INT> indA(_rank, 1), indB(B._rank, 1), indC(C._rank, 1);

			for (int k = 0; k < C.GetDim(0); k++)
			{
				for (unsigned INT IndexB = 0; IndexB < B._length; IndexB += B._dims[0] * B._dims[1])
				{
					for (unsigned int r = 0; r < _rank; r++)
						indA(r) = k;
					for (unsigned int r = 0; r < B._rank; r++)
						indB(r) = k;

					indC(0) = 0;
					indC(1) = 0;
					for (unsigned int r = 2; r < _rank; r++)
						indC(r) = k;

					CMatrix<unsigned INT> iB = B.Index2indices(IndexB);
					for (unsigned int r = _rank; r < final_index.GetLength(); r++)
						indC(r) = iB(r - _rank + 2);

					double Av = (*this)(indA);
					FMatrix BM(B._dims[0], B._dims[1], &B._data[IndexB]);

					FMatrix CM = Av*BM;

					unsigned INT IndexC = C.indices2Index(indC);
					double* pointer = &C._data[IndexC];

					memcpy_s(pointer, CM.GetLength() * sizeof(double), CM.ToDoublePtr(), CM.GetLength() * sizeof(double));
				}
			}

			return C;
		}
		else
		{

			CMatrix<unsigned int> permutA(_rank, 1);
			if (_rank >= 2)
			{
				permutA(0) = _rank - 2;
				permutA(1) = _rank - 1;
				for (unsigned int r = 2; r < _rank; r++)
					permutA(r) = r - 2;
			}
			else
			{
				permutA(0) = 0;
			}

			FTensor TA = T(*this, permutA.GetDataPtr());

			CMatrix<unsigned INT> final_index(_rank - 1 + B._rank - 1, 1);

			if (_rank >= 2)
			{
				final_index(0) = TA._dims[0];
				final_index(1) = B._dims[1];

				for (unsigned int r = 2; r < _rank; r++)
					final_index(r) = TA._dims[r];
				for (unsigned int r = _rank; r < final_index.GetLength(); r++)
					final_index(r) = B._dims[r - _rank + 2];
			}
			else
			{
				final_index(0) = B._dims[1];
				for (unsigned int r = 1; r < final_index.GetLength(); r++)
					final_index(r) = B._dims[r + 1];
			}


			FTensor C(final_index);

			for (unsigned INT IndexA = 0; IndexA < _length; IndexA += TA._dims[0] * TA._dims[1])
			{
				for (unsigned INT IndexB = 0; IndexB < B._length; IndexB += B._dims[0] * B._dims[1])
				{
					FMatrix AM((_rank >= 2) ? TA._dims[0] : 1, (_rank >= 2) ? TA._dims[1] : TA._dims[0], &TA._data[IndexA]);
					FMatrix BM(B._dims[0], B._dims[1], &B._data[IndexB]);

					CMatrix<unsigned INT> iTA = TA.Index2indices(IndexA);
					CMatrix<unsigned INT> iB = B.Index2indices(IndexB);

					if (AM.GetLength() == 1 && BM.GetLength() == 1)
					{
						CMatrix<unsigned INT> iC(final_index.GetLength(), 1);
						iC(0) = 0;
						iC(1) = 0;
						for (unsigned int r = 2; r < _rank; r++)
							iC(r) = iTA(r);
						for (unsigned int r = _rank; r < final_index.GetLength(); r++)
							iC(r) = iB(r - _rank + 2);

						unsigned INT IndexC = C.indices2Index(iC);

						C._data[IndexC] = AM(0)*BM(0);
					}
					else
					{
						FMatrix CM = AM * BM;

						CMatrix<unsigned INT> iC(final_index.GetLength(), 1);
						iC(0) = 0;
						iC(1) = 0;
						for (unsigned int r = 2; r < _rank; r++)
							iC(r) = iTA(r);
						for (unsigned int r = _rank; r < final_index.GetLength(); r++)
							iC(r) = iB(r - _rank + 2);

						unsigned INT IndexC = C.indices2Index(iC);
						double* pointer = &C._data[IndexC];

						memcpy_s(pointer, CM.GetLength() * sizeof(double), CM.ToDoublePtr(), CM.GetLength() * sizeof(double));
					}
				}
			}

			if (_rank >= 2)
			{
				CMatrix<unsigned int> permutC(C._rank, 1);
				for (unsigned int r = 0; r < _rank - 2; r++)
					permutC(r) = r + 2;
				permutC(_rank - 2) = 0;
				permutC(_rank - 1) = 1;
				for (unsigned int r = 0; r < B._rank - 2; r++)
					permutC(_rank + r) = _rank + r;

				return FTensor::T(C, permutC.GetDataPtr());
			}
			else
			{
				return C;
			}
		}
	}

	FTensor operator*(const FTensor::FTensorIndex& IA, const FTensor::FTensorIndex& IB)
	{
		const FTensor& A = IA._tensor;
		const FTensor& B = IB._tensor;

		if (A._type == TT_Diagonal && B._type == TT_Diagonal)
		{
			CMatrix<unsigned INT> TC_indices(A._rank - 1 + B._rank - 1, 1);
			for (unsigned int r = 0; r < TC_indices.GetLength(); r++)
				TC_indices(r) = A.GetDim(0);

			FTensor C(TC_indices);

			CMatrix<unsigned INT> indA(A._rank, 1), indB(B._rank, 1), indC(C._rank, 1);

			for (int k = 0; k < C.GetDim(0); k++)
			{
				for (unsigned int r = 0; r < A._rank; r++)
					indA(r) = k;
				for (unsigned int r = 0; r < B._rank; r++)
					indB(r) = k;
				for (unsigned int r = 0; r < C._rank; r++)
					indC(r) = k;

				C(indC) = A(indA)*B(indB);
			}

			C._type = TT_Diagonal;

			return C;
		}
		else if (A._type == TT_Diagonal && B._type == TT_General)
		{
			CMatrix<unsigned int> permutB(B._rank, 1);
			unsigned int indexB = IB._index;
			for (unsigned int r = 0; r < B._rank; r++)
			{
				permutB(r) = indexB;

				indexB++;
				if (indexB >= B._rank)
					indexB = 0;
			}
			FTensor TB = FTensor::T(B, permutB.GetDataPtr());

			CMatrix<unsigned INT> TC_indices(A._rank + TB._rank - 2, 1);

			TC_indices(0) = A._dims[0];
			TC_indices(1) = TB._dims[1];
			unsigned int rinc = 2;
			for (unsigned int r = 0; r < A._rank - 2; r++)
			{
				TC_indices(rinc) = A._dims[0];

				rinc++;
			}
			rinc = A._rank;
			for (unsigned int r = 2; r < B._rank; r++)
			{
				TC_indices(rinc) = B._dims[permutB(r)];
				rinc++;
			}

			FTensor TC(TC_indices);

			CMatrix<unsigned INT> indA(A._rank, 1), indC(TC._rank, 1);

			for (int k = 0; k < TC.GetDim(0); k++)
			{
				for (unsigned INT IndexB = 0; IndexB < B._length; IndexB += B._dims[0] * B._dims[1])
				{
					for (unsigned int r = 0; r < A._rank; r++)
						indA(r) = k;

					indC(0) = 0;
					indC(1) = 0;
					for (unsigned int r = 2; r < A._rank; r++)
						indC(r) = k;

					CMatrix<unsigned INT> iB = B.Index2indices(IndexB);
					for (unsigned int r = A._rank; r < TC_indices.GetLength(); r++)
						indC(r) = iB(r - A._rank + 2);

					double Av = A(indA);
					FMatrix BM(B._dims[0], B._dims[1], &B._data[IndexB]);

					FMatrix CM(TC.GetDim(0), BM.GetNColumns());
					
					for (int j = 0; j < BM.GetNColumns(); j++)
					{
						CM(k, j) = Av * BM(k, j);
					}

					unsigned INT IndexC = TC.indices2Index(indC);
					double* pointer = &TC._data[IndexC];

					memcpy_s(pointer, CM.GetLength() * sizeof(double), CM.ToDoublePtr(), CM.GetLength() * sizeof(double));
				}
			}

			CMatrix<unsigned int> permutTC_step1(TC._rank, 1), permutTC(TC._rank, 1);

			//Step 1
			rinc = 0;
			permutTC_step1(rinc++) = 0;
			for (unsigned int r = 2; r < A._rank; r++)
				permutTC_step1(rinc++) = r;
			permutTC_step1(rinc++) = 1;
			for (unsigned int r = 0; r < B._rank - 2; r++)
				permutTC_step1(rinc++) = A._rank + r;

			FTensor TC_step1 = FTensor::T(TC, permutTC_step1.GetDataPtr());
			//

			//Step2
			unsigned int indexA = (IA._index - 1 == -1) ? (A._rank - 2) : (IA._index - 1);
			rinc = indexA;
			for (unsigned int r = 0; r < A._rank - 1; r++)
			{
				permutTC(rinc++) = r;
				if (rinc >= A._rank - 1)
					rinc = 0;
			}
			indexB = IB._index;
			rinc = indexB;
			for (unsigned int r = 0; r < B._rank - 1; r++)
			{
				permutTC((A._rank - 1) + (rinc++)) = r + A._rank - 1;
				if (rinc >= B._rank - 1)
					rinc = 0;
			}
			//

			return FTensor::T(TC_step1, permutTC.GetDataPtr());
		}
		else
		{

			if (IA._index == IA._tensor._rank - 1 && IB._index == 0)
				return A * B;

			CMatrix<unsigned int>	permutA(A._rank, 1),
				permutB(B._rank, 1);

			unsigned int indexA = (IA._index - 1 == -1) ? (A._rank - 1) : (IA._index - 1);
			for (unsigned int r = 0; r < A._rank; r++)
			{
				permutA(r) = indexA;

				indexA++;
				if (indexA >= A._rank)
					indexA = 0;
			}

			unsigned int indexB = IB._index;
			for (unsigned int r = 0; r < B._rank; r++)
			{
				permutB(r) = indexB;

				indexB++;
				if (indexB >= B._rank)
					indexB = 0;
			}

			FTensor TA = FTensor::T(A, permutA.GetDataPtr()),
				TB = FTensor::T(B, permutB.GetDataPtr());

			CMatrix<unsigned INT> TC_indices(TA._rank + TB._rank - 2, 1);

			TC_indices(0) = TA._dims[0];
			TC_indices(1) = TB._dims[1];
			unsigned int rinc = 2;
			for (unsigned int r = 0; r < A._rank - 2; r++)
			{
				TC_indices(rinc) = A._dims[permutA(rinc)];

				rinc++;
			}
			rinc = A._rank;
			for (unsigned int r = 2; r < B._rank; r++)
			{
				TC_indices(rinc) = B._dims[permutB(r)];
				rinc++;
			}

			FTensor TC(TC_indices);

			for (unsigned INT IndexA = 0; IndexA < TA._length; IndexA += TA._dims[0] * TA._dims[1])
			{
				for (unsigned INT IndexB = 0; IndexB < TB._length; IndexB += TB._dims[0] * TB._dims[1])
				{
					FMatrix TAM(TA._dims[0], TA._dims[1], &TA._data[IndexA]);
					FMatrix TBM(TB._dims[0], TB._dims[1], &TB._data[IndexB]);

					FMatrix TCM = TAM * TBM;

					CMatrix<unsigned INT> iTA = TA.Index2indices(IndexA),
						iTB = TB.Index2indices(IndexB);

					CMatrix<unsigned INT> iTC(TC_indices.GetLength(), 1);
					iTC(0) = 0;
					iTC(1) = 0;
					for (unsigned int r = 2; r < TA._rank; r++)
						iTC(r) = iTA(r);
					for (unsigned int r = TA._rank; r < TC_indices.GetLength(); r++)
						iTC(r) = iTB(r - TA._rank + 2);

					unsigned INT IndexC = TC.indices2Index(iTC);
					double* pointer = &TC._data[IndexC];

					memcpy_s(pointer, TCM.GetLength() * sizeof(double), TCM.ToDoublePtr(), TCM.GetLength() * sizeof(double));
				}
			}

			CMatrix<unsigned int> permutTC_step1(TC._rank, 1), permutTC(TC._rank, 1);

			//Step 1
			rinc = 0;
			permutTC_step1(rinc++) = 0;
			for (unsigned int r = 2; r < A._rank; r++)
				permutTC_step1(rinc++) = r;
			permutTC_step1(rinc++) = 1;
			for (unsigned int r = 0; r < B._rank - 2; r++)
				permutTC_step1(rinc++) = A._rank + r;

			FTensor TC_step1 = FTensor::T(TC, permutTC_step1.GetDataPtr());
			//

			//Step2
			indexA = (IA._index - 1 == -1) ? (A._rank - 2) : (IA._index - 1);
			rinc = indexA;
			for (unsigned int r = 0; r < A._rank - 1; r++)
			{
				permutTC(rinc++) = r;
				if (rinc >= A._rank - 1)
					rinc = 0;
			}
			indexB = IB._index;
			rinc = indexB;
			for (unsigned int r = 0; r < B._rank - 1; r++)
			{
				permutTC((A._rank - 1) + (rinc++)) = r + A._rank - 1;
				if (rinc >= B._rank - 1)
					rinc = 0;
			}
			//

			return FTensor::T(TC_step1, permutTC.GetDataPtr());
		}
	}

	FTensor operator*(const FTensor::FTensorIndex& IA, const FTensor& B)
	{
		return IA*I(B, 0);
	}
	FTensor operator*(const FTensor& A, const FTensor::FTensorIndex& IB)
	{
		return I(A, A._rank - 1)*IB;
	}

	FTensor operator*(const FMatrix& A, const FTensor::FTensorIndex& IB)
	{
		return I(FTensor(A), 1)*IB;
	}
	FTensor operator*(const FTensor::FTensorIndex& IA, const FMatrix& B)
	{
		return IA*I(FTensor(B), 0);
	}

	FTensor operator*(const FMatrix& A, const FTensor& B)
	{
		return A*I(B, 0);
	}
	FTensor operator*(const FTensor& A, const FMatrix& B)
	{
		return I(A, A._rank - 1)*B;
	}

	FTensor FTensor::operator*(const double& b) const
	{
		if (_length == 0)
			return *this;

		FTensor res(*this);

		cblas_dscal(res._length, b, res._data, 1);

		return res;
	}
	FTensor operator*(const double& a, const FTensor B)
	{
		return B*a;
	}

	FTensor FTensor::operator^(const FTensor& B) const
	{
		const FTensor& A = *this;
		const unsigned int rank_A = _rank;
		const unsigned int rank_B = B.GetRank();

		CMatrix<unsigned INT> dims(rank_A + rank_B, 1);
		for (unsigned int i = 0; i < dims.GetLength(); i++)
		{
			if (i < rank_A)
			{
				dims(i) = GetDim(i);
			}
			else
			{
				dims(i) = B.GetDim(i - rank_A);
			}
		}
		FTensor C(dims);

		#pragma omp parallel shared(A,B)
		{
			#pragma omp for
			for (INT indexC = 0; indexC < (INT)C.GetLength(); indexC++)
			{
				CMatrix<unsigned INT> indicesA(rank_A, 1), indicesB(rank_B, 1), indicesC;

				indicesC = C.Index2indices(indexC);

				for (unsigned int i = 0; i < indicesC.GetLength(); i++)
				{
					if (i < rank_A)
					{
						indicesA(i) = indicesC(i);
					}
					else
					{
						indicesB(i - rank_A) = indicesC(i);
					}
				}

				unsigned INT indexA = A.indices2Index(indicesA),
					indexB = B.indices2Index(indicesB);

				C(indexC) = A(indexA)*B(indexB);
			}
		}

		return C;
	}

	FTensor operator^(const FMatrix& A, const FTensor& B)
	{
		const FTensor TA = A;

		return TA ^ B;
	}

	FTensor FTensor::operator+(const FTensor& B) const
	{
		if (B._length == 0)
			return *this;
		else if (_length == 0)
			return B;

		if(_rank != B._rank)
			THROWERROR(L"operator+: Tensors must have the same rank\r\n");

		bool check = true;
		for (unsigned int r = 0; r < _rank; r++)
		{
			if (_dims[r] != B._dims[r])
			{
				check = false;
				break;
			}
		}
		if (!check)// all dimensions match
			THROWERROR(L"operator+: Tensors must have the same dimensions\r\n");

		FTensor C(CMatrix<unsigned INT>(_rank, 1, _dims));

		vdAdd(_length, _data, B._data, C._data);

		return C;
	}

	FTensor FTensor::operator-(const FTensor& B) const
	{
		if (B._length == 0)
			return *this;
		else if (_length == 0)
			return -B;

		if (_rank != B._rank)
			THROWERROR(L"operator-: Tensors must have the same rank\r\n");

		bool check = true;
		for (unsigned int r = 0; r < _rank; r++)
		{
			if (_dims[r] != B._dims[r])
			{
				check = true;
				break;
			}
		}
		if (!check)// No transposition required
			THROWERROR(L"operator-: Tensors must have the same dimensions\r\n");

		FTensor C(CMatrix<unsigned INT>(_rank, 1, _dims));

		vdSub(_length, _data, B._data, C._data);

		return C;
	}

	FTensor FTensor::operator-() const
	{
		return (*this)*(-1.0);
	}


	FTensor& FTensor::operator= (const FTensor& Orig)
	{
		Clear();
		PrivateCopy(Orig);

		return *this;
	}
	FTensor& FTensor::operator= (const FMatrix& Orig)
	{
		Initialize(Orig.ToDoublePtr(), 2, Orig.GetNRows(), Orig.GetNColumns());

		if (Orig.GetType() == Diagonal)
			_type = TT_Diagonal;

		return *this;
	}
	
	const double* FTensor::ToPtr() const
	{
		return _data;
	}

	FTensor::operator CMatrixLib::FMatrix() const
	{
		if (_rank == 0)
			return EmptyMatrix;

		if (_rank != 2)
			THROWERROR(L"operator CMatrixLib::FMatrix: Cannot convert a tensor with a rank higher or lower than 2 to a Matrix");

		return FMatrix(_dims[0], _dims[1], _data);
	}
	
	ofstream& operator<<(ofstream& out, const FTensor& Tensor)
	{
		for (unsigned int r = 0; r < Tensor.GetRank(); r++)
		{
			out << "N" << r + 1 << "= " << Tensor._dims[r] << "\t";
		}

		out.precision(16);
		for (unsigned INT i = 0; i<Tensor._length; i++)
		{
			out << scientific << Tensor(i) << " ";
		}

		return out;
	}
	
	FTensor FTensor::T(const FTensor& Tensor, const unsigned int* permut)
	{
		if (Tensor._type == TT_Diagonal)
			return Tensor;
		FTensor res;

		//Permutation check
		for (unsigned int r = 0; r < Tensor._rank; r++)
		{
			if (permut[r] >= Tensor._rank)
				THROWERROR(L"T: the values of the dimensional indices must be smaller than the matrix rank (zero based).\r\n");

			for (unsigned int r1 = 0; r1 < Tensor._rank; r1++)
			{
				if (r1 == r)
					continue;

				if (permut[r1] == permut[r])
					THROWERROR(L"T: the values of the dimensional indices must be different.\r\n");
			}
		}

		bool check = false;
		for (unsigned int r = 0; r < Tensor.GetRank(); r++)
		{
			if (permut[r] != r)
			{
				check = true;
				break;
			}
		}
		if (!check)// No transposition required
			return Tensor;
		//

		unsigned INT* dims = (unsigned INT*)malloc(Tensor._rank * sizeof(unsigned INT));
		for (unsigned int r = 0; r < Tensor._rank; r++)
		{
			dims[r] = Tensor.GetDim(permut[r]);
		}

		res.PrivateInitialize(0, Tensor._rank, dims);

		int Index = 0;
		//#pragma omp parallel shared(Tensor,res) private(Index)
		{
			//#pragma omp for
			for (Index = 0; Index < Tensor.GetLength(); Index++)
			{
				CMatrix<unsigned INT>	indices_orig = Tensor.Index2indices(Index),
					indices_T(indices_orig.GetLength(), 1);

				for (unsigned int r = 0; r < indices_orig.GetLength(); r++)
				{
					indices_T(r) = indices_orig(permut[r]);
				}

				unsigned INT Index_T = res.indices2Index(indices_T);

				res(Index_T) = Tensor(Index);
			}
		}

		FREE(dims);

		return res;
	}
	FTensor T(const FTensor& Tensor, unsigned int Permut1, ...)
	{
		if (Tensor._type == TT_Diagonal)
		{
			return Tensor;
		}

		unsigned int* permut = (unsigned int*)malloc(Tensor._rank * sizeof(unsigned int));

		va_list args;

		permut[0] = Permut1;
		va_start(args, Permut1);
		for (unsigned int r = 1; r < Tensor._rank; r++)
		{
			permut[r] = (unsigned int)va_arg(args, int);
		}
		va_end(args);

		FTensor res = FTensor::T(Tensor, permut);

		FREE(permut);

		return res;
	}

	FTensor T(const FTensor& Tensor, CMatrix<unsigned int> Permut)
	{
		if (Tensor._type == TT_Diagonal)
		{
			return Tensor;
		}

		FTensor res = FTensor::T(Tensor, Permut.GetDataPtr());

		return res;
	}

	FTensor::FTensorIndex I(const FTensor& Tensor, unsigned int Index)
	{
		FTensor::FTensorIndex t = { Tensor, Index };

		return t;
	}

	RTensor::RTensor()
	{
		_length = 0;

		_orig = 0;
	}
	RTensor::RTensor(const RTensor& RT)
	{
		PrivateCopy(RT);
	}
	RTensor::RTensor(FTensor& FT)
	{
		PrivateCopy(FT);
	}

	RTensor::~RTensor()
	{
		Clear();
	}

	void RTensor::Clear()
	{
		_length = 0;

		_orig = 0;
	}

	FTensor RTensor::GetSubTensor(unsigned int SelectIndex, unsigned int cycle) const
	{
		const unsigned int rank = (unsigned int)_dims.GetLength();
		if (rank == 0)
			return *this;

		CMatrix<unsigned INT> dims(rank - 1, 1);
		unsigned int iter = 0;
		for (unsigned int i = 0; i < rank; i++)
		{
			if (i == SelectIndex)
				continue;

			dims(iter) = _dims(i);

			iter++;
		}

		FTensor res(dims);

		double* data = _orig->_data;
		
		
		//#pragma omp parallel for shared(SelectIndex, cycle,data,res)
		for (INT index = 0; index < (INT)_length; index++)
		{
			CMatrix<unsigned INT> indices = Index2indices(index), indicesRES(indices.GetLength() -1,1);
			unsigned int skip = 0;
			for (unsigned int i = 0; i < indices.GetLength(); i++)
			{
				if (i != SelectIndex)
				{
					indicesRES(skip++) = indices(i);
				}
				
				indices(i) += _start(i);
				
			}

			if (indices(SelectIndex) == cycle)
			{
				unsigned INT IndexRef = _orig->indices2Index(indices);
				unsigned INT IndexRes = res.indices2Index(indicesRES);
				res(IndexRes) = data[IndexRef];
			}
		}

		return res;
	}

	void RTensor::CopyInto(const FTensor& Data)
	{
		if (_orig == 0)
			return;

		for (unsigned int r = 0; r < Data._rank; r++)
		{
			if (_dims(r) != Data._dims[r])
				THROWERROR(L"RTensor::CopyInto: Data to be copied must have the same dimensions of the reference.");
		}
		
		double* rdata = _orig->_data;
		double* mdata = Data._data;
		INT length = _length;

		#pragma omp parallel for shared(mdata,rdata,length)
		for (int I = 0; I < length; I++)
		{
			CMatrix<unsigned INT> indices = Index2indices(I);
			for (unsigned int i = 0; i < indices.GetLength(); i++)
				indices(i) += _start(i);

			unsigned INT IndexRef = _orig->indices2Index(indices);

			_orig->_data[IndexRef] = mdata[I];
		}
	}

	RTensor::RTensor(FTensor& FT, const CMatrix<unsigned INT>& Dims, const CMatrix<unsigned INT>& Start)
	{
		_orig = &FT;

		_dims = Dims;
		_length = 1;
		for (unsigned int i = 0; i < _dims.GetLength(); i++)
			_length *= _dims(i);

		_start = Start;
	}

	void RTensor::PrivateCopy(FTensor& FT)
	{
		_orig = &FT;

		_dims.Initialize(FT._rank, 1, FT._dims);
		_start.Initialize(FT._rank, 1);
		_length = 1;
		for (unsigned int i = 0; i < _dims.GetLength(); i++)
		{
			_length *= _dims(i);
			_start(i) = 0;
		}

	}
	void RTensor::PrivateCopy(const RTensor& RT)
	{
		_orig = RT._orig;

		_dims = RT._dims;
		_length = RT._length;
		_start = RT._start;
	}


	unsigned INT RTensor::indices2Index(CMatrix<unsigned INT> indices) const
	{
		unsigned INT index = indices(0);
		unsigned INT jump = 1;
		for (unsigned INT r = 1; r < _dims.GetLength(); r++)
		{
			jump *= _dims(r - 1);
			index += jump * indices(r);
		}

		return index;
	}

	CMatrix<unsigned INT> RTensor::Index2indices(unsigned INT Index) const
	{
		CMatrix<unsigned INT> indices(_dims.GetLength(), 1);

		unsigned INT temp_Index = Index;
		for (unsigned int r = 0; r < _dims.GetLength(); r++)
		{
			indices(r) = temp_Index % _dims(r);

			temp_Index = (temp_Index - indices(r)) / _dims(r);
		}

		return indices;
	}
	
	double& RTensor::operator() (unsigned INT Index) noexcept(false)
	{
		if (Index >= _length)
			THROWERROR(L"RTensor::operator(): index outside of tensor dimensions\r\n");

		CMatrix<unsigned INT> indices = Index2indices(Index);
		for (unsigned int i = 0; i < indices.GetLength(); i++)
			indices(i) += _start(i);

		unsigned INT IndexRef = _orig->indices2Index(indices);

		return _orig->_data[IndexRef];

	}

	double RTensor::operator() (unsigned INT Index) const noexcept(false)
	{
		if (Index >= _length)
			THROWERROR(L"RTensor::operator(): index outside of tensor dimensions\r\n");

		CMatrix<unsigned INT> indices = Index2indices(Index);
		for (unsigned int i = 0; i < indices.GetLength(); i++)
			indices(i) += _start(i);

		unsigned INT IndexRef = _orig->indices2Index(indices);

		return _orig->_data[IndexRef];
	}

	double& RTensor::operator() (unsigned INT Dim1, unsigned INT Dim2, ...) noexcept(false)
	{
		CMatrix<unsigned INT> indices(_dims.GetLength(), 1);
		va_list args;

		indices(0) = Dim1;
		indices(1) = Dim2;
		va_start(args, Dim2);
		for (unsigned int r = 2; r < indices.GetLength(); r++)
		{
			indices(r) = va_arg(args, int);
		}
		va_end(args);

		for (unsigned int i = 0; i < _dims.GetLength(); i++)
		{
			if (indices(i) >= _dims(i))
				THROWERROR(L"RTensor::operator(): index outside of tensor dimensions\r\n");

			indices(i) += _start(i);
		}

		unsigned INT RefIndex = _orig->indices2Index(indices);

		return _orig->_data[RefIndex];
	}

	double RTensor::operator() (unsigned INT Dim1, unsigned INT Dim2, ...) const noexcept(false)
	{
		CMatrix<unsigned INT> indices(_dims.GetLength(), 1);
		va_list args;

		indices(0) = Dim1;
		indices(1) = Dim2;
		va_start(args, Dim2);
		for (unsigned int r = 2; r < indices.GetLength(); r++)
		{
			indices(r) = va_arg(args, int);
		}
		va_end(args);

		for (unsigned int i = 0; i < _dims.GetLength(); i++)
		{
			if (indices(i) >= _dims(i))
				THROWERROR(L"RTensor::operator(): index outside of tensor dimensions\r\n");

			indices(i) += _start(i);
		}

		unsigned INT RefIndex = _orig->indices2Index(indices);

		return _orig->_data[RefIndex];
	}

	RTensor & RTensor::operator=(FTensor& FT)
	{
		Clear();
		PrivateCopy(FT);
		return *this;
	}
	RTensor& RTensor::operator=(const RTensor& RT)
	{
		Clear();
		PrivateCopy(RT);
		return *this;
	}

	RTensor::operator FTensor() const
	{
		FTensor res(_dims);

		if (_orig == 0)
			return res;

		double* rdata = _orig->_data;
		double* mdata = res._data;
		INT length = _length;

		#pragma omp parallel for shared(mdata,rdata,length)
		for (int I = 0; I < length; I++)
		{
			CMatrix<unsigned INT> indices = Index2indices(I);
			for (unsigned int i = 0; i < indices.GetLength(); i++)
				indices(i) += _start(i);

			unsigned INT IndexRef = _orig->indices2Index(indices);
			

			mdata[I] = _orig->_data[IndexRef];
		}

		return res;
	}
}

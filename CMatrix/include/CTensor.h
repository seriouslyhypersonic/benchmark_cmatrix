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

#ifndef CTENSOR
#define CTENSOR

#include <CMatrix.h>

namespace CMatrixLib
{
	//! Type of Tensor.
	/*!
	This enumeration helps the FTensor class applying special optimizations to the matrix operations.
	*/
	enum TensorType
	{
		TT_None,				//!<Undefined tensor object
		TT_General,			//!<General tensor with no special format.
		TT_Diagonal,			//!<Diagonal tensor.
	};

	class RTensor;

	//!\brief Class for managing nth-rank tensors of doubles.
	/*!
			This class creates and destroys the tensors of doubles and provides a more advanced
			and specialized set of manipulation functions.

			This class is multi-threaded and encapsulates the linear algebra MKL libraries.
	*/
	class CMATRIXLIB_API FTensor
	{
		friend class RTensor;
	private:
		/*!\brief Helper structure to select the inner product indices
		*/
		struct FTensorIndex
		{
			const FTensor& _tensor;	//!< Reference to the tensor to be multiplied
			unsigned int _index;	//!< Index to be used in the inner product
		};
#ifndef __GNUC__
	public:
		_declspec(property(get = GetRank)) unsigned int Rank;						//!< Rank of the tensor

		_declspec(property(get = GetDim)) unsigned INT Dim[];						//!< Size of the given dimension index (zero-based)
		_declspec(property(get = GetDimensions)) CMatrix<unsigned INT> Dimensions;	//!< Array with the sizes of each dimension (zero-based)

		_declspec(property(get = GetLength)) unsigned INT Length;					//!< Total length of the tensor (product of all dimensions)

		_declspec(property(get = GetTensorType, put = SetTensorType)) TensorType Type;
#endif
	protected:
		unsigned int _rank;		//!< Rank of the tensor
		unsigned INT *_dims;	//!< Dimensions of the tensor
		unsigned INT _length;	//!< Total length of the _data pointer;

		TensorType _type;

		double *_data;			//!< Pointer to the tensor data

	public:
		/*!\brief Default FTensor constructor
			Creates a dimensionless empty tensor
		*/
		FTensor();

		/*!\brief Constructs a zeroed tensor with the given rank and dimensions

			This is a variable input method. For the given rank, specify the size of each dimensions.
			Note that, due to how C++ handles ellipsis, the dimensions must be an integer. Otherwise it may create
			an arbitrarily large tensor that occupies a unwanted large quantity of memory.

			If supplied more parameters than the rank, the extra ones will be ignored. If supplied less parameters than the
			tensor rank, the method will pickup whatever is in the next memory position and the size of the tensor will be
			unpredictable. 

			\param[in]	Rank	Desired rank of the tensor.
			\param[in]	Mixed	Size of each dimension.
		*/
		explicit FTensor(unsigned int Rank, ...);

		/*!\brief Constructs a zeroed tensor with the given rank and dimensions

			\param[in]	Indices	Vector containing the size of each dimensions.
						The rank of the tensor will be the length of the Indices vector.
		*/
		FTensor(CMatrix<unsigned INT> indices);


		/*!\brief Constructs a tensor with the given rank, dimensions and fill with data.

		This is a variable input method. For the given rank, specify the size of each dimensions.
		Note that, due to how C++ handles ellipsis, the dimensions must be an integer. Otherwise it may create
		an arbitrarily large tensor that occupies a unwanted large quantity of memory.

		If supplied more parameters than the rank, the extra ones will be ignored. If supplied less parameters than the
		tensor rank, the method will pickup whatever is in the next memory position and the size of the tensor will be
		unpredictable.

		\param[in]	Data	Pointer to the data array to be copied into the tensor (must have a length compatible with the given dimensions).
		\param[in]	Rank	Desired rank of the tensor.
		\param[in]	Mixed	Size of each dimension.
		*/
		FTensor(const double* Data, unsigned int Rank, ...);


		FTensor(const double* data, CMatrix<unsigned INT> indices);
		/*!\brief Default copy constructor

			\param[in]	Orig	Original tensor.
		*/
		FTensor(const FTensor& Orig);

		/*!\brief Converts a FMatrix into a FTensor

			\param[in]	Orig	Original matrix.
		*/
		FTensor(const FMatrix& Orig);

		/*!\brief Default destructor*/
		virtual ~FTensor();

		/*!\brief Clears safely all class members and prepares them for destruction.
		*/
		void Clear();

		/*!\brief Initializes a zeroed tensor with the given rank and dimensions

		This is a variable input method. For the given rank, specify the size of each dimensions.
		Note that, due to how C++ handles ellipsis, the dimensions must be an integer. Otherwise it may create
		an arbitrarily large tensor that occupies a unwanted large quantity of memory.

		If supplied more parameters than the rank, the extra ones will be ignored. If supplied less parameters than the
		tensor rank, the method will pickup whatever is in the next memory position and the size of the tensor will be
		unpredictable.

		\param[in]	Rank	Desired rank of the tensor.
		\param[in]	Mixed	Size of each dimension.
		*/
		void Initialize(unsigned int Rank, ...);

		/*!\brief Initializes a tensor with the given rank, dimensions and fill with data.

		This is a variable input method. For the given rank, specify the size of each dimensions.
		Note that, due to how C++ handles ellipsis, the dimensions must be an integer. Otherwise it may create
		an arbitrarily large tensor that occupies a unwanted large quantity of memory.

		If supplied more parameters than the rank, the extra ones will be ignored. If supplied less parameters than the
		tensor rank, the method will pickup whatever is in the next memory position and the size of the tensor will be
		unpredictable.

		\param[in]	Data	Pointer to the data array to be copied into the tensor (must have a length compatible with the given dimensions).
		\param[in]	Rank	Desired rank of the tensor.
		\param[in]	Mixed	Size of each dimension.
		*/
		void Initialize(const double* Data, unsigned int Rank, ...);

		/*!\brief Returns the tensor rank.
		*/
		unsigned int GetRank() const;

		/*!\brief Returns the size of the given dimension index.

			\param[in]	rank	Zero-based dimension index
		*/
		unsigned INT GetDim(unsigned int rank) const;

		/*!\brief Returns an array with the sizes of each dimension (zero-based)
		*/
		CMatrix<unsigned INT> GetDimensions() const;

		/*!\brief Return the total length of the tensor.
		*/
		unsigned INT GetLength() const;

		/*!\brief Returns a sub-matrix formed by the first two indices of the tensor

			\param[in]	Dim3	Third tensor index
			\param[in]	Mixed	Rest of the tensor indices.
		*/
		FMatrix GetSubMatrix(unsigned INT Dim3, ...) const noexcept(false);

		/*!\brief Sets the value of a sub-matrix formed by the first two indices of the tensor.

			\param[in]	Mat		FMatrix object with the values to be used.
			\param[in]	Dim3	Third tensor index.
			\param[in]	Mixed	Rest of the tensor indices.
		*/
		void SetSubMatrix(const FMatrix& Mat, unsigned INT Dim3, ...) noexcept(false);

		FTensor GetSubTensor(unsigned int SelectIndex, unsigned int cycle) const;

		RTensor RefSubTensor(unsigned int SelectIndex, unsigned int cycle);

		/*!\brief Obtains the value of the memory master index from the tensor indices

		\param[in]	indices	Array with the tensor indices.
		*/
		unsigned INT indices2Index(const CMatrix<unsigned INT>& indices) const;

		/*!\brief Obtains the indices of the tensor from the memory master index.

		\param[in]	Index	Memory master index.
		*/
		CMatrix<unsigned INT> Index2indices(unsigned INT Index) const;

		TensorType GetTensorType() const;
		void SetTensorType(TensorType Type);

	protected:

		/*!	\brief Initializes the class to its default state.

		NOTE: This method only sets the default values of the class members.
		Unless called in a derived method please call FTensor::Clear() first to ensure that the previous
		values of the members are properly disposed of.

		\param[in]	Data	Data to be copies into the tensor. Can be a null vector.
		\param[in]	Rank	Rank of the tensor.
		\param[in]	Dims	Array with the sizes of each dimension of the tensor.
		*/
		void PrivateInitialize(const double* Data, unsigned int Rank,const unsigned INT *Dims, TensorType type = TT_General);

		/*!	\brief Copies the data of another FTensor class.

		NOTE: This method only copies the values of the class members.
		Unless called in a derived method please call FTensor::Clear() first to ensure that the previous
		values of the members are properly disposed of.

		\param[in] Orig	Data to be copied.
		*/
		void PrivateCopy(const FTensor& Orig);

		/*!\brief Obtains the value of the memory master index from the tensor indices

			\param[in]	indices	Array with the tensor indices.
		*/
		unsigned INT indices2Index(unsigned INT* indices) const;


	public:

		/*!\brief Gets or sets the entry in the location given by the memory master index.

			\param[in]	Index	Memory master index.
		*/
		double& operator() (unsigned INT Index) noexcept(false);

		/*!\brief Gets the entry in the location given by the memory master index.

		\param[in]	Index	Memory master index.
		*/
		double operator() (unsigned INT Index) const noexcept(false);

		/*!\brief Gets or sets the entry in the location given by the tensor indices.

		\param[in]	Dim1	First index.
		\param[in]	Dim2	Second index.
		\param[in]	Mixed	Remaining tensor indices
		*/
		double& operator() (unsigned INT Dim1, unsigned INT Dim2, ...) noexcept(false);

		/*!\brief Gets the entry in the location given by the tensor indices.

		\param[in]	Dim1	First index.
		\param[in]	Dim2	Second index.
		\param[in]	Mixed	Remaining tensor indices
		*/
		double operator() (unsigned INT Dim1, unsigned INT Dim2, ...) const noexcept(false);

		/*!\brief Gets or sets the entry in the location given by the tensor indices.

		\param[in]	indices	Vector with the desired set of indices.
		*/
		double& operator() (CMatrix<unsigned INT> indices) noexcept(false);

		/*!\brief Gets the entry in the location given by the tensor indices.

		\param[in]	indices	Vector with the desired set of indices.
		*/
		double operator() (CMatrix<unsigned INT> indices) const noexcept(false);
		
		/*!\brief Performs the inner product between to Tensors

		Performs the inner product of the rightmost index of the
		left tensor with the leftmost index of the right tensor: Ci...j = Ai...k*Bk...j

		\param[in]	B	Tensor
		*/
		FTensor operator*(const FTensor& B) const;

		/*!\brief Performs the inner product between to Tensors

		Performs the inner product between the IA operand and the IB operand, using the indices specified by the I function.

		Example:\n\r
				For: C_(iklm) = A_(ilk)*B_(ljm)\n\r
				Should be: C = I(A,1)*I(B,1)\n\r

		Note that while in index notation the order of the indices is irrelevant, when implementing in software, the order
		of the indices is tied to how the data is physically stored. As consequence, the first set of indices of the resulting tensor
		is the remaining indices of the first operand (in their original order) and the last set is the remaining indices of the second operand
		(in their original order)

		\param[in]	IA	Left operand
		\param[in]	IB	Right operand.
		*/
		friend CMATRIXLIB_API FTensor operator*(const FTensor::FTensorIndex& IA, const FTensor::FTensorIndex& IB);

		/*!\brief Performs the inner product between to Tensors

		Performs the inner product between the IA operand and the B operand, using the indices specified by the I function.

		Example:\n\r
		For: C_(iklm) = A_(ijk)*B_(ljm)\n\r
		Should be: C = I(A,1)*I(B,1)\n\r

		Note that while in index notation the order of the indices is irrelevant, when implementing in software, the order
		of the indices is tied to how the data is physically stored. As consequence, the first set of indices of the resulting tensor
		is the remaining indices of the first operand (in their original order) and the last set is the remaining indices of the second operand
		(in their original order)

		\param[in]	IA	Left operand
		\param[in]	B	Right operand (not indexed tensor, automatically uses the first index).
		*/
		friend CMATRIXLIB_API FTensor operator*(const FTensor::FTensorIndex& IA, const FTensor& B);

		/*!\brief Performs the inner product between to Tensors

		Performs the inner product between the IA operand and the B operand, using the indices specified by the I function.

		Example:\n\r
		For: C_(iklm) = A_(ijk)*B_(ljm)\n\r
		Should be: C = I(A,1)*I(B,1)\n\r

		Note that while in index notation the order of the indices is irrelevant, when implementing in software, the order
		of the indices is tied to how the data is physically stored. As consequence, the first set of indices of the resulting tensor
		is the remaining indices of the first operand (in their original order) and the last set is the remaining indices of the second operand
		(in their original order)

		\param[in]	A	Left operand (not indexed tensor, automatically uses the last index).
		\param[in]	IB	Right operand 
		*/
		friend CMATRIXLIB_API FTensor operator*(const FTensor& A, const FTensor::FTensorIndex& IB);

		/*!\brief Performs the inner product between to a tensor and a matrix

		Performs the inner product between the IA operand and the B operand, using the indices specified by the I function.

		Example:\n\r
		For: C_(iklm) = A_(ijk)*B_(ljm)\n\r
		Should be: C = I(A,1)*I(B,1)\n\r

		Note that while in index notation the order of the indices is irrelevant, when implementing in software, the order
		of the indices is tied to how the data is physically stored. As consequence, the first set of indices of the resulting tensor
		is the remaining indices of the first operand (in their original order) and the last set is the remaining indices of the second operand
		(in their original order)

		\param[in]	A	Left operand
		\param[in]	IB	Right operand
		*/
		friend CMATRIXLIB_API FTensor operator*(const FMatrix& A, const FTensor::FTensorIndex& IB);

		/*!\brief Performs the inner product between to a tensor and a matrix

		Performs the inner product between the IA operand and the B operand, using the indices specified by the I function.

		Example:\n\r
		For: C_(iklm) = A_(ijk)*B_(ljm)\n\r
		Should be: C = I(A,1)*I(B,1)\n\r

		Note that while in index notation the order of the indices is irrelevant, when implementing in software, the order
		of the indices is tied to how the data is physically stored. As consequence, the first set of indices of the resulting tensor
		is the remaining indices of the first operand (in their original order) and the last set is the remaining indices of the second operand
		(in their original order)

		\param[in]	IA	Left operand
		\param[in]	B	Right operand
		*/
		friend CMATRIXLIB_API FTensor operator*(const FTensor::FTensorIndex& IA, const FMatrix& B);

		friend CMATRIXLIB_API FTensor operator*(const FMatrix& A, const FTensor& B);
		friend CMATRIXLIB_API FTensor operator*(const FTensor& A, const FMatrix& B);

		/*!\brief Scalar multiplication between a tensor and a double value

			\param[in]	b	Scalar value.
		*/
		FTensor operator*(const double& b) const;

		/*!\brief Scalar multiplication between a tensor and a double value

		\param[in]	a	Scalar value.
		\param[in]	B	Tensor
		*/
		friend CMATRIXLIB_API FTensor operator*(const double& a, const FTensor B);


		/*!\brief Performs the outer product between two tensors
			
			C_(ijklmn) = A_(ijk)^B_(lmn)

			\param[in]	B	Right operand
		*/
		FTensor operator^(const FTensor& B) const;

		friend CMATRIXLIB_API FTensor operator^(const FMatrix& A, const FTensor& B);

		/*!\brief Tensor addition

		\param[in]	B	Tensor.
		*/
		FTensor operator+(const FTensor& B) const;

		/*!\brief Tensor subtraction

		\param[in]	B	Tensor.
		*/
		FTensor operator-(const FTensor& B) const;

		/*!\brief Tensor negation (multiplies all entry by -1)
		*/
		FTensor operator-() const;

		/*!	\brief Initializes the object with the data of another FTensor class.

			\param[in] Orig	Data to be copied.
		*/
		FTensor& operator= (const FTensor& Orig);

		/*!	\brief Initializes the object with the data of a FMatrix class.

			\param[in] Orig	Data to be copied.
		*/
		FTensor& operator= (const FMatrix& Orig);
		
		/*! \brief Gets the raw data pointer stored in th FTensor object

		NOTE: this is a read only data pointer.
		*/
		const double* ToPtr() const;

		operator FMatrix() const;

		/*!
		Tensor string output to be used with fstream derived class.
		NOTE: this function is equivalent to the SaveToFile but enables multiple saves.

		\param out Ofstream sequence.
		\param Tensor FTensor object to be printed.
		*/
		friend CMATRIXLIB_API ofstream& operator<<(ofstream& out, const FTensor& Tensor);
		
	protected:

		/*!	\brief Tensor transposition

			Switches the order of the indices in memory

			\param[in]	Tensor	Tensor to be transposed.
			\param[in]	permut	Array with the new order of indices (the new order is zero-based,
								must not have repeated indices and must have the size of the tensor rank).
		*/
		static FTensor T(const FTensor& Tensor, const unsigned int* permut);

		/*!	\brief Tensor transposition

		Switches the order of the indices in memory

		\param[in]	Tensor	Tensor to be transposed.
		\param[in]	Permut1	First index
		\param[in]	...		New order of indices (the new order is zero-based,
							must not have repeated indices and must have the size of the tensor rank).
		*/
		friend CMATRIXLIB_API FTensor T(const FTensor& Tensor, unsigned int Permut1, ...);

		/*!	\brief Tensor transposition

		Switches the order of the indices in memory

		\param[in]	Tensor	Tensor to be transposed.
		\param[in]	Permut	Array with the new order of indices (the new order is zero-based,
		must not have repeated indices and must have the size of the tensor rank).
		*/
		friend CMATRIXLIB_API FTensor T(const FTensor& Tensor, CMatrix<unsigned int> Permut);

		/*!\brief Inner product index selection

			This function can only be used in conjunction with the operator*() overloads and
			is used to select which index will be collapsed.

			\param[in]	Tensor	Tensor to be multiplied
			\param[in]	Index	Zero-based index do be used.\n\r
								Example:\n\r
									For: C_(iklm) = A_(ilk)*B_(ljm)\n\r
									Should be: C = I(A,1)*I(B,1)
		*/
		friend CMATRIXLIB_API FTensor::FTensorIndex I(const FTensor& Tensor, unsigned int Index);
	};

	class CMATRIXLIB_API RTensor
	{
		friend class FTensor;
	protected:
		CMatrix<unsigned INT> _dims;
		CMatrix<unsigned INT> _start;
		unsigned INT _length;	//!< Total length of the _data pointer;

		FTensor* _orig;
	public:
		RTensor();
		RTensor(const RTensor& RT);
		RTensor(FTensor& FT);

		~RTensor();

		void Clear();

		FTensor GetSubTensor(unsigned int SelectIndex, unsigned int cycle) const;

		void CopyInto(const FTensor& Data);

	protected:
		RTensor(FTensor& FT, const CMatrix<unsigned INT>& Dims, const CMatrix<unsigned INT>& Start);

		void PrivateCopy(FTensor& FT);
		void PrivateCopy(const RTensor& RT);

		unsigned INT indices2Index(CMatrix<unsigned INT> indices) const;
		CMatrix<unsigned INT> Index2indices(unsigned INT Index) const;

	public:
		/*!\brief Gets or sets the entry in the location given by the memory master index.

		\param[in]	Index	Memory master index.
		*/
		double& operator() (unsigned INT Index) noexcept(false);

		/*!\brief Gets the entry in the location given by the memory master index.

		\param[in]	Index	Memory master index.
		*/
		double operator() (unsigned INT Index) const noexcept(false);

		/*!\brief Gets or sets the entry in the location given by the tensor indices.

		\param[in]	Dim1	First index.
		\param[in]	Dim2	Second index.
		\param[in]	Mixed	Remaining tensor indices
		*/
		double& operator() (unsigned INT Dim1, unsigned INT Dim2, ...) noexcept(false);

		/*!\brief Gets the entry in the location given by the tensor indices.

		\param[in]	Dim1	First index.
		\param[in]	Dim2	Second index.
		\param[in]	Mixed	Remaining tensor indices
		*/
		double operator() (unsigned INT Dim1, unsigned INT Dim2, ...) const noexcept(false);

		RTensor & operator=(FTensor& FT);
		RTensor& operator=(const RTensor& RT);

		operator FTensor() const;
	};
}

#endif
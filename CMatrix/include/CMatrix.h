/*
 * Copyright (c) André Carvalho
 */
#pragma once
////////////////////All Matrices are stored as column matrices\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#ifndef __GNUC__
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

#ifdef __GNUC__
#define NOVARIANT
#endif

#ifndef CMATRIX
#define CMATRIX

#ifndef NOVARIANT
#include <atlbase.h>
#endif

#if defined _DEBUG && _WIN32
	#define _CRTDBG_MAP_ALLOC
	#include <stdlib.h>
	#include <crtdbg.h>

	#include <errno.h>
#endif

#ifndef __GNUC__
	#define THROWERROR(msg)\
			{\
				OutputDebugString(msg);\
				throw(msg);\
			}
#else
	#define THROWERROR(msg){cerr<<msg<<endl;}
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <cstring>
#include <malloc.h>
#include <sys/types.h>

#ifdef __GNUC__
#include <sys/time.h>
#endif

#include <sys/timeb.h>
#include <stdarg.h>
#include <type_traits>

#define INT long long

//#ifdef CMATRIX_EXPORTS
	#define MKL_INT INT

	#include <stdafx.h>
	#include <mkl.h>
	#include <mkl_cblas.h>
	#include <mkl_blas.h>
	#include <mkl_vml.h>
	#include <mkl_vml_defines.h>
	#include <mkl_vml_functions.h>
	#include <mkl_lapack.h>

	#include <omp.h>
//#endif


/*! Defines the condition to test a NaN value
 */
#define isNaN(x) ((x) != (x))

using namespace std;

#pragma warning(disable : 4995)

//!\brief Linear Algebra and matrix manipulation library
/*!
	This library provides general matrix array manipulation through the CMatrix class and a
	more advanced double precision matrix algebra with the FMatrix class.

	The FMatrix class uses the Intel MKL library as basis for most Linear Algebra operations,
	with optimizations for special matrix type, such has diagonal, symmetric or triangular matrices.
	Using the openMP library, the FMatrix class also provides a multithreaded operations for larger matrices.

	Lastly, the BMatrix provides an extended matrix array manipulation for boolean values.
*/
namespace CMatrixLib
{
	//!Sign function
	/*!
		Calculates the sign of a double value.
		\return Sign of the value: -1 for negative value, 1 for positive value and 0 otherwise.
	*/
	CMATRIXLIB_API INT sign(double _X);

	//!Sign Factorial
	/*!
		Calculates the factorial of an 64bit integer value.
	*/
	CMATRIXLIB_API unsigned INT Factorial(unsigned INT _X);

	//!Gamma Function
	/*!
		Implementation of the David W. Cantrell's 23 kt. asymptotic expansion for gamma function.
	*/

	CMATRIXLIB_API double Gamma(double _X);

	//!The principal branch of the LambertW function
	CMATRIXLIB_API double LambertW(double _X, double Tolerance);

	//!Inverse Gamma Function
	/*!
		Inverse of the Gamma function based on the 23 kt. asymptotic expansion.
	*/
	CMATRIXLIB_API double InverseGamma(double _X);

	//!FreeMKLBuffers
	/*!
		Frees any unused memory buffers created by the underlying MKL routines.
		Only to be used is there will be a percieved memory shortage and at the end of the program.
	*/
	CMATRIXLIB_API void FreeMKLBuffers();

	#pragma region CMatrix

	//!Helper function for deletion of arrays when mixing release and debug versions of the delete operator
	template<typename T>
	void SpecialDelete(T** ptr)
	{
		delete[] *ptr;
	}

	//!Helper function for creation of arrays when mixing release and debug versions of the new operator
	template<typename T>
	void SpecialNew(T** ptr, size_t len)
	{
		*ptr = new T[len];
	}

#ifndef SPECIALINT
#define SPECIALINT
	template<>
	CMATRIXLIB_API void SpecialNew<INT>(INT** ptr, size_t len);

	template<>
	CMATRIXLIB_API void SpecialDelete<INT>(INT** ptr);
#endif

	//! Class for managing Matrices of an arbitrary type.
	/*!
		This template class creates and destroys matrices of a type defined by the user
		and provides basic matrix manipulation. For a more specialized matrix class for double
		or boolean values please use the FMatrix and BMatrix classes, respectively.
	 */
	template<typename T>
	class CMATRIXLIB_API CMatrix
	{
	private:
		bool _debug_new;

		protected:
			unsigned INT _Ni;			//!< Number of rows of the matrix.
			unsigned INT _Nj;			//!< Number of columns of the matrix.

			unsigned INT *_dims[2];		//!< Helper member for the VS2012 visualizer

			bool _delete_pointer;

			T *_data;					//!< Data array.
#ifdef _WIN32
		public:
			_declspec(property(get=GetNRows)) unsigned INT NRows;							//!< Number of rows of the matrix (property).
			_declspec(property(get=GetNColumns)) unsigned INT NColumns;						//!< Number of columns of the matrix (property).
			_declspec(property(get=GetLength)) unsigned INT Length;							//!< Number of elements in the matrix (property).
			_declspec(property(get=GetRow, put=SetRow)) CMatrix<T> Row[][];					//!< Returns a determined row of the matrix (property).
			_declspec(property(get=GetColumn, put=SetColumn)) CMatrix<T> Column[][];		//!< Returns a determined row of the matrix (property).
			_declspec(property(get=GetRange,put=SetRange)) CMatrix<T> Range[][][][];		//!< Returns a sub-matrix (property).

			_declspec(property(get = GetDeletePointers, put = SetDeletePointers)) bool DeletePointers; //!<Marks if the pointer data is to be deleted with the CMatrix. NOTE: for pointer type CMatrix only.
#endif
		public:
			//! Creates an empty matrix.
			/*!
				Creates an empty matrix to be initialized afterward by Initialize or by the attribution operator.
			 */
			CMatrix()
			{
				_Ni = 0;
				_Nj = 0;
				_dims[0] = &_Ni;
				_dims[1] = &_Nj;

				_data = 0;

				_delete_pointer = false;

#ifdef _DEBUG
				_debug_new = true;
#else
				_debug_new = false;
#endif
			};

			//! Creates a matrix.
			/*!
				Creates a matrix with the determined dimensions.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
			 */
			CMatrix(unsigned INT Ni,unsigned  INT Nj)
			{
				PrivateInitialize(Ni,Nj,0);
			};

			//! Creates a matrix.
			/*!
				Creates a matrix with the determined dimensions and fills it with data.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param data Data array to fill the matrix. The dimensions of the array must match the length of the created matrix.
						  Extra elements in the array will be ignored and a shorted array will lead to unpredictable results.
			 */
			CMatrix(unsigned INT Ni,unsigned  INT Nj, const T* data)
			{
				PrivateInitialize(Ni,Nj,data);
			};

			//! Copies a matrix.
			/*!
				Copies an existing matrix.

				\param Orig Original matrix to be copied.
			 */
			CMatrix(const CMatrix<T>& Orig)
			{
				PrivateCopy(Orig);
			};

			//! Clear the matrix
			/*!
				Clears the matrix, transforming it into an empty matrix, that can be reinitialized.
			 */
			//General version
			template<class T1=T>
			void Clear(typename std::enable_if<!std::is_pointer<T1>::value>::type* =0)
			{
				bool debug_delete;
#ifdef _DEBUG
				debug_delete = true;
#else
				debug_delete = false;
#endif
				bool debug_new = (((INT)_debug_new) > 0)?true:false;	//Limpa lixo do bool vindo de release
				
				_Ni = 0;
				_Nj = 0;


				if(_data!=0)
				{
					if(debug_delete == debug_new)
						delete[] _data;
					else
						SpecialDelete<T>(&_data);
				}
				_data = 0;
			}

			//Pointer (non function pointer) specialization
			template<class T1 = T>
			void Clear(typename std::enable_if<std::is_pointer<T1>::value && !std::is_function<typename std::remove_pointer<T1>::type>::value>::type* = 0)
			{
				bool debug_delete;
#ifdef _DEBUG
				debug_delete = true;
#else
				debug_delete = false;
#endif
				bool debug_new = (((INT)_debug_new) > 0) ? true : false;	//Limpa lixo do bool vindo de release

				if (_delete_pointer)
				{
					for (unsigned INT e = 0; e < _Ni*_Nj; e++)
					{
						if (_data[e] != 0)
						{
							delete _data[e];
							_data[e] = 0;
						}
					}
				}

				_Ni = 0;
				_Nj = 0;


				if (_data != 0)
				{
					if (debug_delete == debug_new)
						delete[] _data;
					else
						SpecialDelete<T>(&_data);
				}
				_data = 0;
			}

			//Function pointer specialization
			template<class T1 = T>
			void Clear(typename std::enable_if<std::is_pointer<T1>::value && std::is_function<typename std::remove_pointer<T1>::type>::value>::type* = 0)
			{
				bool debug_delete;
#ifdef _DEBUG
				debug_delete = true;
#else
				debug_delete = false;
#endif
				bool debug_new = (((INT)_debug_new) > 0) ? true : false;	//Limpa lixo do bool vindo de release

				_Ni = 0;
				_Nj = 0;


				if (_data != 0)
				{
					if (debug_delete == debug_new)
						delete[] _data;
					else
						SpecialDelete<T>(&_data);
				}
				_data = 0;
			}

			//! Destructor of the CMatrix class.
			~CMatrix()
			{
				Clear();
			}

			//! Initializes a matrix.
			/*!
				Initializes an empty matrix or reinitializes an existing one with the determined dimensions and fills it with data.

				\param Ni Number of rows of the matrix.
				\param Nj Number of columns of the matrix.
				\param data Data array to fill the matrix. The dimensions of the array must match the length of the created matrix.
						  Extra elements in the array will be ignored and a shorted array will lead to unpredictable results.
			 */
			void Initialize(unsigned INT Ni,unsigned  INT Nj, const T* data =0)
			{
				Clear();

				PrivateInitialize(Ni,Nj,data);
			}

			//! Return a element of the matrix.
			/*!
				Returns the address of a determined element of the matrix (read and write).

				\param i Desired row (zero based).
				\param j Desired column (zero based).
				\return Desired element address.
			 */
			T& operator() (unsigned INT i,unsigned  INT j) noexcept(false)
			{
				if(i >= (unsigned INT)_Ni || j >=(unsigned INT) _Nj)
					THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

				return _data[i+_Ni*j];
			}

			//! Return a element of the matrix.
			/*!
				Returns a determined element of the matrix (read only).

				\param i Desired row (zero based).
				\param j Desired column (zero based).
				\return Desired element.
			 */
			T& operator() (unsigned INT i,unsigned  INT j) const noexcept(false)
			{
				if(i >= (unsigned INT)_Ni || j >= (unsigned INT)_Nj)
					THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

				return _data[i+_Ni*j];
			}

			//! Return a element of the matrix.
			/*!
				Returns the address of a determined element of the matrix (read and write).

				\param i Desired element (zero based in a column major organization).
				\return Desired element address.
			 */
			T& operator() (unsigned INT i) noexcept(false)
			{
				if(i >= (unsigned INT)(_Ni*_Nj))
					THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

				return _data[i];
			}

			//! Return a element of the matrix.
			/*!
				Returns a determined element of the matrix (read only).

				\param i Desired element (zero based in a column major organization).
				\return Desired element.
			 */
			T& operator() (unsigned INT i) const noexcept(false)
			{
				if(i >= _Ni*_Nj)
					THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

				return _data[i];
			}

			//! Return row of the matrix.
			/*!
				Returns a determined row of the matrix (read only).

				\param i Desired row (zero based).
				\return Desired row.
			 */
			CMatrix<T> GetRow(unsigned INT i) const
			{
				if(i >= (unsigned INT)_Ni)
					THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

				CMatrix<T> res(1,_Nj);

				for(unsigned INT j = 0; j < _Nj; j++)
					res(0,j) = _data[i+_Ni*j];

				return res;
			};

			//! Return row of the matrix.
			/*!
				Returns a set of rows of the matrix (read only).

				\param i Desired row (zero based).
				\return Desired row.
			 */
			CMatrix<T> GetRow(unsigned INT i0, unsigned INT i1) const
			{
				if(i0 < 0)
					THROWERROR(L"GetLine: i0 must be greater or equal to 0\r\n")
				else if(i1 > _Ni-1)
					THROWERROR(L"GetLine: i1 is outside the matrix limits\r\n")
				else if(i0 > i1)
					THROWERROR(L"GetLine: i1 must be greater or equal than i0\r\n")

				unsigned INT NI = 0;

				if(i0 == i1)
					NI = 1;
				else
					NI = i1-i0+1;

				CMatrix<T> res(NI,_Nj);

				for(unsigned INT i = 0; i < NI; i++)
				{
					for(unsigned INT j = 0; j<_Nj; j++)
					{
						res(i,j) = _data[i0+i+_Ni*j];
					}
				}

				return res;
			};

			//!Replaces the specified row with the values of a vector with the same number of elements
			/*!
				\param i Index of the row to be selected (zero-based).
				\param NewLine Vector to be copied into the row.
			 */
			void SetRow(unsigned INT i,const CMatrix<T>& NewLine)
			{
				if(NewLine.Ni > 1)
					THROWERROR(L"NewLine must be a line vector\r\n");

				if(NewLine.Nj != _Nj)
					THROWERROR(L"NewLine must have the same number of columns as the matrix\r\n");

				for(unsigned INT j = 0; j < _Nj; j++)
				{
					(*this)(i,j) = NewLine(j);
				}
			}

			//!Replaces the specified range line with the values of a matrix with equivalent dimensions
			/*!
				\param i0 Index of the first row to be selected (zero-based).
				\param i1 Index of the last row to be selected (zero-based).
				\param NewLine Matrix to be copied into the row range.
			 */
			void SetRow(unsigned INT i0, unsigned INT i1, const CMatrix<T>& NewLine)
			{
				unsigned INT NI = 0;

				if(i0 == i1)
					NI = 1;
				else
					NI = i1-i0+1;

				if(NewLine.Ni != NI)
					THROWERROR(L"The rhs must have the same number of columns as specified\r\n");

				if(NewLine.Nj != _Nj)
					THROWERROR(L"NewLine must have the same number of columns as the matrix\r\n");

				for(unsigned INT i = 0; i < NI; i++)
				{
					for(unsigned INT j = 0; j < _Nj; j++)
					{
						(*this)(i+i0,j) = NewLine(i,j);
					}
				}
			}

			//! Returns a column of the matrix.
			/*!
				Returns a determined column of the matrix (read only).

				\param j Desired column (zero based).
				\return Desired column.
			 */
			CMatrix<T> GetColumn(unsigned INT j) const
			{
				if(j >= (unsigned INT)_Nj)
					THROWERROR(L"GetColumn: index outside of matrix dimensions\r\n");

				CMatrix<T> res(_Ni,1);

				for(unsigned INT i = 0; i < _Ni; i++)
					res(i,0) = _data[i+_Ni*j];

				return res;
			};

			//!Gets the specified column range
			/*!
				\param j0 Index of the first column to be selected (zero-based).
				\param j1 Index of the last column to be selected (zero-based).
			
				\return Selected column range.
			 */
			CMatrix<T> GetColumn(unsigned INT j0, unsigned INT j1) const
			{
				if(j0 > j1)
					THROWERROR(L"GetColumn - The index of the last column must be larger than the firt one\r\n");

				if(j0 < 0)
					THROWERROR(L"GetColumn - The lower index is outside the matrix bounds\r\n");

				if(j1 >= _Nj)
					THROWERROR(L"GetColumn - The upper index is outside the matrix bounds\r\n");

				unsigned INT NJ = 0;

				if(j0 == j1)
					NJ = 1;
				else
					NJ = j1-j0+1;

				CMatrix<T> res(_Ni,NJ);

				for(unsigned INT j = 0; j<NJ; j++)
				{
					for(unsigned INT i = 0; i < _Ni; i++)
					{
						res(i,j) = _data[i+_Ni*(j0+j)];
					}
				}

				return res;
			};

			//!Replaces the specified column with the values of a vector with the same number of elements
			/*!
				\param j Index of the column to be selected (zero-based).
				\param NewCol Vector to be copied into the column.
			 */
			void SetColumn(INT j,const CMatrix<T>& NewCol)
			{
				if(NewCol.Nj > 1)
					THROWERROR(L"NewCol must be a column vector\r\n");

				if(NewCol.Ni != _Ni)
					THROWERROR(L"NewCol must have the same number of lines as the matrix\r\n");

				for(unsigned INT i = 0; i < _Ni; i++)
				{
					(*this)(i,j) = NewCol(i);
				}
			}

			//!Replaces the specified column range with the values of a matrix with equivalent dimensions
			/*!
				\param j0 Index of the first column to be selected (zero-based).
				\param j1 Index of the last column to be selected (zero-based).
				\param NewCol Matrix to be copied into the column range.
			 */
			void SetColumn(INT j0, INT j1, const CMatrix<T>& NewCol)
			{
				INT NJ = 0;

				if(j0 == j1)
					NJ = 1;
				else
					NJ = j1-j0+1;

				if(NewCol.Nj != NJ)
					THROWERROR(L"The rhs must have the same number of columns as specified\r\n");

				if(NewCol.Ni != _Ni)
					THROWERROR(L"NewCol must have the same number of lines as the matrix\r\n");

				for(unsigned INT j = 0; j < NJ; j++)
				{
					for(unsigned INT i = 0; i < _Ni; i++)
					{
						(*this)(i,j+j0) = NewCol(i,j);
					}
				}
			}

			bool IsEmpty() const
			{
				if (_Ni == 0 || _Nj == 0)
					return true;
				else
					return false;
			}

			//! Gets pointer to the matrix data array.
			/*!
				Gets a safe (read only) pointer to the matrix data array.

				\return Pointer to the data.
			 */
			const T* GetDataPtr() const
			{
				return _data;
			};


			//! Gets number of row of the matrix.
			/*!
				\return Number of rows.
			 */
			inline unsigned INT GetNRows() const
			{
				return _Ni;
			};

			//! Gets number of columns of the matrix.
			/*!
				\return Number of columns.
			 */
			inline unsigned INT GetNColumns() const
			{
				return _Nj;
			};
			//! Gets number of elements of the matrix.
			/*!
				\return Number of elements.
			 */
			inline unsigned INT GetLength() const
			{
				return _Ni*_Nj;
			};

			//! Gets if the data pointers are to be deleted. NOTE: for pointer type CMatrix only.
			inline bool GetDeletePointers() const
			{
				return _delete_pointer;
			};

			//! Sets if the data pointers are to be deleted. NOTE: for pointer type CMatrix only.
			inline void SetDeletePointers(bool Set)
			{
				_delete_pointer = Set;
			};

			//! Copies a matrix into another matrix.
			/*!
				Copies an existing matrix. Can be used to initialize empty matrices.

				\param Orig Original matrix to be copied.
			 */
			CMatrix<T>& operator= (const CMatrix<T> &Orig)
			{
				Clear();

				PrivateCopy(Orig);

				return *this;
			}

			//!Gets the submatrix defined by beg_line, end_line, beg_column and end_column
			/*!
				\param beg_line Index of the first row to be selected (zero-based).
				\param end_line Index of the last row to be selected (zero-based).
				\param beg_col Index of the first column to be selected (zero-based).
				\param end_col Index of the last column to be selected (zero-based).

				\return Submatrix
			 */
			CMatrix<T> GetRange(unsigned INT beg_line,unsigned  INT end_line,unsigned  INT beg_col,unsigned  INT end_col) const
			{
				INT NI, NJ;

				if(beg_line == end_line)
					NI = 1;
				else
					NI = end_line-beg_line+1;

				if(beg_col == end_col)
					NJ = 1;
				else
					NJ = end_col-beg_col+1;

				CMatrix<T> temp(_Ni,NJ), res(NI,NJ);

	
				for(INT j=0; j < NJ; j++)
				{
					temp.Column[j] = this->Column[j+beg_col];
				}

				for(INT i =0; i < NI; i++)
				{
					res.Row[i] = temp.Row[i+beg_line];
				}

				return res;
			}

			//!Sets the values the submatrix defined by beg_line, end_line, beg_column and end_column with the values of another matrix (with equal dimensions)
			/*!
				\param beg_line Index of the first row to be selected (zero-based).
				\param end_line Index of the last row to be selected (zero-based).
				\param beg_col Index of the first column to be selected (zero-based).
				\param end_col Index of the last column to be selected (zero-based).
				\param newMat Matrix to be copied into the submatrix.
			 */
			void SetRange(unsigned INT beg_line,unsigned  INT end_line,
						  unsigned INT beg_col,unsigned  INT end_col,const CMatrix<T>& newMat)
			{
				unsigned INT NI, NJ;

				if(beg_line == end_line)
					NI = 1;
				else
					NI = end_line-beg_line+1;

				if(beg_col == end_col)
					NJ = 1;
				else
					NJ = end_col-beg_col+1;

				T *temp = &_data[beg_line+_Ni*beg_col];
				for(unsigned INT j=0;j<NJ;j++)
				{
					for(unsigned INT i=0;i<NI;i++)
					{
						temp[i] = newMat(i,j);
					}
					temp += _Ni;
				}
			}

			//! Resizes a matrix
			/*!
				Resizes an existing CMatrix object.
				If the matrix is cropped, the extra elements are deleted.

				\param Ni New number of rows of the matrix.
				\param Nj New number of columns of the matrix.
			 */
			void Resize(unsigned INT Ni,unsigned  INT Nj)
			{
				CMatrix<T> temp = *this;

				this->Initialize(Ni,Nj);

				if(Ni >= temp.Ni && Nj >= temp.Nj)
				{
					this->Range[0][temp.Ni-1][0][temp.Nj-1] = temp;
				}
				else if(Ni < temp.Ni && Nj >= temp.Nj)
				{
					this->Column[0][temp.Nj-1] = temp.Row[0][Ni-1];
				}
				else if(Ni >= temp.Ni && Nj < temp.Nj)
				{
					this->Row[0][temp.Ni-1] = temp.Column[0][Nj-1];
				}
				else if(Ni < temp.Ni && Nj < temp.Nj)
				{
					this->Range[0][Ni-1][0][Nj-1] = temp.Range[0][Ni-1][0][Nj-1];
				}
			}

			void Reshape(unsigned INT NewNi,unsigned INT NewNj)
			{
				if(NewNi*NewNj != _Ni*_Nj)
					THROWERROR(L"Reshape: Matrix must have the same number of elements\r\n");

				_Ni = NewNi;
				_Nj = NewNj;
			}

			void ForceRelease()
			{
				if(_data == 0)
					_debug_new = false;
			}

		protected:

			//! Initializes a matrix.
			/*!
				Initializes an empty matrix or reinitializes an existing one with the determined dimensions and fills it with data.

				\param Ni Number of rows of the matrix.
				\param Nj Number of columns of the matrix.
				\param data Data array to fill the matrix. The dimensions of the array must match the length of the created matrix.
						  Extra elements in the array will be ignored and a shorted array will lead to unpredictable results.
			 */
			template<class T1=T>
			void PrivateInitialize(unsigned INT Ni,unsigned  INT Nj, const T* data, typename std::enable_if<!std::is_pointer<T1>::value>::type* = 0)
			{
				_delete_pointer = false;

				if(Ni*Nj == 0)
				{
					_data = 0;

					_Ni = 0;
					_Nj = 0;

					return;
				}

				_Ni = Ni;
				_Nj = Nj;

				bool prev_debug_new = (((INT)_debug_new) > 0)?true:false;

#ifdef _DEBUG
				_debug_new = true;
#else
				_debug_new = false;
#endif

				if(!prev_debug_new && _debug_new)
				{
					SpecialNew<T>(&_data,_Ni*_Nj);
					_debug_new = prev_debug_new;
				}
				else
				{
					_data = new T[_Ni*_Nj];
				}

				if(data != 0)
				{
					for(unsigned INT i=0; i<_Ni*_Nj;i++)
						_data[i] = data[i];
				}

				_dims[0] = &_Ni;
				_dims[1] = &_Nj;
			}
			template<class T1 = T>
			void PrivateInitialize(unsigned INT Ni, unsigned  INT Nj, const T* data, typename std::enable_if<std::is_pointer<T1>::value>::type* = 0)
			{
				_delete_pointer = false;

				if (Ni*Nj == 0)
				{
					_data = 0;

					_Ni = 0;
					_Nj = 0;

					return;
				}

				_Ni = Ni;
				_Nj = Nj;

				bool prev_debug_new = (((INT)_debug_new) > 0) ? true : false;

#ifdef _DEBUG
				_debug_new = true;
#else
				_debug_new = false;
#endif

				if (!prev_debug_new && _debug_new)
				{
					SpecialNew<T>(&_data, _Ni*_Nj);
					_debug_new = prev_debug_new;
				}
				else
				{
					_data = new T[_Ni*_Nj];
				}

				if (data != 0)
				{
					for (unsigned INT i = 0; i<_Ni*_Nj; i++)
						_data[i] = data[i];
				}
				else
				{
					for (unsigned INT i = 0; i<_Ni*_Nj; i++)
						_data[i] = 0;
				}

				_dims[0] = &_Ni;
				_dims[1] = &_Nj;
			}

			//! Copies a matrix into another matrix.
			/*!
				Copies an existing matrix. Can be used to initialize empty matrices.

				\param Orig Original matrix to be copied.
			 */
			void PrivateCopy(const CMatrix<T>& Orig)
			{
				PrivateInitialize(Orig._Ni,Orig._Nj,Orig._data);
			}
	};

	

	//! Creates an empty matrix.
	/*!
		\return Empty matrix.
	 */
	template<typename T>
	CMatrix<T> EmptyCMatrix()
	{
		CMatrix<T> empty;

		return empty;
	}

	/*!
		Matrix string output to be used with fstream derived class.
		
		\attention This function only works for supported types by wofstream

		\param out Ofstream sequence.
		\param Mat FMatrix object to be printed.
	*/
	template<typename T>
	wofstream& operator<<(wofstream& out,CMatrix<T> &Mat)
	{
		out<<"Ni= "<<Mat.NRows<<" Nj= "<<Mat.NColumns<<" ";
		for(unsigned INT i=0;i<Mat.NRows*Mat.NColumns;i++)
		{
			out<<Mat(i)<<" ";
		}

		return out;
	}

	/*!
		Matrix string input to be used with fstream derived class.
		
		\attention This function only works for supported types by wifstream

		\param in Ifstream sequence.
		\param Mat FMatrix object to be printed.
	*/
	template<typename T>
	wifstream& operator>>(wifstream& in,CMatrix<T> &Mat)
	{
		wchar_t temp[20];

		INT ni = 0,no = 0;

		in>>temp>>ni>>temp>>no;

		Mat.Initialize(ni,no);

		for(INT i = 0; i < ni*no; i++)
		{
			in>>Mat(i);
		}

		return in;
	}

	/*!
		Vertical concatenation of matrix A and B.
		NOTE: The A and B matrices must have the same number of columns.

		\param A Top matrix.
		\param B Bottom matrix.

		\return Concatenated matrix
	*/
	template<typename T>
	CMatrix<T> VCat(const CMatrix<T>& A, const CMatrix<T>& B)
	{
		if(A.Length == 0)
		{
			return B;
		}
		else if(B.Length == 0)
			return A;

		if(A.NColumns != B.NColumns)
			THROWERROR(L"VCat: When concatenating vertically, the matrices must have the same number of columns\r\n");

		CMatrix<T> res(A.NRows+B.NRows,A.NColumns);
	
		for(unsigned INT j =0; j<A.NColumns;j++)
		{
			unsigned INT temp = 0;
			for(unsigned INT i=0; i<A.NRows;i++)
			{
				res(temp,j) = A(i,j);
				temp++;
			}

			for(unsigned INT i=0; i<B.NRows;i++)
			{
				res(temp,j) = B(i,j);
				temp++;
			}
		}

		return res;
	}
	#pragma endregion

	#pragma region FMatrix
	//!Type of matrix definition.
	/*!
		Classifies a square matrix according to its definition.
	*/
	enum MatrixDefinition
	{
		PositiveDefinite,		//!<A Positive Definite matrix has all its eigenvalues greater than zero.
		NegativeDefinite,		//!<A Negative Definite matrix has all its eigenvalues smaller than zero.
		PositiveSemiDefinite,	//!<A Positive Semidefinite matrix has all its eigenvalues greater or equal than zero.
		NegativeSemiDefinite,	//!<A Negative Semidefinite matrix has all its eigenvalues smaller or equal than zero.
		Indefinite,				//!<An Indefinite matrix has eigen vaules in the full Real domain. 
		Singular				//!<A Singular matrix has all its eigen vaules equal to zero. This type of matrix cannot be inverted.
	};

	//! Type of matrix.
	/*!
		This enumeration helps the FMatrix class applying special optimizations to the matrix operations.
	*/
	enum MatrixType
	{
		General,			//!<General matrix with no special format.
		UpperTriangular,	//!<Upper triangular matrix.
		LowerTriangular,	//!<Lower triangular matrix.
		Diagonal,			//!<Diagonal matrix.
		Symmetric			//!<Symmetric Matrix.
	};

//	#ifdef CMATRIX_EXPORTS
		#define REMOVE_LU (Auto|QRdecomposition|Choleskydecomposition|Equilibrate|Refine|ExtraRefine)
		#define REMOVE_QR (Auto|LUdecomposition|Choleskydecomposition|Equilibrate|Refine|ExtraRefine)
		#define REMOVE_CL (Auto|LUdecomposition|QRdecomposition|Equilibrate|Refine|ExtraRefine)
		#define DECOMP_METHODS (LUdecomposition|QRdecomposition|Choleskydecomposition)
//	#endif

	//! Options for the Solve function
	enum SolveOptions : unsigned INT
	{
		Auto					= 0x1,	//!< Automaticly find the options for a best possible solution.
		LUdecomposition			= 0x2,	//!< Forces the use an LU decomposition.
		QRdecomposition			= 0x4,	//!< Forces the use a QR decomposition.
		Choleskydecomposition	= 0x8,	//!< Forces the use a Cholesky decomposition.
		Equilibrate				= 0x10, //!< Tries to equilibrate the system matrix.
		Refine					= 0x20, //!< Performs a coarse refinenment of the solution.
		ExtraRefine				= 0x40, //!< Performs a fine refinenment of the solution.

		None					= 0x0	//!< No special options.
	};

	//! Combines two SolveOptions in an OR operation.
	/*!
		This function enables the specification of several options in the Solve function.
		Conflicting options will be ignored.

		\param A First option.
		\param B Second option.
		\return Unsigned INT value containing the conjugated options.
	*/
	unsigned INT operator|(SolveOptions A, SolveOptions B);

	//! Combines two SolveOptions in an AND operation.
	/*!
		\param A First option.
		\param B Second option.
		\return Unsigned INT value containing the conjugated options.
	*/
	unsigned INT operator&(SolveOptions A, SolveOptions B);

	class RMatrix;

	//! Class for managing matrices of doubles.
	/*!
		This class creates and destroys matrices of doubles	and provides a more advanced and specialized set matrix manipulation functions for double matrices.
		This class is multithreaded and encapsulates the linear algebra MKL libraries.
	 */
	class CMATRIXLIB_API FMatrix
	{
		friend class RMatrix;

		protected:
		
			//---Private Data Members-----------------------------------------------
			//
			unsigned INT _Ni;			//!< Number of rows.
			unsigned INT _Nj;			//!< Number of columns.

			unsigned INT *_dims[2];		//!< Helper member for the NatVis visualizer
        
			double *_data;		//!< Matrix values.

			MatrixType _type;	//!< Type of matrix
			//
			//----------------------------------------------------------------------
#ifdef _WIN32
		public:

			__declspec(property(get=GetNRows)) unsigned INT NRows;								//!< Number of rows of the matrix (property).
			__declspec(property(get=GetNColumns)) unsigned INT NColumns;						//!< Number of columns of the matrix (property).
			__declspec(property(get=GetLength)) unsigned INT Length;							//!< Number of elements in the matrix (property).
			__declspec(property(get=GetSizeInBytes)) size_t SizeInBytes;						//!< Amount of memory occupied by the matrix object (property).

			__declspec(property(get=GetColumn,put=ReplaceColumn)) FMatrix Column[];				//!< Returns a determined column or a span of rows of the matrix (property).
			__declspec(deprecated,property(get=GetLine,put=ReplaceLine)) FMatrix Line[];		//!< Returns a determined row or a span of rows of the matrix (property).
			__declspec(property(get=GetLine,put=ReplaceLine)) FMatrix Row[];					//!< Returns a determined row or a span of rows of the matrix (property).
			__declspec(property(get=GetRange,put=SetRange)) FMatrix Range[];					//!< Returns a sub-matrix (property).
			__declspec(property(get=GetType,put=SetType)) MatrixType Type;						//!< Type of the matrix (property).
#endif
		public:

			//---Constructors-------------------------------------------------------
			//
			//! Default contructor.
			/*!	
				Creates an empty FMatrix to be initialized afterward by #Initialize or by the atribution operator.
			 */
			FMatrix();
			//! Creates a zero Ni by Nj matrix.
			/*!
				Creates a zero valued matrix with the determined (Ni and Nj) dimensions.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
			 */
			FMatrix(unsigned INT Ni,unsigned INT Nj);
			//! Creates a Ni by Nj matrix and fills it with values from data.
			/*!
				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param data Data array to fill the matrix. The dimensions of the array must match the length of the created matrix.
						  Extra elements in the array will be ignored and missing elements will be filled with zeros.
			 */
			FMatrix(unsigned INT Ni, unsigned INT Nj, const double* data);
			//! Creates a Ni by Nj matrix.
			/*!
				Creates a Ni by Nj matrix and fills all or just the diagonal elements with a predetermined value.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param val Value to fill the matrix elements with.
				\param diag_only Chooses between filling only the diagonal (true) or all elements (false).
			 */
			FMatrix(unsigned INT Ni, unsigned INT Nj, double val, bool diag_only = true);		

			//! Creates a Ni by Nj random matrix.
			/*!
				Creates a Ni by Nj and fills all or just the diagonal elements with a random matrix values in the interval [min,max].

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param min Minimum bound of the random values interval.
				\param max Maximum bound of the random values interval.
				\param diag_only Chooses between filling only the diagonal (true) or all elements (false).
			 */
			FMatrix(unsigned INT Ni, unsigned INT Nj, double min, double max, bool diag_only = true);

			//! Creates a matrix from a file.
			/*!
				Creates a matrix with the dimensions and values in a file previously created by the #SaveToFile method.

				\param Filename Filename of the matrix file.
			 */
			FMatrix(const char* Filename);

			//! Copies a matrix.
			/*!
				Creates a matrix copied from an existing FMatrix.

				\param Mat Original matrix to be copied.
			 */
			FMatrix(const FMatrix& Mat);

#ifndef NOVARIANT
			//! Copies a matrix.
			/*!
				Creates a matrix copied from an existing matrix in a VARIANT form.

				\param Mat Original matrix to be copied.
			 */
			FMatrix(const VARIANT& Mat);
#endif
			//! Creates a matrix from a string.
			/*!
				Creates a matrix with the dimensions and values in a string previously created by the #ToString method.

				\param MatrixString string with matrix data.
			 */
			FMatrix(const string &MatrixString);

			//! Copies a matrix.
			/*!
				Creates a matrix copied from an existing matrix in a CMatrix form.

				\param Mat Original matrix to be copied.
			 */
			FMatrix(const CMatrix<double>& Mat); 

			//! Creates a one-entry matrix from Val
			FMatrix(const double& Val);
			//! Creates a one-entry matrix from Val
			FMatrix(const float& Val);
			//! Creates a one-entry matrix from Val
			FMatrix(const INT& Val);
			//! Creates a one-entry matrix from Val
			FMatrix(const unsigned INT& Val);
			//
			//----------------------------------------------------------------------

			//---Destructor---------------------------------------------------------
			//
			//! Destructor method
			virtual ~FMatrix();
			//
			//----------------------------------------------------------------------

			//---Public Methods-----------------------------------------------------
			//

			//! Gets number of row of the matrix.
			/*!
				\return Number of rows.
			 */
			inline unsigned INT GetNRows() const {return _Ni;};

			//! Gets number of columns of the matrix.
			/*!
				\return Number of columns.
			 */
			inline unsigned INT GetNColumns() const {return _Nj;};

			//! Gets number of elements of the matrix.
			/*!
				\return Number of elements.
			 */
			inline unsigned INT GetLength() const {return _Ni*_Nj;};

			//! Initializes a matrix.
			/*!
				Initializes an empty FMatrix or reinitializes an existing one with the determined dimensions.

				\param Ni Number of rows of the matrix.
				\param Nj Number of columns of the matrix.
			 */
			void Initialize(unsigned INT Ni,unsigned  INT Nj);

			//! Initializes a matrix.
			/*!
				Initializes an empty FMatrix or reinitializes an existing one with the determined dimensions and fills it with data.

				\param Ni Number of rows of the matrix.
				\param Nj Number of columns of the matrix.
				\param data Data array to fill the matrix. The dimensions of the array must match the length of the created matrix.
						  Extra elements in the array will be ignored and a shorted array will lead to unpredictable results.
			 */
			void Initialize(unsigned INT Ni,unsigned  INT Nj, const double *data);

			//! Initializes a matrix.
			/*!
				Initializes an empty FMatrix or reinitializes an existing one with the determined dimensions
				and fills all or just the diagonal elements with a predetermined value.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param val Value to fill the matrix elements with.
				\param diag_only Chooses between filling only the diagonal (true) or all elements (false).
			 */
			void Initialize(unsigned INT Ni, unsigned INT Nj, double val, bool diag_only = true) noexcept(false);	

			//! Initializes a matrix.
			/*!
				Initializes an empty FMatrix or reinitializes an existing one with the determined dimensions and
				fills all or just the diagonal elements with a random matrix values in the interval [min,max].

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param min Minimum bound of the random values interval.
				\param max Maximum bound of the random values interval.
				\param diag_only Chooses between filling only the diagonal (true) or all elements (false).
			 */
			void Initialize(unsigned INT Ni, unsigned INT Nj, double min, double max, bool diag_only = true);	

			//! Initializes a matrix.
			/*!
				Initializes an empty FMatrix or reinitializes an existing one with the dimensions 
				and values in a file previously created by the #SaveToFile method.

				\param Filename Filename of the matrix file.
			 */
			void Initialize(const char* Filename);											

			//! Resizes a matrix
			/*!
				Resizes an existing FMatrix object.
				If the resulting matrix is larger than the original, the new elements will be filled with zeros.
				If the matrix is cropped, the extra elements are deleted.

				\param Ni New number of rows of the matrix.
				\param Nj New number of columns of the matrix.
			 */
			void Resize(unsigned INT Ni, unsigned INT Nj);

			//!Gets the submatrix defined by beg_line, end_line, beg_column and end_column
			/*!
				\param beg_line Index of the first row to be selected (zero-based).
				\param end_line Index of the last row to be selected (zero-based).
				\param beg_col Index of the first column to be selected (zero-based).
				\param end_col Index of the last column to be selected (zero-based).

				\return Submatrix
			 */
			FMatrix GetRange(unsigned INT beg_line, unsigned INT end_line, unsigned INT beg_col, unsigned INT end_col) const;

			RMatrix RefRange(unsigned INT beg_line, unsigned INT end_line, unsigned INT beg_col, unsigned INT end_col);

			//!Sets the values the submatrix defined by beg_line, end_line, beg_column and end_column with the values of another matrix (with equal dimensions)
			/*!
				\param beg_line Index of the first row to be selected (zero-based).
				\param end_line Index of the last row to be selected (zero-based).
				\param beg_col Index of the first column to be selected (zero-based).
				\param end_col Index of the last column to be selected (zero-based).
				\param newMat Matrix to be copied into the submatrix.
			 */
			void SetRange(unsigned INT beg_line, unsigned INT end_line,unsigned INT beg_col, unsigned INT end_col,const FMatrix& newMat);

			//!Gets the submatrix defined by the indices contained in the point_list.
			/*!
				\param point_list Vector containing the indices to be extracted.

				\return Vector containing the entries defined in the point_list.
			 */
			FMatrix GetRange(const CMatrix<unsigned INT>& point_list) const;

			//!Sets the values the submatrix defined by the indices contained in the point_list with the values of another matrix (with the same length as point_list)
			/*!
				\param point_list Vector containing the indices to be replaced.
				\param newMat Vector (or matrix with equivalent length) containing the new values.

				\param newMat Values to be copied into the submatrix.
			 */
			void SetRange(const CMatrix<unsigned INT>& point_list,const FMatrix& newMat);

			//!Sets the values the submatrix defined by the indices contained in the point_list with a determined value.
			/*!
				\param point_list Vector containing the indices to be replaced.
				\param val Value to be written in the defined indices.
			 */
			void SetRange(const CMatrix<unsigned INT>& point_list,double val);

			//!Replaces the specified column with the values of a vector with the same number of elements
			/*!
				\param j Index of the column to be selected (zero-based).
				\param NewCol Vector to be copied into the column.
			 */
			void ReplaceColumn(unsigned INT j,const FMatrix& NewCol) noexcept(false);

			//!Replaces the specified row with the values of a vector with the same number of elements
			/*!
				\param i Index of the row to be selected (zero-based).
				\param NewLine Vector to be copied into the row.
			 */
			void ReplaceLine(unsigned INT i,const FMatrix& NewLine) noexcept(false);

			//!Replaces the specified column range with the values of a matrix with equivalent dimensions
			/*!
				\param j0 Index of the first column to be selected (zero-based).
				\param j1 Index of the last column to be selected (zero-based).
				\param NewCol Matrix to be copied into the column range.
			 */
			void ReplaceColumn(unsigned INT j0, unsigned INT j1, const FMatrix& NewCol) noexcept(false);

			//!Replaces the specified range line with the values of a matrix with equivalent dimensions
			/*!
				\param i0 Index of the first row to be selected (zero-based).
				\param i1 Index of the last row to be selected (zero-based).
				\param NewLine Matrix to be copied into the row range.
			 */
			void ReplaceLine(unsigned INT i0, unsigned INT i1, const FMatrix& NewLine) noexcept(false);

			//!Gets the specified column
			/*!
				\param j Index of the column to be selected (zero-based).

				\return Selected column
			 */
			FMatrix GetColumn(unsigned INT j) const;


			RMatrix RefColumn(unsigned INT j);

			//!Gets the specified row
			/*!
				\param i Index of the row to be selected (zero-based).

				\return Selected row
			 */
			FMatrix GetLine(unsigned INT i) const;

			RMatrix RefRow(unsigned INT i);

			//!Gets the specified column range
			/*!
				\param j0 Index of the first column to be selected (zero-based).
				\param j1 Index of the last column to be selected (zero-based).
			
				\return Selected column range.
			 */
			FMatrix GetColumn(unsigned INT j0, unsigned INT j1) const;

			//!Gets the specified row range
			/*!
				\param i0 Index of the first row to be selected (zero-based).
				\param i1 Index of the last row to be selected (zero-based).
			
				\return Selected column row.
			 */
			FMatrix GetLine(unsigned INT i0, unsigned INT i1) const;


			//!Reshapes the matrix to match the specified new line and column sizes, provided that the number of elements remains the same.
			/*!
				\param NewNi New number of rows of the matrix.
				\param NewNj New number of columns of the matrix.
			 */
			void Reshape(unsigned INT NewNi, unsigned INT NewNj) noexcept(false);

			//! Clear the matrix
			/*!
				Clears the matrix, transforming it into an empty FMatrix that can be reinitialized.
			 */
			void Clear();

			//! Checks if the matrix has zero dimensions (empty matrix)
			/*!
				\return True if it is an empty FMatrix, false otherwise.
			 */
			bool IsEmpty() const;

			//! Gets a read-only pointer to the data stored in the matrix.
			/*!
				\return Constant pointer to the matrix data.
			*/
			const double* ToDoublePtr() const;

			//! Converts a scalar FMatrix into a double precision value.
			/*!
				NOTE: for non scalar FMatrix, this method returns the first element.

				\return Converted value.
			*/
			double ToDouble() const;

			//!Saves the matrix to a file
			/*!
				Saves the matrix into a text file that can be later opened by the LoadFromFile or the Initialize methods.

				\param Filename Filename of the file to be written with extension.
			*/
			void SaveToFile(const char* Filename) const;

			//!Loads a matrix from a file
			/*!
				Load the matrix from a text file saved with the SaveToFile method.

				\param Filename Filename of the file (with extension) to be read.
			*/
			void LoadFromFile(const char* Filename);

			//!Saves the matrix to a compressed file
			/*!
				Saves the matrix into a compressed text file that can be later opened by the LoadFromCompressedFile or the Initialize methods.
				NOTE: unlike the file written by SaveToFile, this method produces a machine readable file only.

				\param Filename Filename of the file to be written with extension.
			*/
			void SaveToCompressedFile(const char* Filename) const;

			//!Loads a matrix from a compressed file
			/*!
				Load the matrix from a compressed text file saved with the SaveToCompressedFile method.

				\param Filename Filename of the file (with extension) to be read.
			*/
			void LoadFromCompressedFile(const char* Filename);

			//!Saves the matrix to a CSV file
			/*!
				Saves the matrix into a text file using the Comma Seperated Values (CSV) file format.

				\param Filename Filename of the file to be written with extension (usually *.csv).
			*/
			void SaveToCSVFile(const char* Filename) const;	

			//!Loads a matrix from a CSV file
			/*!
				Loads a matrix from a text file using the Comma Seperated Values (CSV) file format.

				\param Filename Filename of the file to be written with extension (usually *.csv).
			*/
			void LoadFromCSVFile(const char* Filename);

			//!Resamples a vector.
			/*!
				Resample a vector keeping only the values with the index given by the increment.

				\param increment Increment os the selected indices.
			*/
			void Pack(unsigned INT increment);

			//!Converts the matrix to a CSV formated string
			/*!
				Converts the matrix into a string using the Comma Seperated Values (CSV) file format.
				The resulting string is equivalent to the text written by the SaveToCSVFile method.

				\return string object containing the matrix
			*/
			string ToCSVstring();	
			
			//!Converts the matrix to a string
			/*!
				Converts the matrix into a string using the same file format as the SaveToFile method.

				\return string object containing the matrix
			*/
			string ToString() const;

			//!Converts the matrix to a compressed string
			/*!
			Converts the matrix into a string using the same file format as the SaveToCompressedFile method.

			\return string object containing the matrix
			*/
			string ToCompressedString() const;

			//!Returns the type of matrix
			/*!
				Gets the type of matrix as defined by the MatrixType enumeration.

				\return Matrix Type
			*/
			inline MatrixType GetType() const {return _type;};


			//!Set the type of matrix
			/*!
				Sets the type of matrix as defined by the MatrixType enumeration.
				NOTE: if the check parameter is set, the method will verify is the given type is compatible with the matrix.
					  If it isn't campatible, the method will throw an exception.

				\param type Type of matrix to be setted.
				\param check Signals the method to perform a check is the given matrix type is compatible (set by default).
			*/
			void SetType(MatrixType type, bool check = true);

			//!Returns the size of the matrix in bytes
			/*!
				This method return an approximated size of the FMatrix object.

				\return Size of the object in bytes.
			*/
			size_t GetSizeInBytes();

			//!Calculates the determinant of the matrix.
			/*!
				NOTE: this method is only valid for square matrices.

				\return value of the determinant.
			*/
			double det(){return Det(*this);};
			//
			//----------------------------------------------------------------------

			//---Public Operators---------------------------------------------------
			//

			//!Gets or sets the (i,j) matrix entry
			/*!
				\param i Zero-based row index.
				\param j Zero-based column index.

			*/
			double& operator() (unsigned INT i, unsigned INT j) noexcept(false);	

			//!Gets the (i,j) matrix entry
			/*!
				\param i Zero-based row index.
				\param j Zero-based column index.

			*/
			double operator() (unsigned INT i, unsigned INT j) const noexcept(false);

			//Gets or sets the i entry of the vector/matrix (on vector form)
			/*!
				\param i Zero-based index.
			*/
			double& operator() (unsigned INT i) noexcept(false);

			//Gets the i entry of the vector/matrix (on vector form)
			/*!
				\param i Zero-based index.
			*/
			double operator() (unsigned INT i) const noexcept(false);	

			//!Matrix addition
			/*!
				Sums two matrices with the same dimentions.

				\param B Second matrix in the addition
			*/
			FMatrix operator+(const FMatrix& B) const noexcept(false);

			//!Matrix addition with a full matrix filled with B (integer)
			/*!
				Sums the matrix with a matrix with the entries equal to B.

				\param B Value of the second matrix entries.
			*/
			FMatrix operator+(INT B);

			//!Matrix addition with a full matrix filled with B (double precision)
			/*!
				Sums the matrix with a matrix with the entries equal to B.

				\param B Value of the second matrix entries.
			*/
			FMatrix operator+(double B);

			//!Matrix subtraction
			/*!
				Subtracts two matrices with the same dimentions.

				\param B Second matrix in the addition.
			*/
			FMatrix operator-(const FMatrix& B) const;

			//!Matrix subtraction with a full matrix filled with B (integer).
			/*!
				Subtracts the matrix with a matrix with the entries equal to B.

				\param B Value of the second matrix entries.
			*/
			FMatrix operator-(INT B) const;	

			//!Matrix subtraction with a full matrix filled with B (double precision).
			/*!
				Subtracts the matrix with a matrix with the entries equal to B.

				\param B Value of the second matrix entries.
			*/
			FMatrix operator-(double B) const;

			//Matrix product by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (integer).

				\param B Value of the scalar.
			*/
			FMatrix operator*(INT B);
			FMatrix operator*(INT B) const;

			//Matrix product by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (double precision).

				\param B Value of the scalar.
			*/
			FMatrix operator*(double B);

			//Matrix product by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (double precision).
				NOTE: the method specialization for constant matrices.

				\param B Value of the scalar.
			*/
			FMatrix operator*(double B) const;

			//Matrix product by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (single precision).

				\param B Value of the scalar.
			*/
			FMatrix operator*(float B);

			//!Matrix Product
			/*!
				Performs a matrix multiplication between two matrices with compatible dimensions (nxm * mxk).

				\param B Second matrix of the multiplication.
			*/
			FMatrix operator*(const FMatrix& B) const noexcept(false);

			//!Element-by-element matrix product
			/*!
				Multiplies directly the elements of two matrices with the same dimensions. 

				\param B Second matrix of the multiplication.
			*/
			FMatrix operator,(FMatrix& B) noexcept(false);

			//Matrix division by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (integer).

				\param B Value of the scalar.
			*/
			FMatrix operator/(INT B) const;
			
			//Matrix division by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (unsigned integer).

				\param B Value of the scalar.
			*/
			FMatrix operator/(unsigned INT B) const;

			//Matrix division by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (double precision).

				\param B Value of the scalar.
			*/
			FMatrix operator/(double B) const;

			//Matrix division by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (single precision).

				\param B Value of the scalar.
			*/
			FMatrix operator/(float B) const;	
			
			//!Matrix division -> A/B=A*Inverse(B)
			/*!
				Performs a matrix division between two matrices with equivalent dimentions.
				This method is equivalent to A*Inverse(B).

				NOTE: This method throw an exception if B is singular.

				\param B Second matrix of the division.
			*/
			FMatrix operator/(const FMatrix& B) const noexcept(false);

			//!Element-by-element matrix division
			/*!
				Divides directly the elements of two matrices with the same dimensions. 

				\param B Second matrix of the multiplication.
			*/
			FMatrix operator|(const FMatrix& B) const noexcept(false);

			//Matrix integer power -> A^2=A*A, A^-1=Inverse(A), A^-2 = Inverse(A^2), A^0 = I
			/*!
				Performs a sequential matrix multiplication of as square matrix.
				NOTE: for operations requiring an inverse, the matrix must not be singular.

				\param B Number of multiplications
			*/
			FMatrix operator^(INT B);

			//!Returns of the symmetric of the matrix -> A + (-A) = 0
			FMatrix operator-() const;

			//! AND operation between two FMatrices
			/*!
				Equivalent to (FMatrix)((BMatrix)A && (BMatrix)B)

				\param B Second matrix in the operation
			*/
			FMatrix operator&&(const FMatrix& B) const;

			//! OR operation between two BMatrices
			/*!
				Equivalent to (FMatrix)((BMatrix)A || (BMatrix)B)

				\param B Second matrix in the operation
			*/
			FMatrix operator||(const FMatrix& B) const;

			public:

			//Elementwise modulus operator
			/*!
				NOTE: This operator is an integer operator both operands will be truncated.

				\param B Matrix to perform the modulus.
			*/
			FMatrix operator%(const FMatrix& B) const;

			//Elementwise right shift operator
			/*!
				NOTE: This operator is an integer operator both operands will be truncated.

				\param B Matrix to perform the shift.
			*/
			FMatrix operator>>(const FMatrix& B) const;

			//Elementwise left shift operator
			/*!
				NOTE: This operator is an integer operator both operands will be truncated.

				\param B Matrix to perform the shift.
			*/
			FMatrix operator<<(const FMatrix& B) const;

			//Elementwise bitwise AND operator
			/*!
				NOTE: This operator is an integer operator both operands will be truncated.

				\param B Matrix to perform the operation.
			*/
			FMatrix operator&(const FMatrix& B) const;

			//!Convertion form a double array.
			/*!
				Converts a double array pointer to a FMatrix object
				NOTE: data must have at least NixNj elements

				\param Mat Pointer to the double precision array.
			*/
			FMatrix& operator= (const double *Mat) noexcept(false);

			//!Convertion form a integer array.
			/*!
				Converts an INT array pointer to a FMatrix object
				NOTE: data must have at least NixNj elements

				\param Mat Pointer to the integer array.
			*/
			FMatrix& operator= (const INT *Mat) noexcept(false);

			//!Copies the values of another FMatrix object.
			/*!
				\param Mat Original matrix.
			*/
			FMatrix& operator= (const FMatrix &Mat);

#ifndef NOVARIANT
			//!Convertion form a VARIANT object.
			/*!
				Converts a VARIANT object to a FMatrix object
				NOTE: the VARIANT must be defined with the VT_DOUBLE|VT_DOUBLE qualifier

				\param Mat VARIANT object.
			*/
			FMatrix& operator= (const VARIANT Mat);
#endif
			//!Copies the values of another double precision CMatrix object.
			/*!
				\param Mat Original matrix.
			*/
			FMatrix& operator= (const CMatrix<double>& Mat);

			//!Copies the values of a double precision variable.
			/*!
				\param Val Value to copy.
			*/
			FMatrix& operator= (const double& Val);

			//!Copies the values of a floating point variable.
			/*!
				\param Val Value to copy.
			*/
			FMatrix& operator= (const float& Val);

			//!Copies the values of an integer variable.
			/*!
				\param Val Value to copy.
			*/
			FMatrix& operator= (const INT& Val);

			//!Copies the values of an unsigned integer variable.
			/*!
				\param Val Value to copy.
			*/
			FMatrix& operator= (const unsigned INT& Val);


			//!Matrix addition and atribution
			/*!
				Sums the matrix with a matrix with the entries equal to B and copies back the result.

				\param B Value of the second matrix entries.
			*/
			FMatrix& operator+=(const FMatrix& B);

			//!Matrix subtraction and atribution
			/*!
				Subtracts the matrix with a matrix with the entries equal to B and copies back the result.

				\param B Value of the second matrix entries.
			*/
			FMatrix& operator-=(const FMatrix& B);												//Matrix subtraction and atribution

			//!Matrix product and atribution
			/*!
				Performs a matrix multiplication between two matrices with compatible dimensions (nxm * mxk) and copies back the result.
				NOTE: the matrix will become an nxk matrix

				\param B Second matrix of the multiplication.
			*/
			FMatrix& operator*=(FMatrix& B);

			//!Division by a scalar and atribution
			/*!
				Multiplies the matrix by a scalar value defined by B.

				\param B Scalar.
			*/
			FMatrix& operator /=(double B);
#ifndef NOVARIANT
			//!Convertion to a VARIANT object
			/*!
				Converts this FMatrix object into a VARIANT object
			*/
			operator VARIANT() const;
#endif
			//!Convertion to a double precision CMatrix object
			/*!
				Converts this FMatrix object into a double precision CMatrix object.
			*/
			operator CMatrix<double>() const;
			//
			//----------------------------------------------------------------------

		private:
			void DestroyMatrix();										//!<Destroys all arrays and resets scalar values.
			bool CheckType(MatrixType _type);							//!<Checks the type of the matrix (for optimization purposes).
			MatrixType OperationResType(const FMatrix& B) const;		//!<Gets the type of matrix resulting from an operation with B.

		public:

			//---Friend Functions and Operators-------------------------------------
			//

			/*!
				Returns the inverse of a non-singular matrix A.
				NOTE: this method will throw an exception if the matrix A is singular.

				\param A Matrix to be inverted.
			*/
			friend CMATRIXLIB_API FMatrix Inverse(const FMatrix &A) noexcept(false);

			/*!
				Calculates the determinant of A.
				\param A Square matrix.
			*/
			friend CMATRIXLIB_API double Det(const FMatrix &A) noexcept(false);

			/*!
				Determines the reciprocal of the condition number of A (1/k(A)).
				The condition number is a measure of how close a matrix is to singularity.
				Matrices with a high condition numbers are prone to generate numerical errors when inverted.

				\param A Square matrix to be used.
				\param infinity_norm Selects the type of norm used to calculate the condition number. Set to use the infinity norm, reset to use the 1-norm.
			*/
			friend CMATRIXLIB_API double ConditionNumber(const FMatrix &A, bool infinity_norm = true) noexcept(false);

			/*!
				LU decomposition of a square matrix -> A = P*L*U.
				\param A Matrix to be decomposed.
				\param P Permutation matrix.
				\param L Lower triangular matrix.
				\param U Upper triangular matrix.
			*/
			friend CMATRIXLIB_API void LUDecomposition(FMatrix &A, FMatrix *P, FMatrix *L, FMatrix *U);

			/*!
				Cholesky decomposition of a symmetric positive-definite matrix -> A = L*Transpose(L).
				NOTE: if the matrix is unsuitable for this transformation (A is non-symmetric or non-positive-definite), the function returns a empty matrix

				\param A A symmetric positive-definite matrix.
				\param L Lower triangular matrix. 

			*/
			friend CMATRIXLIB_API void CholeskyDecomposition(const FMatrix& A, FMatrix& L);

			//!QR decomposition of a generic matrix -> A = Q*R
			/*!
				QR decomposition of a generic matrix -> A = Q*R.
				\param A Matrix do decompose.
				\param Q Orthogonal matrix.
				\param R Upper triangular matrix.
			*/
			friend CMATRIXLIB_API void QRDecomposition(const FMatrix& A,FMatrix* Q, FMatrix* R, bool simple = false);

			/*!
				Calculates the Eigenvalues of a square matrix.
				\param A Input matrix.

				\return Matrix containing the real (first column) and imaginary (second column) parts of the Eigen values.
			*/
			friend CMATRIXLIB_API FMatrix Eigenvalues(const FMatrix& A);

			/*!
				Calculates the Eigenvsystem of a square matrix.

				\param A Input matrix.
				\param LeftEV Resulting Left Eigenvector matrix. For complex eigen values, the imaginary part of the corresponding eigen vector is stored in the column next to the real part. E.g. for a matrix with two real eigen values, each column of the eigenvector matrix corresponds to a eigen value. For two complex eigen values, the first column of the eigenvector is the real part and the second colmn is the imaginary part.
				\param EigenValues Resulting Eigen values. For complex eigen values, the imaginary part is stored on the second column.
				\param LeftEV Resulting Right Eigenvector matrix.
			*/
			friend CMATRIXLIB_API void EigenSystem(const FMatrix& A, FMatrix& LeftEV, FMatrix& EigenValues, FMatrix& RightEV);

			/*!
				Determines the type of definition of a matrix.
				\param A Input matrix.
				\param eigenvalues Optional parameter that returns matrix A eigen values (using the same format as the Eigenvalues function).

				\return Matrix definition type.
			*/
			friend CMATRIXLIB_API MatrixDefinition CheckDefinition(const FMatrix& A, FMatrix *eigenvalues);

			/*!
				Multiplies the matrix by a scalar value defined by B (integer).
				This function is the commutative version of the FMatrix product operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator*(INT A, FMatrix& B);

			//Matrix product by a scalar value
			/*!
				Multiplies the matrix by a scalar value defined by B (double precision).
				This function is the commutative version of the FMatrix product operator.
				NOTE: this function is a specialization for constant matrices.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator*(double A,const FMatrix& B);
			friend CMATRIXLIB_API FMatrix operator*(double A, FMatrix& B);

			/*!
				Multiplies the matrix by a scalar value defined by B (double precision).
				This function is the commutative version of the FMatrix product operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator*(double A, const FMatrix& B);

			/*!
				Multiplies the matrix by a scalar value defined by B (single precision).
				This function is the commutative version of the FMatrix product operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator*(float A, const FMatrix& B);

			/*!
				Element-by-element matrix division of a full matrix with values equal to A (integer) by B.
				This function is the commutative version of the FMatrix Element-by-element matrix division operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator|(INT A,const FMatrix& B);

			/*!
				Element-by-element matrix division of a full matrix with values equal to A (double precision) by B.
				This function is the commutative version of the FMatrix Element-by-element matrix division operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator|(double A,const FMatrix& B);

			/*!
				Sums the matrix with a matrix with the entries equal to A.
				This function is the commutative version of the FMatrix addition operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator+(INT A, const FMatrix& B);

			/*!
				Sums the matrix with a matrix with the entries equal to A.
				This function is the commutative version of the FMatrix addition operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator+(double A,const FMatrix& B);

			/*!
				Subtracts the matrix with a matrix with the entries equal to A.
				This function is the commutative version of the FMatrix subtraction operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator-(INT A, const FMatrix& B);

			/*!
				Subtracts the matrix with a matrix with the entries equal to A.
				This function is the commutative version of the FMatrix subtraction operator.

				\param A Value of the scalar.
				\param B Matrix.
			*/
			friend CMATRIXLIB_API FMatrix operator-(double A, const FMatrix& B);

			/*!
				Calculates the transpose of a matrix.
				\param A Input matrix.
				\return Transpose of A.
			*/
			friend CMATRIXLIB_API FMatrix T(const FMatrix& A);						

			/*!
				Element-wise sine of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix sin(const FMatrix& _X);

			/*!
				Element-wise cosine of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix cos(const FMatrix& _X);

			/*!
				Element-wise tangent of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix tan(const FMatrix& _X);

			/*!
				Element-wise arc-sine of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix asin(FMatrix& _X);

			/*!
				Element-wise arc-cosine of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix acos(FMatrix& _X);

			/*!
				Element-wise arc-tangent of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix atan(FMatrix& _X);

			/*!
				Element-wise arc-tangent 2 of a matrix.
				\param _Y "Top" matrix
				\param _X "Bottom" matrix
			*/
			friend CMATRIXLIB_API FMatrix atan2(const FMatrix& _Y,const FMatrix& _X);

			/*!
				Element-wise hyperbolic tangent of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix tanh(FMatrix& _X);

			/*!
				Element-wise absolute of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix abs(const FMatrix& _X);


			/*!
				Element-wise exponential of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix exp(FMatrix& _X);


			/*!
				Element-wise natural logarithm of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix log(FMatrix& _X);


			/*!
				Element-wise logistic sigma of a matrix.
				\param _X Input matrix
			*/
			friend CMATRIXLIB_API FMatrix logsigma(FMatrix& _X);


			/*!
				Element-wise square root of a matrix.
				\param _X Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix sqrt(FMatrix& _X);


			/*!
				Element-wise hyperbolic cosine of a matrix.
				\param _X Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix cosh(const FMatrix& _X);

			/*!
				Element-wise hyperbolic sine of a matrix.
				\param _X Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix sinh(const FMatrix& _X);


			/*!
				Calculates the root mean square value of a matrix.
				\param _X Input matrix.
			*/
			friend CMATRIXLIB_API double rms(const FMatrix& _X);

			/*!
				Obtains the value of the largest element of a matrix.
				\param _X Input matrix.
			*/
			friend CMATRIXLIB_API double Max(const FMatrix& _X);

			/*!
				Obtains the value of the largest element of each column or row of a matrix.
				\param _X Input matrix.
				\param byColumns Selects if the vector of maximums is calculated column (true) or row-wise (false).
			*/
			friend CMATRIXLIB_API FMatrix Max(const FMatrix& _X, bool byColumns);

			/*!
				Obtains the value of the smallest element of a matrix.
				\param _X Input matrix.
			*/
			friend CMATRIXLIB_API double Min(const FMatrix& _X);

			/*!
				Obtains the value of the smallest element of each column or row of a matrix.
				\param _X Input matrix.
				\param byColumns Selects if the vector of minimums is calculated column (true) or row-wise (false).
			*/
			friend CMATRIXLIB_API FMatrix Min(const FMatrix& _X, bool byColumns);

			/*!
				Computes the screw matrix of a three-dimensional vector.
				\param A Input vector
			*/
			friend CMATRIXLIB_API FMatrix Screw(const FMatrix& A);

			/*!
				Computes the three-dimensional vector from a screw matrix.
				\param A Screw matrix.
			*/
			friend CMATRIXLIB_API FMatrix UnScrew(const FMatrix& A);

			/*!
				Computes the dot product of two vectors.
				\param A First vector.
				\param B Second vector.
			*/
			friend CMATRIXLIB_API double Dot(const FMatrix& A,const FMatrix& B) noexcept(false);

			/*!
				Computes the Euclidean norm of a matrix.
				\param A Input matrix.
			*/
			friend CMATRIXLIB_API double Norm(const FMatrix& A);

			/*!
				Normalizes a vector using the Euclidean norm
				\param A Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix Normalize(const FMatrix& A);

			/*!
				Computes the cross product of two three-dimensional vectors.
				\param A First vector.
				\param B Second vector.
			*/
			friend CMATRIXLIB_API FMatrix Cross(const FMatrix& A,const FMatrix& B) noexcept(false);

			/*!
				Matrix string output to be used with cout or any ostream derived class.
				\param in Stream sequence.
				\param Mat FMatrix object to be printed.
			*/
			friend CMATRIXLIB_API ostream& operator<<(ostream& in,FMatrix &Mat);

			/*!
				Matrix string output to be used with fstream derived class.
				NOTE: this function is equivalent to the SaveToFile but enables multiple saves.

				\param out Ofstream sequence.
				\param Mat FMatrix object to be printed.
			*/
			friend CMATRIXLIB_API ofstream& operator<<(ofstream& out, FMatrix &Mat);

			/*!
				Matrix string output to be used with fstream derived class.
				NOTE: this function is equivalent to the SaveToFile but enables multiple saves.

				\param out Ofstream sequence.
				\param Mat FMatrix object to be printed.
			*/
			friend CMATRIXLIB_API wofstream& operator<<(wofstream& out,FMatrix &Mat);

			/*!
				Matrix string input to be used with fstream derived class.
				NOTE: this function is equivalent to the LoadFromFile but enables multiple loads.

				\param in Ifstream sequence.
				\param Mat FMatrix object to be printed.
			*/
			friend CMATRIXLIB_API ifstream& operator>>(ifstream& in, FMatrix &Mat);

			/*!
				Matrix string input to be used with fstream derived class.
				NOTE: this function is equivalent to the LoadFromFile but enables multiple loads.

				\param in Ifstream sequence.
				\param Mat FMatrix object to be printed.
			*/
			friend CMATRIXLIB_API wifstream& operator>>(wifstream& in, FMatrix &Mat);

			/*!
				Calculates the Row and Columns Equilibrating matrices -> Ac = R*A*C
				Tries to equilibrate (reduce the condition number) an ill-conditioned matrix by rescaling the matrix entries to values as close to 1 as possible.
				NOTE: if R and/or C are empty the matrix does not need row and/or column equilibration

				\param A Matrix to be calibrated.
				\param R Row calibration matrix.
				\param C Column calibration matrix.
				\param PowerRadix Uses the matrix radix to equilibrate the matrix, making the matrix entries between sqrt(radix) and 1/sqrt(radix), instead of approximating them to 1 (set by default). 

			*/
			friend CMATRIXLIB_API void EquilibrateMatrix(const FMatrix& A,FMatrix& R,FMatrix& C, bool PowerRadix = true);

			/*!
				This function solves any type of system of linear equations (regular, under- or over-constrained).
				For under- or over-constrained systems, the given solution is an approximation calculated by the minimum squared error.

				The function will automatically choose the best method to solve the system (unless forced by the Options) and apply optimizations to special matrices in order to decrease computation times.

				NOTE: is the system is impossible to solve, the result will be an empty FMatrix object.

				\param A Sytem Matrix. Can be any invertible (or pseudo-invertible) matrix.
				\param B Right-hand side of the system. This parameter can be a vector or matrix (for multiple system solutions) with the same rows as the system matrix A.
				\param Options Special options to be used by the function. The options can be one or more conjugated SolveOptions entries.

				\return Solution of the system of equations (if succeeded, otherwise returns an empty FMatrix object).
			*/
			friend CMATRIXLIB_API FMatrix Solve(const FMatrix& A,const FMatrix& B, unsigned INT Options = None);

			/*!
				Uses the Successive over-relaxation method (SOR) to solve a system of linear equations.
				Although the SOR method can be applied to any square system, the convergence can only be guaranteed for diagonally dominant and positive-definite systems.

				\param A Non-singular system matrix.
				\param B Right-hand side vector.
				\param X Initial estimate of the solution. It is overwritten by the final solution.
			*/
			friend CMATRIXLIB_API void SORSolve(const FMatrix& A, const FMatrix& B, FMatrix& X);

			/*!
				Uses the Conjugated gradient method to solve a symmetric positive-definite system of linear equations.
				This iterative method can be used as an alternative to direct methods (such as the Cholesky decomposition) for large systems.
				
				\param A Non-singular system matrix.
				\param B Right-hand side vector.
				\param X Initial estimate of the solution. It is overwritten by the final solution.
			*/
			friend CMATRIXLIB_API void ConjugateGradientSolve(const FMatrix& A, const FMatrix& B, FMatrix& X);

			/*!
				Reshapes the matrix A to match the specified new line and column size, provided that the number of elements remains the same.
				\param A Matrix to be reshaped.
				\param NewNi New number of rows of the matrix.
				\param NewNj New number of columns of the matrix.
			 */
			friend CMATRIXLIB_API FMatrix Reshape(const FMatrix& A,INT NewNi, INT NewNj);

			/*!
				Obtains a submatrix defined by beg_line, end_line, beg_column and end_column.
				\param A Input matrix.
				\param beg_line Index of the first row to be selected (zero-based).
				\param end_line Index of the last row to be selected (zero-based).
				\param beg_col Index of the first column to be selected (zero-based).
				\param end_col Index of the last column to be selected (zero-based).

				\return Submatrix
			 */
			friend CMATRIXLIB_API FMatrix Range(const FMatrix& A, INT beg_line, INT end_line,
												INT beg_col, INT end_col);

			/*!
				Reverses the order of the rows (or columns) of a matrix.
				\param A Matrix to be reversed.
				\param Columns Set to reverse the columns, reset to reverse the rows (reset by default).
			*/
			friend CMATRIXLIB_API FMatrix Reverse(const FMatrix& A, bool Columns = false);

			/*!
				Vertical concatenation of matrix A and B.
				NOTE: The A and B matrices must have the same number of columns.

				\param A Top matrix.
				\param B Bottom matrix.

				\return Concatenated matrix
			*/
			friend CMATRIXLIB_API FMatrix VCat(const FMatrix& A,const FMatrix& B) noexcept(false);

			/*!
				Horizontal concatenation of matrix A and B.
				NOTE: The A and B matrices must have the same number of rows.

				\param A Left matrix.
				\param B Right matrix.

				\return Concatenated matrix
			*/
			friend CMATRIXLIB_API FMatrix HCat(const FMatrix& A,const FMatrix& B) noexcept(false);

			/*!
				Sum of all elements of matrix A.
				\param A Input matrix
				\return double precision value of the sum of matrix entries.
			*/
			friend CMATRIXLIB_API double Sum(const FMatrix& A);

			/*!
				Sum of all rows or columns  of matrix A.
				\param A Input matrix
				\param Rows Set to sum the rows, reset to sum the columns.
				\return Vector with the sum of the rows (row vector) or columns (column vector) of the matrix.
			*/
			friend CMATRIXLIB_API FMatrix Sum(const FMatrix& A, bool Rows);

			/*!
				Product of all elements of A.
				\param A Input matrix
				\return double precision value of the product of matrix entries.
			*/
			friend CMATRIXLIB_API double Product(const FMatrix& A);

			/*!
				Converts a scalar FMatrix into a double precision value.
				NOTE: for non scalar FMatrix, this method returns the first element.

				\param A Input FMatrix object.

				\return Converted value.
			*/
			friend CMATRIXLIB_API double ToDouble(const FMatrix& A) noexcept(false);

			/*!
				Converts the matrix into a string using the same file format as the SaveToFile method.

				\return string object containing the matrix
			*/
			friend CMATRIXLIB_API char* ToString(const FMatrix& A);

			/*!
				Returns a read-only pointer to the data stored in the matrix.
				\return Constant pointer to the matrix data.
			*/
			friend CMATRIXLIB_API const double* ToDoublePtr(const FMatrix& A);

			/*!
				Creates a scalar FMatrix object from a given value.
				\param val Value of the scalar FMatrix object.

				\return Scalar FMatrix object
			*/
			friend CMATRIXLIB_API FMatrix FromDouble(double val);

			/*!
				Calculates the mean value of the entries of a matrix.
				\param A Input Matrix

				\return Mean value.
			*/
			friend CMATRIXLIB_API double Mean(const FMatrix& A);

			/*!
				Calculates the mean value of the rows or columns of a matrix.

				\param A Input Matrix
				\param Rows Set to get the mean of the rows, reset to obtain the mean of the columns 

				\return Vector with the mean of the rows (row vector) or columns (column vector) of the matrix.
			*/
			friend CMATRIXLIB_API FMatrix Mean(const FMatrix& A,bool Rows);

			/*!
				Calculates the power of a matrix (element- or matrix-wise).
				NOTE: for a matrix-wise operation, this function is equivalent to the _X^_Y method.
				\param _X Base matrix.
				\param _Y Integer exponent.
				\param elementwise Set to calculate the power of each element (can be used with any type of FMatrix), reset to calculate the power of the matrix (for square matrices only). The parameter is reset by default.
			*/
			friend CMATRIXLIB_API FMatrix pow(FMatrix _X, INT _Y, bool elementwise = false);

			/*!
				Resamples a vector keeping only the values with the index given by the increment.

				\param A Input vector.
				\param increment Increment of the selected indices.
			*/
			friend CMATRIXLIB_API FMatrix Pack(const FMatrix& A, INT increment);

			/*!
				Calculates the differential of a matrix or column vector by subtracting two adjoining rows.
				NOTE: the resulting vector/matrix has one less row than the original

				\param A Input matrix.
				\return Differential of the matrix/vector.
			*/
			friend CMATRIXLIB_API FMatrix Diff(const FMatrix& A);

			/*!
				Creates a large matrix consisting of an n-by-m tiling of copies of A.

				\param A Matrix to be repeated
				\param n Number of repetitions row-wise.
				\param m Number of repetitions column-wise.

				\return Repeated matrix
			*/
			friend CMATRIXLIB_API FMatrix RepMat(const FMatrix& A,unsigned INT n, unsigned INT m);

			/*!
				Creates a diagonal matrix with the entries of a vector.

				\param A Input vector.

				\return Square diagonal matrix with the dimensions of the input vector
			*/
			friend CMATRIXLIB_API FMatrix VectorToDiagonalMatrix(const FMatrix& A);

			/*!
				Element-wise floor of a matrix.
				\param A Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix floor(const FMatrix& A);

			/*!
				Element-wise ceiling of a matrix.
				\param A Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix ceil(const FMatrix& A);

			/*!
				Element-wise round of a matrix.
				\param A Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix round(const FMatrix& A);

			/*!
				Element-wise truncature of a matrix.
				\param A Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix trunc(const FMatrix& A);

			/*!
				Element-wise sign of a matrix.
				\param A Input matrix.
			*/
			friend CMATRIXLIB_API FMatrix sign(const FMatrix& _X);
			//
			//----------------------------------------------------------------------
	};

	//!Creates a square Identity matrix with N rows
	/*!
		\param N Number of rows/columns.
	*/
	CMATRIXLIB_API FMatrix Eye(INT N);

	//!Creates a square zero matrix with N rows
	/*!
		\param N Number of rows/columns.
	*/
	CMATRIXLIB_API FMatrix Zero(INT N);

	//!Creates a rectangular zero matrix with Ni rows and Nj columns
	/*!
		\param Ni Number of rows.
		\param Nj Number of columns.
	*/
	CMATRIXLIB_API FMatrix Zero(INT Ni, INT Nj);


	CMATRIXLIB_API FMatrix Ones(INT Ni, INT Nj);

	//!Creates a rectangular matrix with the entries given by the values in the ellipsis (column-wise)
	/*!
		\param Ni Number of rows.
		\param Nj Number of columns.
	*/
	CMATRIXLIB_API FMatrix Matrix(INT Ni, INT Nj, ...);

	//!Creates a homogeneous translation matrix
	/*!
		\param X Translation amount in X axis.
		\param Y Translation amount in Y axis.
		\param Z Translation amount in Z axis.
	*/
	CMATRIXLIB_API FMatrix HomogeneousTranslation(double X, double Y, double Z);

	//!Creates a homogeneous scaling matrix
	/*!
		\param X Scaling amount in X axis.
		\param Y Scaling amount in Y axis.
		\param Z Scaling amount in Z axis.
	*/
	CMATRIXLIB_API FMatrix HomogeneousScale(double sX, double sY, double sZ);

	//!Creates a homogeneous rotation about X matrix
	/*!
		\param angle Angle to rotate.
	*/
	CMATRIXLIB_API FMatrix HomogeneousRotationX(double angle);

	//!Creates a homogeneous rotation about Y matrix
	/*!
		\param angle Angle to rotate.
	*/
	CMATRIXLIB_API FMatrix HomogeneousRotationY(double angle);

	//!Creates a homogeneous rotation about Z matrix
	/*!
		\param angle Angle to rotate.
	*/
	CMATRIXLIB_API FMatrix HomogeneousRotationZ(double angle);

	CMATRIXLIB_API FMatrix FromCompressedString(const string& data);

	//!An empty FMatrix object
	extern CMATRIXLIB_API const FMatrix EmptyMatrix;

	#pragma endregion

	#pragma region BMatrix

	//! Class for managing matrices of boolean values.
	/*!
		This class creates and destroys matrices of doubles	and provides a more advanced and specialized set matrix manipulation functions for boolean matrices.
	 */
	class CMATRIXLIB_API BMatrix
	{
		private:
			INT _Ni;		//!< Number of rows. 
			INT _Nj;		//!< Number of columns.

			INT *_dims[2];	//!< Helper member for the VS2012 visualizer

			bool *_data;	//!< Matrix values.
#ifdef _WIN32
		public:
			__declspec(property(get=GetNRows)) INT NRows;							//!< Number of rows of the matrix (property).
			__declspec(property(get=GetNColumns)) INT NColumns;						//!< Number of columns of the matrix (property).
			__declspec(property(get=GetLength)) INT Length;							//!< Number of elements in the matrix (property).

			_declspec(property(get=GetColumn,put=ReplaceColumn)) BMatrix Column[];	//!< Returns a determined column or a span of rows of the matrix (property).
			_declspec(property(get=GetLine,put=ReplaceLine)) BMatrix Line[];		//!< Returns a determined row or a span of rows of the matrix (property).
			_declspec(property(get=GetRange,put=SetRange)) BMatrix Range[][][][];	//!< Returns a sub-matrix (property).
#endif
		public:
			//---Constructors-------------------------------------------------------
			//
			//! Default contructor.
			/*!	
				Creates an empty BMatrix to be initialized afterward by Initialize or by the atribution operator.
			 */
			BMatrix();

			//! Creates a false Ni by Nj matrix.
			/*!
				Creates a false valued matrix with the determined (Ni and Nj) dimensions.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
			 */
			BMatrix(INT Ni, INT Nj);

			//! Creates a Ni by Nj matrix and fills it with values from data.
			/*!
				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param data Data array to fill the matrix. The dimensions of the array must match the length of the created matrix.
						    Extra elements in the array will be ignored and missing elements will be filled with false.
			 */
			BMatrix(INT Ni, INT Nj, const bool* data);

			//! Creates a Ni by Nj matrix.
			/*!
				Creates a Ni by Nj matrix and fills all or just the diagonal elements with a predetermined value.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param val Value to fill the matrix elements with.
				\param diag_only Chooses between filling only the diagonal (true) or all elements (false).
			 */
			BMatrix(INT Ni, INT Nj, bool val, bool diag_only = true);

			//! Copies a matrix.
			/*!
				Creates a matrix copied from an existing BMatrix.

				\param Mat Original matrix to be copied.
			 */
			BMatrix(const BMatrix& Mat);

			//! Copies a matrix.
			/*!
				Creates a matrix copied from an existing FMatrix.
				NOTE: it is considered a false value every entry in FMatrix smaller than 1.

				\param Mat Original matrix to be copied.
			 */
			BMatrix(const FMatrix& Mat);
			//
			//----------------------------------------------------------------------

			//---Destructor---------------------------------------------------------
			//
			//! Destructor method
			~BMatrix();
			//
			//----------------------------------------------------------------------

			//---Public Methods-----------------------------------------------------
			//

			//! Initializes a matrix.
			/*!
				Initializes an empty Matrix or reinitializes an existing one with the determined dimensions.

				\param Ni Number of rows of the matrix.
				\param Nj Number of columns of the matrix.
			 */
			void Initialize(INT Ni, INT Nj);

			//! Initializes a matrix.
			/*!
				Initializes an empty BMatrix or reinitializes an existing one with the determined dimensions and fills it with data.

				\param Ni Number of rows of the matrix.
				\param Nj Number of columns of the matrix.
				\param data Data array to fill the matrix. The dimensions of the array must match the length of the created matrix.
						    Extra elements in the array will be ignored and a shorted array will lead to unpredictable results.
			 */
			void Initialize(INT Ni, INT Nj, const bool* data);

			//! Initializes a matrix.
			/*!
				Initializes an empty BMatrix or reinitializes an existing one with the determined dimensions
				and fills all or just the diagonal elements with a predetermined value.

				\param Ni Number of rows of the new matrix.
				\param Nj Number of columns of the new matrix.
				\param val Value to fill the matrix elements with.
				\param diag_only Chooses between filling only the diagonal (true) or all elements (false).
			 */
			void Initialize(INT Ni, INT Nj, bool val, bool diag_only = true);

			//! Initializes a matrix.
			/*!
				Initializes an empty BMatrix or reinitializes an existing one with the dimensions and values of another BMatrix.

				\param Mat Original matrix to be copied.
			 */
			void Initialize(const BMatrix& Mat);

			//! Initializes a matrix.
			/*!
				Initializes an empty BMatrix or reinitializes an existing one with the dimensions and values of a FMatrix.
				NOTE: it is considered a false value every entry in FMatrix smaller than 1.

				\param Mat Original matrix to be copied.
			 */
			void Initialize(const FMatrix& Mat);

			//! Clear the matrix
			/*!
				Clears the matrix, transforming it into an empty BMatrix that can be reinitialized.
			 */
			void Clear();

			//! Checks if the matrix has zero dimensions (empty matrix)
			/*!
				\return True if it is an empty BMatrix, false otherwise.
			 */
			bool IsEmpty() const;

			//! Gets number of row of the matrix.
			/*!
				\return Number of rows.
			 */
			inline INT GetNRows() const {return _Ni;};

			//! Gets number of columns of the matrix.
			/*!
				\return Number of columns.
			 */
			inline INT GetNColumns() const {return _Nj;};

			//! Gets number of elements of the matrix.
			/*!
				\return Number of elements.
			 */
			inline INT GetLength() const {return _Ni*_Nj;};

			//!Replaces the specified column with the values of a vector with the same number of elements
			/*!
				\param j Index of the column to be selected (zero-based).
				\param NewCol Vector to be copied into the column.
			 */
			void ReplaceColumn(INT j,const BMatrix& NewCol) noexcept(false);

			//!Replaces the specified row with the values of a vector with the same number of elements
			/*!
				\param i Index of the row to be selected (zero-based).
				\param NewLine Vector to be copied into the row.
			 */
			void ReplaceLine(INT i,const BMatrix& NewLine) noexcept(false);

			//!Replaces the specified column range with the values of a matrix with equivalent dimensions
			/*!
				\param j0 Index of the first column to be selected (zero-based).
				\param j1 Index of the last column to be selected (zero-based).
				\param NewCol Matrix to be copied into the column range.
			 */
			void ReplaceColumn(INT j0, INT j1, const BMatrix& NewCol) noexcept(false);

			//!Replaces the specified range of rows with the values of a matrix with equivalent dimensions
			/*!
				\param i0 Index of the first row to be selected (zero-based).
				\param i1 Index of the last row to be selected (zero-based).
				\param NewLine Matrix to be copied into the row range.
			 */
			void ReplaceLine(INT i0, INT i1, const BMatrix& NewLine) noexcept(false);

			//!Gets the specified column
			/*!
				\param j Index of the column to be selected (zero-based).

				\return Selected column
			 */
			BMatrix GetColumn(INT j) const;

			//!Gets the specified row
			/*!
				\param i Index of the row to be selected (zero-based).

				\return Selected row
			 */
			BMatrix GetLine(INT i) const;

			//!Gets the specified column range
			/*!
				\param j0 Index of the first column to be selected (zero-based).
				\param j1 Index of the last column to be selected (zero-based).
			
				\return Selected column range.
			 */
			BMatrix GetColumn(INT j0, INT j1) const;

			//!Gets the specified row range
			/*!
				\param i0 Index of the first row to be selected (zero-based).
				\param i1 Index of the last row to be selected (zero-based).
			
				\return Selected column row.
			 */
			BMatrix GetLine(INT i0, INT i1) const;


			//!Gets the submatrix defined by beg_line, end_line, beg_column and end_column
			/*!
				\param beg_line Index of the first row to be selected (zero-based).
				\param end_line Index of the last row to be selected (zero-based).
				\param beg_col Index of the first column to be selected (zero-based).
				\param end_col Index of the last column to be selected (zero-based).

				\return Submatrix
			 */
			BMatrix GetRange(INT beg_line, INT end_line, INT beg_col, INT end_col) const;

			//!Sets the values the submatrix defined by beg_line, end_line, beg_column and end_column with the values of another matrix (with equal dimensions)
			/*!
				\param beg_line Index of the first row to be selected (zero-based).
				\param end_line Index of the last row to be selected (zero-based).
				\param beg_col Index of the first column to be selected (zero-based).
				\param end_col Index of the last column to be selected (zero-based).
				\param newMat Matrix to be copied into the submatrix.
			 */
			void SetRange(INT beg_line, INT end_line, INT beg_col, INT end_col,const BMatrix& newMat);

			//!Reshapes the matrix to match the specified new line and column sizes, provided that the number of elements remains the same.
			/*!
				\param NewNi New number of rows of the matrix.
				\param NewNj New number of columns of the matrix.
			 */
			void Reshape(INT NewNi, INT NewNj) noexcept(false);	

			//!Gets or sets the (i,j) matrix entry
			/*!
				\param i Zero-based row index.
				\param j Zero-based column index.

			*/
			bool& operator() (INT i, INT j) noexcept(false);

			//!Gets the (i,j) matrix entry
			/*!
				\param i Zero-based row index.
				\param j Zero-based column index.

			*/
			bool operator() (INT i, INT j) const noexcept(false);

			//Gets or sets the i entry of the vector/matrix (on vector form)
			/*!
				\param i Zero-based index.
			*/
			bool& operator() (INT i) noexcept(false);

			//Gets the i entry of the vector/matrix (on vector form)
			/*!
				\param i Zero-based index.
			*/
			bool operator() (INT i) const noexcept(false);

			//! AND operation between two BMatrices
			/*!
				Performs a AND operation between two matrices, where each entry of the new matrix is the AND of the operands corresponding entries.

				\param B Second matrix in the operation
			*/
			BMatrix operator&& (const BMatrix &B);

			//! OR operation between two BMatrices
			/*!
				Performs a OR operation between two matrices, where each entry of the new matrix is the OR of the operands corresponding entries.

				\param B Second matrix in the operation
			*/
			BMatrix operator|| (const BMatrix &B);

			//!Returns of the neggation of the matrix.
			BMatrix operator!() const;

			//!Conversion form a boolean array.
			/*!
				Converts a boolean array pointer to a BMatrix object
				NOTE: data must have at least NixNj elements

				\param data Pointer to the boolean array.
			*/
			BMatrix& operator= (const bool* data);

			//!Copies the values of a boolean CMatrix object.
			/*!
				\param Mat Original matrix.
			*/
			BMatrix& operator= (const CMatrix<bool>& Mat);

			//!Copies the values of another BMatrix object.
			/*!
				\param Mat Original matrix.
			*/
			BMatrix& operator= (const BMatrix &Mat);

			//! Copies the values of an FMatrix object.
			/*!
				Initializes an empty BMatrix or reinitializes an existing one with the dimensions and values of a FMatrix.
				NOTE: it is considered a false value every entry in FMatrix smaller than 1.

				\param Mat Original matrix to be copied.
			 */
			BMatrix& operator= (const FMatrix &Mat);

			//!Convertion to an FMatrix object
			/*!
				Converts this BMatrix object into an FMatrix object whit 0 for false and 1 for true values.
			*/
			operator FMatrix() const;

			//!Convertion to a boolean CMatrix object
			/*!
				Converts this BMatrix object into a boolean CMatrix object.
			*/
			operator CMatrix<bool>() const;

			operator bool() const;

		private:

			void PrivateInitialization(INT Ni, INT Nj);
			void PrivateInitialization(INT Ni, INT Nj, const bool* data);
			void PrivateInitialization(INT Ni, INT Nj, bool val, bool diag_only = true);
			void PrivateInitialization(const FMatrix& Mat);
		
			void PrivateCopy(const BMatrix& Mat);

		public:

			/*!
				Determines if the FMatrix is smaller than B, element-wise.

				\param A Matrix.
				\param B Double precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator<(const FMatrix& A, double B);

			/*!
				Determines if the FMatrix is smaller than B, element-wise.

				\param A Matrix.
				\param B Single precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator<(const FMatrix& A, float B);

			/*!
				Determines if the FMatrix A is smaller than the FMatrix B, element-wise.

				\param A Left matrix.
				\param B Right matrix.
			*/
			friend CMATRIXLIB_API BMatrix operator<(const FMatrix& A, const FMatrix& B);

			/*!
				Determines if the FMatrix is smaller or equal than B, element-wise.

				\param A Matrix.
				\param B Double precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator<=(const FMatrix& A, double B);

			/*!
				Determines if the FMatrix is smaller or equal than B, element-wise.

				\param A Matrix.
				\param B Single precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator<=(const FMatrix& A, float B);

			/*!
				Determines if the FMatrix A is smaller or equal than the FMatrix B, element-wise.

				\param A Left matrix.
				\param B Right matrix.
			*/
			friend CMATRIXLIB_API BMatrix operator<=(const FMatrix& A, const FMatrix& B);

			/*!
				Determines if the FMatrix is greater than B, element-wise.

				\param A Matrix.
				\param B Double precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator>(const FMatrix& A, double B);

			/*!
				Determines if the FMatrix is greater than B, element-wise.

				\param A Matrix.
				\param B Single precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator>(const FMatrix& A, float B);

			/*!
				Determines if the FMatrix A is greater than the FMatrix B, element-wise.

				\param A Left matrix.
				\param B Right matrix.
			*/
			friend CMATRIXLIB_API BMatrix operator>(const FMatrix& A, const FMatrix& B);

			/*!
				Determines if the FMatrix is greater or equal than B, element-wise.

				\param A Matrix.
				\param B Double precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator>=(const FMatrix& A, double B);

			/*!
				Determines if the FMatrix is greater or equal than B, element-wise.

				\param A Matrix.
				\param B Single precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator>=(const FMatrix& A, float B);

			/*!
				Determines if the FMatrix A is greater or equal than the FMatrix B, element-wise.

				\param A Left matrix.
				\param B Right matrix.
			*/
			friend CMATRIXLIB_API BMatrix operator>=(const FMatrix& A, const FMatrix& B);

			/*!
				Determines if the FMatrix is equal to B, element-wise.

				\param A Matrix.
				\param B Double precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator==(const FMatrix& A, double B);

			/*!
				Determines if the FMatrix is equal to B, element-wise.

				\param A Matrix.
				\param B Single precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator==(const FMatrix& A, float B);

			/*!
				Determines if the FMatrix A is equal to the FMatrix B, element-wise.

				\param A Left matrix.
				\param B Right matrix.
			*/
			friend CMATRIXLIB_API BMatrix operator==(const FMatrix& A, const FMatrix& B);

			/*!
				Determines if the FMatrix is different than B, element-wise.

				\param A Matrix.
				\param B Double precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator!=(const FMatrix& A, double B);

			/*!
				Determines if the FMatrix is different than B, element-wise.

				\param A Matrix.
				\param B Single precision value.
			*/
			friend CMATRIXLIB_API BMatrix operator!=(const FMatrix& A, float B);

			/*!
				Determines if the FMatrix A is different than the FMatrix B, element-wise.

				\param A Left matrix.
				\param B Right matrix.
			*/
			friend CMATRIXLIB_API BMatrix operator!=(const FMatrix& A, const FMatrix& B);

			//!Checks if at least one entry of the BMatrix A is true
			/*!
				\param A BMatrix to test.
			*/
			friend CMATRIXLIB_API bool Any(const BMatrix& A);

			//!Checks if all the entries of the BMatrix A is true
			/*!
				\param A BMatrix to test.
			*/
			friend CMATRIXLIB_API bool All(const BMatrix& A);

			//!Finds the indices of the BMatrix A that are true
			/*!
				\param A BMatrix to be searched.
				\return Vector of indices corresponding to the true entries.
			*/
			friend CMATRIXLIB_API CMatrix<unsigned INT> Find(const BMatrix& A);


			/*!
				Matrix string output to be used with fstream derived class.
				NOTE: this function is equivalent to the SaveToFile but enables multiple saves.

				\param out Ofstream sequence.
				\param Mat BMatrix object to be printed.
			*/
			friend CMATRIXLIB_API wofstream& operator<<(wofstream& out,BMatrix &Mat);

			/*!
				Matrix string output to be used with fstream derived class.
				NOTE: this function is equivalent to the LoadFromFile.

				\param out Ofstream sequence.
				\param Mat BMatrix object to be scanned.
			*/
			friend CMATRIXLIB_API wifstream& operator>>(wifstream& in,BMatrix &Mat);
	};
#ifdef _WIN32
	#pragma warning(default : 4995)
#endif
	//!An empty BMatrix object
	extern CMATRIXLIB_API const BMatrix EmptyBMatrix;

	#pragma endregion

#pragma region RMAtrix
	class CMATRIXLIB_API RMatrix
	{
		friend class FMatrix;

	protected:
		unsigned INT _NRows;
		unsigned INT _NColumns;
		unsigned INT _length;

		FMatrix* _orig;

		unsigned INT _startR, _startC;

#ifdef _WIN32
	public:
		__declspec(property(get = GetNRows)) unsigned INT NRows;		//!< Number of rows of the matrix (property).
		__declspec(property(get = GetNColumns)) unsigned INT NColumns;	//!< Number of columns of the matrix (property).
		__declspec(property(get = GetLength)) unsigned INT Length;		//!< Number of elements in the matrix (property).
#endif
	public:
		RMatrix();
		RMatrix(FMatrix& Mat);
		RMatrix(const RMatrix& RMat);

		~RMatrix();

		void Clear();

		//! Gets number of row of the matrix.
		/*!
		\return Number of rows.
		*/
		inline unsigned INT GetNRows() const { return _NRows; };

		//! Gets number of columns of the matrix.
		/*!
		\return Number of columns.
		*/
		inline unsigned INT GetNColumns() const { return _NColumns; };

		//! Gets number of elements of the matrix.
		/*!
		\return Number of elements.
		*/
		inline unsigned INT GetLength() const { return _length; };

		void CopyInto(const FMatrix& Data);

	protected:
		RMatrix(FMatrix* Orig, unsigned INT NRows, unsigned INT NColumns, unsigned INT StartR, unsigned INT StartC);

		void PrivateCopy(FMatrix& Mat);
		void PrivateCopy(const RMatrix& RMat);

	public:

		//!Gets or sets the (i,j) matrix entry
		/*!
		\param i Zero-based row index.
		\param j Zero-based column index.

		*/
		double& operator() (unsigned INT i, unsigned INT j) noexcept(false);

		//!Gets the (i,j) matrix entry
		/*!
		\param i Zero-based row index.
		\param j Zero-based column index.

		*/
		double operator() (unsigned INT i, unsigned INT j) const noexcept(false);

		//Gets or sets the i entry of the vector/matrix (on vector form)
		/*!
		\param i Zero-based index.
		*/
		double& operator() (unsigned INT i) noexcept(false);

		//Gets the i entry of the vector/matrix (on vector form)
		/*!
		\param i Zero-based index.
		*/
		double operator() (unsigned INT i) const noexcept(false);

		RMatrix & operator=(FMatrix& Mat);
		RMatrix& operator=(const RMatrix& RMat);

		operator FMatrix() const;
	};
#pragma endregion //RMAtrix

	#if defined _DEBUG && !defined DUMP
		//!Memory leak detection (Must be at the beginning of the program)
		CMATRIXLIB_API void DumpMemoryLeaks();		

		#define DUMP
	#endif
}

#endif

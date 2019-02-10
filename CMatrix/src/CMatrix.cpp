/*
 * Copyright (c) André Carvalho
 */
#include <stdafx.h>
#include <CMatrix.h>

#ifndef __GNUC__
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#ifdef _DEBUG
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif

#pragma warning(disable : 4995)
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

//Custom function replacements
#ifndef __GNUC__

#else

#endif // __GNUC__
//


#ifndef _WIN32
char *strrev(char *str)
{
	char *p1, *p2;

	if (!str || !*str)
		return str;
	for (p1 = str, p2 = str + strlen(str) - 1; p2 > p1; ++p1, --p2)
	{
		*p1 ^= *p2;
		*p2 ^= *p1;
		*p1 ^= *p2;
	}
	return str;
}
#endif

#define FREE_DATA {if(_data !=0) free(_data); _data = 0;}
#define FFREE_DATA \
{\
	if(_data !=0)\
		free(_data);\
	_data = 0;\
}

#define FREE(p) \
{\
	if((p) != 0)\
		free((p));\
	(p) = 0;\
}

namespace CMatrixLib
{
	template<>
	void SpecialNew<INT>(INT** ptr, size_t len)
	{		
		*ptr = new INT[len];
	}

	template<>
	void SpecialDelete<INT>(INT** ptr)
	{
		delete[] *ptr;
	}

	void ToCompressedString(unsigned long long val, char* buff,INT len)
	{
		memset(buff,' ',len);

		unsigned long long div = val, rem = 0;

		if(div < 250)
		{
			buff[0] = (char)(div+5);
			buff[1] = 0;
		}
		else
		{
			INT i = 0;
			do
			{
				rem = div%250;
				div = div/250;

				buff[i] = (char)(rem+5);

				i++;
			}while(div > 250);

			buff[i] = (char)(div+5);
			buff[i+1] = 0;
		}

#ifdef _WIN32
		strcpy_s(buff,len*sizeof(char),_strrev(buff));
#else
		strcpy(buff, strrev(buff));
#endif

	}

	unsigned long long FromCompressedString(char* buff, INT n_B_to_read)
	{
		unsigned long long res = 0;

		for(INT i=0;i<n_B_to_read;i++)
		{
			res+= (unsigned long long)(((INT)((unsigned char)buff[i])-5)*std::pow(250.0,n_B_to_read-i-1));
		}

		return res;
	}

	INT sign(double _X)
	{
		if(_X != 0 )
			return (INT)(_X/fabs(_X));
		else
			return 0;
	}

	unsigned INT Factorial(unsigned INT  _X)
	{
		if(_X == 0)
			return 0;
		else if(_X == 1)
			return 1;
		else
			return Factorial(_X-1)*_X;
	}
	
	double Gamma(double _X)
	{
		if(_X == 0)
			return 0;
		else
			return std::sqrt(2*std::acos(-1)/_X)*std::pow(_X/std::exp(1.0),_X);
	}
	double LambertW(double _X, double Tolerance)
	{
		double w = 0, wp = 0;

		do
		{
			wp = w;

			w = wp - (wp*std::exp(wp)-_X)/(std::exp(wp)+wp*std::exp(wp));
		}
		while(fabs(w-wp) > Tolerance);

		return w;
	}
	
	double LFunction(double x)
	{
		double k = 1.461632; // psi(k) = 0

		double c = std::sqrt(2*std::acos(-1))/std::exp(1) - Gamma(k);

		return std::log((x + c)/std::sqrt(2*std::acos(-1)));
	}
	double InverseGamma(double _X)
	{
		return LFunction(_X)/LambertW(LFunction(_X)/std::exp(1),1e-5)-.5;
	}

	void FreeMKLBuffers()
	{
		mkl_thread_free_buffers();
	}

	//Matrix with zero size
	CMATRIXLIB_API const FMatrix EmptyMatrix;


	unsigned INT operator|(SolveOptions A, SolveOptions B)
	{
		return (SolveOptions)((unsigned INT)A | (unsigned INT)B);
	}

	unsigned INT operator&(SolveOptions A, SolveOptions B)
	{
		return (SolveOptions)((unsigned INT)A & (unsigned INT)B);
	}



	//---Constructors-------------------------------------------------------
	//
	FMatrix::FMatrix()
	{
		_Ni=0;
		_Nj=0;
		_data = 0;
		
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		_type = General;
	}
	FMatrix::FMatrix(unsigned INT Ni, unsigned INT Nj)
	{
		_data = (double*)calloc((size_t)Ni*(size_t)Nj,sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}

		_Ni = Ni;
		_Nj = Nj;

		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		_type = General;
	}
	FMatrix::FMatrix(const FMatrix &Mat)
	{
		_Ni = Mat._Ni;
		_Nj = Mat._Nj;

		if(_Ni == 0 || _Nj == 0)
		{
			_data = 0;
			return;
		}

		_data = (double*)malloc((size_t)_Ni*(size_t)_Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}
		
		memcpy_s(_data,(size_t)_Ni*(size_t)_Nj*sizeof(double),Mat._data,(size_t)Mat._Ni*(size_t)Mat._Nj*sizeof(double));
	
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		_type = Mat._type;
	}

	FMatrix::FMatrix(unsigned INT Ni, unsigned INT Nj, const double* data)
	{
		_data = (double*)malloc((size_t)Ni*(size_t)Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}

		memcpy_s(_data,(size_t)Ni*(size_t)Nj*sizeof(double),data,(size_t)Ni*(size_t)Nj*sizeof(double));

		_Ni = Ni;
		_Nj = Nj;

		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		_type = General;
	}

	FMatrix::FMatrix(unsigned INT Ni, unsigned INT Nj, double val, bool diag_only)
	{
		_data = (double*)malloc((size_t)Ni*(size_t)Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}

		_Ni = Ni;
		_Nj = Nj;

		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		if(diag_only)
		{
			memset(_data,0,(size_t)Ni*(size_t)Nj*sizeof(double));
			for(unsigned INT i=0;i<Ni;i++) _data[(size_t)i+(size_t)Ni*(size_t)i] = val;

			_type = Diagonal;
		}
		else
		{
			for(size_t i=0;i<(size_t)Ni*(size_t)Nj;i++) _data[i] = val;

			_type = General;
		}
	}

	FMatrix::FMatrix(unsigned INT Ni, unsigned INT Nj, double min, double max, bool diag_only)
	{
		_data = (double*)malloc((size_t)Ni*(size_t)Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}

		_Ni = Ni;
		_Nj = Nj;

		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		memset(_data,0,(size_t)Ni*(size_t)Nj*sizeof(double));

		unsigned INT seed;
#ifndef __GNUC__
	#ifdef _WIN64
		__timeb64 t;
		_ftime64_s(&t);
	
		seed = (unsigned INT)(t.millitm) * (unsigned INT)(this)*rand();
	#else
		__timeb32 t;
		_ftime32_s(&t);

		seed = (unsigned INT)(t.millitm) * (unsigned INT)(this);

	#endif
#else
		timeb t;
		ftime(&t);

		seed = (unsigned INT)(t.millitm) * (unsigned INT)(this)*rand();
#endif // !__GNUC__

		srand((unsigned int)seed);
		if(diag_only)
		{
			memset(_data,0,(size_t)Ni*(size_t)Nj*sizeof(double));
			for(unsigned INT i=0;i<Ni;i++) 
				_data[i+Ni*i] = ((double)rand()/(double)RAND_MAX)*(max - min)+min;

			_type = Diagonal;
		}
		else
		{
			for(size_t i=0;i<(size_t)Ni*(size_t)Nj;i++)
				_data[i] = ((double)rand()/(double)RAND_MAX)*(max - min)+min;

			_type = General;
		}
	}

	FMatrix::FMatrix(const char* Filename)
	{
		FILE *file;
#ifndef __GNUC__
		fopen_s(&file,Filename,"r");
#else
		file = fopen(Filename, "r");
#endif

		char f_char;
		fread(&f_char,sizeof(char),1,file);
		rewind(file);

		if(f_char == 'N')
		{
			fclose(file);

			ifstream in(Filename);

			char temp[20];
			in>>temp>>_Ni>>temp>>_Nj;

			_data = (double*)malloc((size_t)_Ni*(size_t)_Nj*sizeof(double));
			if(errno == ENOMEM || _data == 0)
			{
				THROWERROR(L"FMatrix: Not enough memory\r\n");
			}

			_dims[0] = &_Ni;
			_dims[1] = &_Nj;

			for(size_t i = 0; i < (size_t)_Ni*(size_t)_Nj; i++)
			{
				in>>_data[i];
			}
		
			in.close();
		}
		else
		{
			char buff[100];

			while(!feof(file))
			{
				fscanf_s(file,"%s\n\r",buff,100);


				if(_Ni == 0)
				{
					char* iterator = buff;
					char* location = iterator;

					while(location != 0)
					{
						location = strchr(location+1,',');
						_Nj++;
					}
				}
				_Ni++;
			}

			rewind(file);

			memset(buff,0,100);

			_data = (double*)malloc((size_t)_Ni*(size_t)_Nj*sizeof(double));
			if(errno == ENOMEM || _data == 0)
			{
				THROWERROR(L"FMatrix: Not enough memory\r\n");
			}

			_dims[0] = &_Ni;
			_dims[1] = &_Nj;

			for(unsigned INT i = 0; i < _Ni; i++)
			{
				for(unsigned INT j = 0; j < _Nj; j++)
				{
					if(j < _Nj-1)
						fscanf_s(file,"%le,",&_data[(size_t)i+(size_t)_Ni*(size_t)j]);
					else
						fscanf_s(file,"%le\n\r",&_data[(size_t)i+(size_t)_Ni*(size_t)j]);
				}
			}

			fclose(file);
		}

		_type = General;
	}

#ifndef __GNUC__
	FMatrix::FMatrix(const VARIANT& Mat)
	{
		if(Mat.vt == VT_EMPTY)
		{
			_Ni = 0;
			_Nj = 0;
			_data = 0;

			_dims[0] = &_Ni;
			_dims[1] = &_Nj;

			return;
		}
		//Matrix Dimensions
		if(Mat.parray->cDims == 2)
		{
			_Ni = Mat.parray->rgsabound[1].cElements;
			_Nj = Mat.parray->rgsabound[0].cElements;
		}
		else if(Mat.parray->cDims == 1)
		{
			_Ni = Mat.parray->rgsabound[0].cElements;
			_Nj = 1;
		}
		//
		double *data = 0;

		SafeArrayAccessData(Mat.parray,(void**)&data);

		_data = (double*)malloc((size_t)_Ni*(size_t)_Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}

		memcpy(_data,data,(size_t)_Ni*(size_t)_Nj*sizeof(double));

		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		SafeArrayUnaccessData(Mat.parray);

		data = 0;

		_type = General;
	}
#endif

	FMatrix::FMatrix(const string &MatrixString)
	{
		string::size_type si = 0, ei = 0;

		si = MatrixString.find(" ");
		ei = MatrixString.find(" ",si+1);

		unsigned int Ni = 0;
		sscanf_s(MatrixString.substr(si, ei - si).c_str(),"%d",&Ni);
		_Ni = Ni;

		si = MatrixString.find(" ",ei+3);
		ei = MatrixString.find(" ",si+1);

		unsigned int Nj = 0;
		sscanf_s(MatrixString.substr(si,ei-si).c_str(),"%d",&Nj);
		_Nj = Nj;

		_data = (double*)malloc((size_t)_Ni*(size_t)_Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}

		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		for(size_t i = 0; i < (size_t)_Ni*(size_t)_Nj; i++)
		{
			si = ei+1;
			ei = MatrixString.find(" ",si+1);

			sscanf_s(MatrixString.substr(si,ei-si).c_str(),"%le",&_data[i]);
		}

	}

	FMatrix::FMatrix(const CMatrix<double>& Mat)
	{
		_Ni = Mat.GetNRows();
		_Nj = Mat.GetNColumns();

		if(_Ni == 0 || _Nj == 0)
		{
			_data = 0;
			return;
		}

		_data = (double*)malloc((size_t)_Ni*(size_t)_Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"FMatrix: Not enough memory\r\n");
		}

		memcpy_s(_data,(size_t)_Ni*(size_t)_Nj*sizeof(double),Mat.GetDataPtr(),(size_t)Mat.NRows*(size_t)Mat.NColumns*sizeof(double));
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		_type = General;
	}

	FMatrix::FMatrix(const double& Val)
	{

		_Ni = 1;
		_Nj = 1;

		_data = (double*)malloc(_Ni*_Nj*sizeof(double));

		_data[0] = Val;

		_type = General;
	}

	FMatrix::FMatrix(const float& Val)
	{

		_Ni = 1;
		_Nj = 1;

		_data = (double*)malloc(_Ni*_Nj*sizeof(double));

		_data[0] = Val;

		_type = General;
	}

	FMatrix::FMatrix(const INT& Val)
	{

		_Ni = 1;
		_Nj = 1;

		_data = (double*)malloc(_Ni*_Nj*sizeof(double));

		_data[0] = (double)Val;

		_type = General;
	}

	FMatrix::FMatrix(const unsigned INT& Val)
	{

		_Ni = 1;
		_Nj = 1;

		_data = (double*)malloc(_Ni*_Nj*sizeof(double));

		_data[0] = (double)Val;

		_type = General;
	}

	//
	//----------------------------------------------------------------------

	//---Destructor---------------------------------------------------------
	//
	FMatrix::~FMatrix()
	{
		if(_data !=0)
			free(_data);
		_data = 0;
	}
	//
	//----------------------------------------------------------------------

	//---Public Methods-----------------------------------------------------
	//
	void FMatrix::Initialize(unsigned INT Ni,unsigned INT Nj)
	{
		if((Ni != 0)&&(Nj != 0))
		{
			if((_Ni != Ni)||(_Nj != Nj))
			{
				_data = (double*)realloc(_data,(size_t)Ni*(size_t)Nj*sizeof(double));
				if(errno == ENOMEM || _data == 0)
				{
					THROWERROR(L"Initialization: Not enough memory\r\n");
				}

				_Ni = Ni;
				_Nj = Nj;
			}

			memset(_data,0,(size_t)Ni*(size_t)Nj*sizeof(double));

			_type = General;
		}
		else
		{
			DestroyMatrix();
		}
	}
	void FMatrix::Initialize(unsigned INT Ni, unsigned INT Nj, const double *data)
	{
		_data = (double*)realloc(_data,(size_t)Ni*(size_t)Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"Initialization: Not enough memory\r\n");
		}

		memcpy_s(_data,(size_t)Ni*(size_t)Nj*sizeof(double),data,(size_t)Ni*(size_t)Nj*sizeof(double));

		_Ni = Ni;
		_Nj = Nj;

		_type = General;
	}
	void FMatrix::Initialize(unsigned INT Ni,unsigned INT Nj,double val, bool diag_only)
	{
		if((Ni != 0)||(Nj != 0))
		{
			if((_Ni != Ni)||(_Nj != Nj))
			{
				_data = (double*)realloc(_data,(size_t)Ni*(size_t)Nj*sizeof(double));
				if(errno == ENOMEM || _data == 0)
				{
					THROWERROR(L"Initialization: Not enough memory\r\n");
				}

				if(_data == 0)
				{
					wchar_t buff[100];
#ifndef __GNUC__
					_wcserror_s(buff,sizeof(buff),errno);
#endif
					THROWERROR(buff);
				}

				_Ni = Ni;
				_Nj = Nj;
			}

			if(diag_only)
			{
				memset(_data,0,(size_t)Ni*(size_t)Nj*sizeof(double));
				for(unsigned INT i=0;i<Ni;i++) _data[(size_t)i+(size_t)Ni*(size_t)i] = val;

				_type = Diagonal;
			}
			else
			{
				for(size_t i=0;i<(size_t)Ni*(size_t)Nj;i++) _data[i] = val;

				_type = General;
			}
		}
		else
		{
			DestroyMatrix();
		}
	}
	void FMatrix::Initialize(unsigned INT Ni,unsigned INT Nj, double min, double max, bool diag_only)
	{	
		_data = (double*)realloc(_data,(size_t)Ni*(size_t)Nj*sizeof(double));
		if(errno == ENOMEM || _data == 0)
		{
			THROWERROR(L"Initialization: Not enough memory\r\n");
		}

		unsigned INT seed;
#ifndef __GNUC__
#ifdef _WIN64
		__timeb64 t;
		_ftime64_s(&t);

		seed = (unsigned INT)(t.millitm) * (unsigned INT)(this)*rand();
#else
		__timeb32 t;
		_ftime32_s(&t);

		seed = (unsigned INT)(t.millitm) * (unsigned INT)(this);

#endif
#else
		timeb t;
		ftime(&t);

		seed = (unsigned INT)(t.millitm) * (unsigned INT)(this)*rand();
#endif // !__GNUC__

		srand((unsigned int)seed);
		if(diag_only)
		{
			memset(_data,0,(size_t)Ni*(size_t)Nj*sizeof(double));
			for(unsigned INT i=0;i<Ni;i++) 
				_data[(size_t)i+(size_t)Ni*(size_t)i] = ((double)rand()/(double)RAND_MAX)*(max - min)+min;

			_type = Diagonal;
		}
		else
		{
			for(size_t i=0;i<(size_t)Ni*(size_t)Nj;i++)
				_data[i] = ((double)rand()/(double)RAND_MAX)*(max - min)+min;

			_type = General;
		}

		_Ni = Ni;
		_Nj = Nj;
	}
	void FMatrix::Initialize(const char* Filename)
	{
		FILE *file;
#ifndef __GNUC__
		fopen_s(&file, Filename, "r");
#else
		file = fopen(Filename, "r");
#endif

		INT Ni=0, Nj=0;

		char f_char;
		fread(&f_char,sizeof(char),1,file);
		rewind(file);

		if(f_char == 'N')
		{
			fclose(file);

			ifstream in(Filename);

			char temp[20];
			in>>temp>>Ni>>temp>>Nj;

			if((_Ni != Ni)&&(_Nj != Nj))
			{
				_data = (double*)realloc(_data,(size_t)Ni*(size_t)Nj*sizeof(double));
				if(errno == ENOMEM || _data == 0)
				{
					THROWERROR(L"Initialization: Not enough memory\r\n");
				}

				_Ni = Ni;
				_Nj = Nj;
			}

			for(size_t i = 0; i < (size_t)_Ni*(size_t)_Nj; i++)
			{
				in>>_data[i];
			}
		
			in.close();
		}
		else
		{
			char buff[100];

			while(!feof(file))
			{
				fscanf_s(file,"%s\n\r",buff,100);

				if(Ni == 0)
				{
					char* iterator = buff;
					char* location = iterator;

					while(location != 0)
					{
						location = strchr(location+1,',');
						Nj++;
					}
				}
				Ni++;
			}

			rewind(file);

			memset(buff,0,100);

			if((_Ni != Ni)&&(_Nj != Nj))
			{
				_data = (double*)realloc(_data,(size_t)Ni*(size_t)Nj*sizeof(double));
				if(errno == ENOMEM || _data == 0)
				{
					THROWERROR(L"Initialization: Not enough memory\r\n");
				}

				_Ni = Ni;
				_Nj = Nj;
			}

			for(unsigned INT i = 0; i < _Ni; i++)
			{
				for(unsigned INT j = 0; j < _Nj; j++)
				{
					if(j < _Nj-1)
						fscanf_s(file,"%le,",&_data[(size_t)i+(size_t)_Ni*(size_t)j]);
					else
						fscanf_s(file,"%le\n\r",&_data[(size_t)i+(size_t)_Ni*(size_t)j]);
				}
			}

			fclose(file);
		}

		_type = General;
	}

	void FMatrix::Resize(unsigned INT Ni,unsigned INT Nj)
	{
		FMatrix temp = *this;

		this->Initialize(Ni,Nj);

		if(Ni >= temp._Ni && Nj >= temp._Nj)
		{
			SetRange(0, temp._Ni - 1, 0, temp._Nj - 1, temp);
		}
		else if(Ni < temp._Ni && Nj >= temp._Nj)
		{
			ReplaceColumn(0, temp._Nj - 1, temp.GetLine(0, Ni - 1));
		}
		else if(Ni >= temp._Ni && Nj < temp._Nj)
		{
			ReplaceLine(0, temp._Ni - 1, temp.GetColumn(0, Nj - 1));
		}
		else if(Ni < temp._Ni && Nj < temp._Nj)
		{
			SetRange(0, Ni - 1, 0, Nj - 1, temp.GetRange(0, Ni - 1, 0, Nj - 1));
		}
	}

	FMatrix FMatrix::GetRange(unsigned INT beg_line, unsigned INT end_line, unsigned INT beg_col, unsigned INT end_col) const
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

		FMatrix temp(_Ni,NJ), res(NI,NJ);

	
		for(unsigned INT j=0; j < NJ; j++)
			temp.ReplaceColumn(j, GetColumn(j + beg_col));

		for(unsigned INT i =0; i < NI; i++)
			temp.ReplaceLine(i, temp.GetLine(i + beg_line));

		return res;
	}

	RMatrix FMatrix::RefRange(unsigned INT beg_line, unsigned INT end_line, unsigned INT beg_col, unsigned INT end_col)
	{
		unsigned INT NI, NJ;

		if (beg_line == end_line)
			NI = 1;
		else
			NI = end_line - beg_line + 1;

		if (beg_col == end_col)
			NJ = 1;
		else
			NJ = end_col - beg_col + 1;

		RMatrix res(this, NI, NJ, beg_line, beg_col);

		return res;

	}

	void FMatrix::SetRange(unsigned INT beg_line, unsigned INT end_line, unsigned INT beg_col, unsigned INT end_col,const FMatrix& newMat)
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

		double *temp = &_data[(size_t)beg_line+(size_t)_Ni*(size_t)beg_col];
		for(unsigned INT j=0;j<NJ;j++)
		{
			memcpy_s(temp,NI*sizeof(double),newMat.GetColumn(j)._data,NI*sizeof(double));
			temp += _Ni;
		}

		_type = General;
	}

	FMatrix FMatrix::GetRange(const CMatrix<unsigned INT>& poINT_list) const
	{
		FMatrix res(poINT_list.GetNRows(),1);

		for(unsigned INT i=0;i<(unsigned INT)res.GetLength();i++)
		{
			res._data[i] = _data[poINT_list(i)];
		}

		return res;
	}

	void FMatrix::SetRange(const CMatrix<unsigned INT>& poINT_list,const FMatrix& newMat)
	{
		if(newMat.GetLength() != poINT_list.GetLength())
			THROWERROR(L"SetRange: The number of new values must be equal to the munber of indices\r\n");

		for(unsigned INT i =0;i<poINT_list.GetLength();i++)
			_data[poINT_list(i)] = newMat(i);

		_type = General;
	}

	void FMatrix::SetRange(const CMatrix<unsigned INT>& poINT_list,double val)
	{
		for(unsigned INT i =0;i<poINT_list.GetLength();i++)
			_data[poINT_list(i)] = val;

		_type = General;
	}

	void FMatrix::ReplaceColumn(unsigned INT j,const FMatrix& NewCol)
	{
		if(NewCol._Nj > 1)
			THROWERROR(L"NewCol must be a column vector\r\n");

		if(NewCol._Ni != _Ni)
			THROWERROR(L"NewCol must have the same number of lines as the matrix\r\n");

		double *temp = &_data[(size_t)_Ni*(size_t)j];

		memcpy_s(temp,sizeof(double)*_Ni,NewCol._data,sizeof(double)*_Ni);

		_type = General;

	}

	void FMatrix::ReplaceColumn(unsigned INT j0, unsigned INT j1, const FMatrix& NewCol)
	{
		unsigned INT NJ = 0;

		if(j0 == j1)
			NJ = 1;
		else
			NJ = j1-j0+1;

		if(NewCol._Nj != NJ)
			THROWERROR(L"The rhs must have the same number of columns as specified\r\n");

		if(NewCol._Ni != _Ni)
			THROWERROR(L"NewCol must have the same number of lines as the matrix\r\n");

		double *temp = &_data[(size_t)_Ni*(size_t)j0];

		memcpy_s(temp,sizeof(double)*((size_t)_Ni*(size_t)NJ),NewCol._data,sizeof(double)*((size_t)_Ni*(size_t)NJ));

		_type = General;
	}

	void FMatrix::ReplaceLine(unsigned INT i,const FMatrix& NewLine)
	{
		if(NewLine._Ni > 1)
			THROWERROR(L"NewLine must be a line vector\r\n");

		if(NewLine._Nj != _Nj)
			THROWERROR(L"NewLine must have the same number of columns as the matrix\r\n");

		for(unsigned INT j = 0; j < _Nj; j++)
		{
			(*this)(i,j) = NewLine(j);
		}

		_type = General;
	}

	void FMatrix::ReplaceLine(unsigned INT i0, unsigned INT i1, const FMatrix& NewLine)
	{
		unsigned INT NI = 0;

		if(i0 == i1)
			NI = 1;
		else
			NI = i1-i0+1;

		if(NewLine._Ni != NI)
			THROWERROR(L"The rhs must have the same number of columns as specified\r\n");

		if(NewLine._Nj != _Nj)
			THROWERROR(L"NewLine must have the same number of columns as the matrix\r\n");

		for(unsigned INT i = 0; i < NI; i++)
		{
			for(unsigned INT j = 0; j < _Nj; j++)
			{
				(*this)(i+i0,j) = NewLine(i,j);
			}
		}

		_type = General;
	}


	FMatrix FMatrix::GetColumn(unsigned INT j) const
	{
		FMatrix res(_Ni,1);

		memcpy_s(res._data,sizeof(double)*_Ni,&_data[(size_t)_Ni*(size_t)j],sizeof(double)*_Ni);

		return res;
	}

	RMatrix FMatrix::RefColumn(unsigned INT j)
	{
		RMatrix res(this, _Ni, 1, 0, j);

		return res;
	}

	FMatrix FMatrix::GetColumn(unsigned INT j0,unsigned  INT j1) const
	{
		if(j0 > j1)
			THROWERROR(L"FMatrix::GetColumn - The index of the last column must be larger than the firt one\r\n");

		if(j0 < 0)
			THROWERROR(L"FMatrix::GetColumn - The lower index is outside the matrix bounds\r\n");

		if(j1 >= _Nj)
			THROWERROR(L"FMatrix::GetColumn - The upper index is outside the matrix bounds\r\n");

		unsigned INT NJ = 0;

		if(j0 == j1)
			NJ = 1;
		else
			NJ = j1-j0+1;

		FMatrix res(_Ni,NJ);

		memcpy_s(res._data,sizeof(double)*((size_t)_Ni*(size_t)NJ),&_data[(size_t)_Ni*(size_t)j0],sizeof(double)*((size_t)_Ni*(size_t)NJ));

		return res;
	}

	FMatrix FMatrix::GetLine(unsigned INT i) const
	{
		FMatrix res(1,_Nj);

		for(unsigned INT j = 0; j < _Nj; j++)
			res(0,j) = _data[(size_t)i+(size_t)_Ni*(size_t)j];

		return res;
	}

	RMatrix FMatrix::RefRow(unsigned INT i)
	{
		RMatrix res(this, 1, _Nj, i, 0);

		return res;
	}

	FMatrix FMatrix::GetLine(unsigned INT i0, unsigned INT i1) const
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

		FMatrix res(NI,_Nj);


		for(unsigned INT i = 0; i < NI; i++)
		{
			for(unsigned INT j = 0; j<_Nj; j++)
			{
				res(i,j) = _data[(size_t)i0+(size_t)i+(size_t)_Ni*(size_t)j];
			}
		}

		return res;
	}

	void FMatrix::Reshape(unsigned INT NewNi, unsigned INT NewNj)
	{
		if((size_t)NewNi*(size_t)NewNj != (size_t)_Ni*(size_t)_Nj)
			THROWERROR(L"Reshape: Matrix must have the same number of elements\r\n");

		_Ni = NewNi;
		_Nj = NewNj;

		_type = General;
	}

	void FMatrix::Clear()
	{
		DestroyMatrix();
	}


	bool FMatrix::IsEmpty() const
	{
		if(_Ni == 0 || _Nj == 0)
			return true;
		else
			return false;
	}

	const double* FMatrix::ToDoublePtr() const
	{
		return _data;
	}

	double FMatrix::ToDouble() const
	{
		if(_Ni !=1 || _Nj != 1)
			THROWERROR(L"ToDouble: The supplied FMatrix must be a scalar to be converted to double\r\n");

		return _data[0];
	}

	void FMatrix::SaveToFile(const char* Filename) const
	{
		ofstream out(Filename);

		out<<"Ni= "<<_Ni<<" Nj= "<<_Nj<<" ";

		out.precision(16);
		for(size_t i=0;i<(size_t)_Ni*(size_t)_Nj;i++)
		{
			out<<scientific<<_data[i]<<" ";
		}

		out.close();
	}

	void FMatrix::LoadFromFile(const char* Filename)
	{
		char temp[20];

		INT ni,no;
		double *data;

		ifstream in(Filename);

		in>>temp>>ni>>temp>>no;

		data = (double*)malloc((size_t)ni*(size_t)no*sizeof(double));

		for(size_t i = 0; i < (size_t)ni*(size_t)no; i++)
		{
			in>>data[i];
		}

		Initialize(ni,no,data);

		free(data);
		data = 0;
	}

	void FMatrix::SaveToCompressedFile(const char* Filename) const
	{
		char buff[100];

		char div = 3;
		char dec = 1;
		char sgn = 4;
		char exp = 2;

		FILE *file;

#ifndef __GNUC__
		fopen_s(&file, Filename, "w");
#else
		file = fopen(Filename, "w");
#endif

		CMatrixLib::ToCompressedString(_Ni,buff, sizeof(buff)/sizeof(char));

		fwrite(buff,sizeof(char),strlen(buff),file);
		fwrite(&div,sizeof(char),1,file);

		CMatrixLib::ToCompressedString(_Nj,buff, sizeof(buff)/sizeof(char));
		fwrite(buff,sizeof(char),strlen(buff),file);
		fwrite(&div,sizeof(char),1,file);

		for(size_t i =0;i<(size_t)_Ni*(size_t)_Nj; i++)
		{
			double dval = _data[i];

			INT exp_part;
			INT exp_sign;
			INT INT_part;
			INT INT_sign;

			if(dval != 0)
			{
				exp_part = (INT)std::floor(fabs(log10(fabs(dval))));
				exp_sign = sign(log10(fabs(dval)));

				dval *= std::pow(10.0,-exp_sign*exp_part);

				INT_part = (INT)fabs(dval);
				INT_sign = sign(dval);
			}
			else
			{
				exp_part = 0;
				exp_sign = 1;
				INT_part = 0;
				INT_sign = 1;
			}

			unsigned long long dec_part = (unsigned long long)((fabs(dval)-INT_part)*std::pow(10.0,16));

			if(INT_sign < 0)
				fwrite(&sgn,sizeof(char),1,file);

			CMatrixLib::ToCompressedString(INT_part,buff,sizeof(buff)/sizeof(char));
			fwrite(buff,sizeof(char),strlen(buff),file);
			fwrite(&dec,sizeof(char),1,file);

			CMatrixLib::ToCompressedString(dec_part,buff,sizeof(buff)/sizeof(char));
			fwrite(buff,sizeof(char),strlen(buff),file);
			fwrite(&exp,sizeof(char),1,file);

			if(exp_sign < 0)
				fwrite(&sgn,sizeof(char),1,file);

			CMatrixLib::ToCompressedString(exp_part,buff,sizeof(buff)/sizeof(char));
			fwrite(buff,sizeof(char),strlen(buff),file);
			fwrite(&div,sizeof(char),1,file);
		}

		fclose(file);
	}

	void FMatrix::LoadFromCompressedFile(const char* Filename)
	{
		char buff[100];

		char div = 3;
		char dec = 1;
		char sgn = 4;
		char exp = 2;

		ifstream file(Filename,ios_base::in|ios_base::binary);

		file.getline(buff,sizeof(buff)/sizeof(char),div);
		unsigned INT Ni = (unsigned INT)FromCompressedString(buff,(INT)strlen(buff));
		file.getline(buff,sizeof(buff)/sizeof(char),div);
		unsigned INT Nj = (unsigned INT)FromCompressedString(buff,(INT)strlen(buff));

		double *data = (double*)malloc((size_t)Ni*(size_t)Nj*sizeof(double));

		for(size_t i =0;i<(size_t)Ni*(size_t)Nj;i++)
		{
			file.getline(buff,sizeof(buff)/sizeof(char),dec);
			INT INT_sign = 0;
		
			char* ptr = buff;

			if(buff[0] == sgn)
			{
				ptr++;
				INT_sign = -1;
			}
			else
			{
				INT_sign = 1;
			}

			INT INT_part = (INT)FromCompressedString(ptr,(INT)strlen(ptr));

			file.getline(buff,sizeof(buff)/sizeof(char),exp);
			unsigned long long dec_part = FromCompressedString(buff,(INT)strlen(buff));

			double dval = INT_sign*(INT_part + dec_part*std::pow(10.0,-16));

			file.getline(buff,sizeof(buff)/sizeof(char),div);
			INT exp_sign = 0;
		
			ptr = buff;

			if(buff[0] == sgn)
			{
				ptr++;
				exp_sign = -1;
			}
			else
			{
				exp_sign = 1;
			}

			INT exp_part = (INT)FromCompressedString(ptr,(INT)strlen(ptr));

			data[i] = dval*std::pow(10.0,exp_sign*exp_part);

		}

		Initialize(Ni,Nj,data);

	
		free(data);
		data = 0;
	}

	void FMatrix::SaveToCSVFile(const char* Filename) const
	{
		ofstream file(Filename);
		file.precision(16);

		for(unsigned INT i = 0; i < _Ni; i++)
		{
			for(unsigned INT j = 0; j < _Nj; j++)
			{
				file<<scientific<<_data[(size_t)i+(size_t)_Ni*(size_t)j];

				if(j < _Nj-1)
					file<<",";
				else
					file<<endl;
			}
		}

		file.close();
	}

	void FMatrix::LoadFromCSVFile(const char* Filename)
	{
		char buff[1000], *buff2, *sep, *line;
		unsigned INT Ni = 0,Nj = 0;

		FILE *file = 0;

#ifndef __GNUC__
		errno_t err = fopen_s(&file, Filename, "r");
#else
		file = fopen(Filename, "r");
#endif

		//Check sizes
		unsigned INT nj_count = 0, ni_count = 0;
		while(!feof(file))
		{
			long init = ftell(file);
			INT retries = 0;
			while(1)
			{
				buff2 = fgets(buff,100,file);

				sep = strchr(buff2,',');
				line = strchr(buff2,'\n');


				if(sep != 0 || line != 0)
					break;

				retries++;
			};

			//find # of commas
			char *location = buff2;
			while(1)
			{
				location = strchr(location+1,',');

				if(location == 0)
					break;

				nj_count++;
			}
			//

			//Put cursor on last comma or ends
			if(strrchr(buff2,'\n') != 0)
			{
				nj_count++;
				break;
			}

			long end = ftell(file);
			location = strrchr(buff2,',');
			long pos = end - ((long)retries*100 + (100-(long)(location-buff2)))+2;

			fseek(file,pos,SEEK_SET);
			//
		}
		//

		ni_count = 1;
		while(!feof(file))
		{
			while(1)
			{
				buff2 = fgets(buff,100,file);

				if(buff2 == 0)
					break;

				sep = strchr(buff2,'\n');

				if(sep != 0)
					break;
			};

			if(buff2 == 0)
				break;

			ni_count++;
		}

		//rewind(file);
		
		fclose(file);

#ifndef __GNUC__
		err = fopen_s(&file, Filename, "r");
#else
		file = fopen(Filename, "r");
#endif

		memset(buff,0,100);

		Initialize(ni_count,nj_count);

		for(unsigned INT i = 0; i < ni_count; i++)
		{
			for(unsigned INT j = 0; j < nj_count; j++)
			{
				if(j < _Nj-1)
					fscanf_s(file,"%lf,",&_data[(size_t)i+(size_t)_Ni*(size_t)j]);
				else
					fscanf_s(file,"%lf\n",&_data[(size_t)i+(size_t)_Ni*(size_t)j]);
			}
		}

		fclose(file);
	}

	string FMatrix::ToCSVstring()
	{
		string res = "";

		if(this->IsEmpty())
			return "";

		char* temp = new char[40];

		if(_Ni == 1 && _Nj == 1)
		{
			sprintf_s(temp,40*sizeof(char),"%17.16e\n",_data[0]);
			res.append(temp);
		}
		else
		{
			for(unsigned INT i = 0; i < _Ni; i++)
			{
				for(unsigned INT j = 0; j < _Nj; j++)
				{
					sprintf_s(temp,40*sizeof(char),"%17.16e",_data[(size_t)i+(size_t)_Ni*(size_t)j]);
					res.append(temp);

					if(j < _Nj-1)
						res.append(",");
					else if(j >= _Nj-1 && _Ni != 1)
						res.append("\n");
				}
			}
		}

		delete[] temp;
		temp = 0;

		return res;
	}

	string FMatrix::ToString() const
	{
		string res = "";
		char temp[100];
		memset(temp,0,sizeof(temp));

		sprintf_s(temp,sizeof(temp),"Ni= %d Nj= %d ",(int)_Ni, (int)_Nj);
		res = temp;
		memset(temp,0,sizeof(temp));

		for(unsigned INT i = 0; i < (size_t)_Ni*(size_t)_Nj; i++)
		{
			sprintf_s(temp,sizeof(temp),"%17.16e ",_data[i]);
			res.append(temp);
		}

		return res;
	}
	
	string FMatrix::ToCompressedString() const
	{
		string res = "";
		char buff[100];
		memset(buff, 0, sizeof(buff));

		char div = 3;
		char dec = 1;
		char sgn = 4;
		char exp = 2;

		CMatrixLib::ToCompressedString(_Ni, buff, sizeof(buff) / sizeof(char));

		res = buff;
		res.append(&div, 1);

		memset(buff, 0, sizeof(buff));
		CMatrixLib::ToCompressedString(_Nj, buff, sizeof(buff) / sizeof(char));
		res.append(buff);
		res.append(&div, 1);

		for (size_t i = 0; i<(size_t)_Ni*(size_t)_Nj; i++)
		{
			double dval = _data[i];

			INT exp_part;
			INT exp_sign;
			INT INT_part;
			INT INT_sign;

			if (dval != 0)
			{
				exp_part = (INT)std::floor(fabs(log10(fabs(dval))));
				exp_sign = sign(log10(fabs(dval)));

				dval *= std::pow(10.0, -exp_sign * exp_part);

				INT_part = (INT)fabs(dval);
				INT_sign = sign(dval);
			}
			else
			{
				exp_part = 0;
				exp_sign = 1;
				INT_part = 0;
				INT_sign = 1;
			}

			unsigned long long dec_part = (unsigned long long)((fabs(dval) - INT_part)*std::pow(10.0, 16));

			if (INT_sign < 0)
				res.append(&sgn,1);

			memset(buff, 0, sizeof(buff));
			CMatrixLib::ToCompressedString(INT_part, buff, sizeof(buff) / sizeof(char));
			res.append(buff);
			res.append(&dec, 1);

			memset(buff, 0, sizeof(buff));
			CMatrixLib::ToCompressedString(dec_part, buff, sizeof(buff) / sizeof(char));
			res.append(buff);
			res.append(&exp, 1);

			if (exp_sign < 0)
				res.append(&sgn, 1);

			memset(buff, 0, sizeof(buff));
			CMatrixLib::ToCompressedString(exp_part, buff, sizeof(buff) / sizeof(char));
			res.append(buff);
			res.append(&div, 1);
		}

		return res;
	}

	void FMatrix::SetType(MatrixType type, bool check)
	{
		if(check)
		{
			if(!CheckType(_type))
				THROWERROR(L"SetType: Supplied Type does not match the matrix data\r\n");
		}

		_type = type;
	}

	size_t FMatrix::GetSizeInBytes()
	{
		size_t res = sizeof(FMatrix);

		if(_data != 0)
		{
			if(_type == Diagonal)
				res += _Ni*sizeof(double);
			else if(_type == UpperTriangular)
			{
				if(_Ni==_Nj || _Ni > _Nj)
					res += (size_t)(0.5*(size_t)_Ni*(1+(size_t)_Ni)*sizeof(double));
				else
					res += (size_t)((0.5*(size_t)_Ni*(1+(size_t)_Ni)+((size_t)_Nj-(size_t)_Ni)*(size_t)_Ni)*sizeof(double));
			}
			else if(_type == LowerTriangular)
			{
				if(_Ni==_Nj || _Ni < _Nj)
					res += (size_t)(0.5*(size_t)_Nj*(1+(size_t)_Nj)*sizeof(double));
				else
					res += (size_t)((0.5*(size_t)_Nj*(1+(size_t)_Nj)+((size_t)_Ni-(size_t)_Nj)*(size_t)_Nj)*sizeof(double));
			}
			else
			{
				res+= _Ni*_Nj*sizeof(double);
			}
		}

		return res;
	}

	void FMatrix::Pack(unsigned INT increment)
	{
		if(_Ni != 1 && _Nj != 1)
		{
			THROWERROR(L"Pack: this function \"Pack(INT increment)\" is only valid for vectors\r\n"); 
		}

		unsigned INT new_size = (unsigned INT)(((size_t)_Ni*(size_t)_Nj)/(size_t)increment);
		double* temp_data = (double*)malloc(new_size*sizeof(double));

		vdPackI(new_size,_data,increment,temp_data);

		if(_Nj == 1)
		{
			Initialize(new_size,1,temp_data);
		}
		else if(_Ni == 1)
		{
			Initialize(1,new_size,temp_data);
		}
	
		free(temp_data);
		temp_data = 0;
	}
	//
	//----------------------------------------------------------------------

	//---Public Operators---------------------------------------------------
	//
	double& FMatrix::operator() (unsigned INT i,unsigned INT j)
	{
		if(i >= _Ni || j >= _Nj || i < 0 || j < 0)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		if(_type == Diagonal && i!=j)
			_type = General;

		return _data[(size_t)i+(size_t)_Ni*(size_t)j];
	}

	double FMatrix::operator() (unsigned INT i, unsigned INT j) const
	{
		if(i >= _Ni || j >= _Nj || i < 0 || j < 0)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _data[(size_t)i+(size_t)_Ni*(size_t)j];
	}

	double& FMatrix::operator() (unsigned INT i)
	{
		if(i >= (size_t)_Ni*(size_t)_Nj || i < 0)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _data[i];
	}

	double FMatrix::operator() (unsigned INT i) const
	{
		if(i >= (size_t)_Ni*(size_t)_Nj || i < 0)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _data[i];
	}

	FMatrix FMatrix::operator+(const FMatrix& B) const
	{
		if(B.IsEmpty())
			return *this;
		else if(IsEmpty())
			return B;

		if(_Ni != B._Ni || _Nj != B._Nj)
			THROWERROR(L"operator+: Matrices must have the same dimensions\r\n");

		FMatrix C(_Ni,_Nj); 

		vdAdd(_Ni*_Nj,_data,B._data,C._data);

		C._type = OperationResType(B);

		return C;
	}

	FMatrix FMatrix::operator +(INT B)
	{
		FMatrix C(*this),b(_Ni,_Nj,(double)B,false);

		C._type = General;

		return C+b;
	}

	FMatrix FMatrix::operator +(double B)
	{
		FMatrix C(*this),b(_Ni,_Nj,B,false);

		C._type = General;

		return C+b;
	}


	FMatrix FMatrix::operator-(const FMatrix& B) const 
	{
		if(_Ni != B._Ni || _Nj != B._Nj)
			THROWERROR(L"operator - : The matrices must have the same dimensions\r\n");

		FMatrix C(_Ni,_Nj); 

		vdSub(_Ni*_Nj,_data,B._data,C._data);

		C._type = OperationResType(B);

		return C;
	}

	FMatrix FMatrix::operator-(INT B) const
	{
		FMatrix C(*this),b(_Ni,_Nj,(double)B,false);

		C._type = General;

		return C-b;
	}

	FMatrix FMatrix::operator-(double B) const
	{
		FMatrix C(*this),b(_Ni,_Nj,B,false);

		C._type = General;

		return C-b;
	}


	FMatrix FMatrix::operator*(INT B)
	{
		FMatrix res(*this);

		cblas_dscal(_Ni*_Nj, (double)B,res._data,1);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator*(INT B) const
	{
		FMatrix res(*this);

		cblas_dscal(_Ni*_Nj, (double)B, res._data, 1);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator*(double B) const
	{
		if(_Ni == 1 && _Nj == 1)
		{
			FMatrix res(1,1,_data[0]*B);

			return res;
		}
		else
		{
			FMatrix res(*this);

			cblas_dscal(_Ni*_Nj,B,res._data,1);

			res._type = _type;

			return res;
		}
	}

	FMatrix FMatrix::operator*(double B)
	{
		if(_Ni == 1 && _Nj == 1)
		{
			FMatrix res(1,1,_data[0]*B);

			return res;
		}
		else
		{
			FMatrix res(*this);

			cblas_dscal(_Ni*_Nj,B,res._data,1);

			res._type = _type;

			return res;
		}
	}

	FMatrix FMatrix::operator*(float B)
	{
		FMatrix res(*this);

		cblas_dscal(_Ni*_Nj,B,res._data,1);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator*(const FMatrix& B) const
	{
		if(_Nj != B._Ni)
			THROWERROR(L"operator*: Matrices must have compatible dimensions (A.Nj == B.Ni)\r\n");

		FMatrix C;

		if(_Ni == 1 || _Nj == 1 || B._Nj == 1)
		{
			if (_Ni == 1 && _Nj == 1 && B._Ni == 1 && B._Nj == 1)
			{
				C.Initialize(1, 1, _data[0] * B._data[0]);
				return C;
			}
			else if(_Ni == 1 && _Nj == 1)
			{
				return B*_data[0];
			}
			else if(B._Ni == 1 && B._Nj == 1)
			{
				return *this*B._data[0];
			}
			else if(_Ni == 1 && B._Nj == 1)
			{
				C.Initialize(1,1);
				C(0) = Dot(*this,B);
			}
			else if(_Ni != 1 && B._Nj == 1)
			{
				C.Initialize(_Ni,B._Nj);

				cblas_dgemv(CblasColMajor,CblasNoTrans,_Ni,_Nj,1,_data,_Ni,B._data,1,0,C._data,1);
			}
			else if(_Ni == 1 && B._Nj != 1)
			{
				C.Initialize(B._Nj,_Ni);

				cblas_dgemv(CblasColMajor,CblasTrans,B._Ni,B._Nj,1,B._data,B._Ni,_data,1,0,C._data,1);

				INT temp = C._Ni;
				C._Ni = C._Nj;
				C._Nj = temp;
			}
			else if(_Nj == 1 && B._Ni == 1)
			{
				C.Initialize(_Ni,B._Nj);

				for(unsigned INT j = 0; j < C._Nj; j++)
					for(unsigned INT i = 0; i < C._Ni; i++)
						C(i,j) = _data[i]*B(j);
			}
		}
		else if(_type == Symmetric)
		{
			C = B;
			cblas_dsymm(CblasColMajor,CblasLeft,CblasLower,C._Ni,C._Nj,1,_data,_Ni,B._data,B._Ni,0,C._data,C._Ni);
		}
		else if(B._type == Symmetric)
		{
			C = *this;
			cblas_dsymm(CblasColMajor,CblasRight,CblasLower,C._Ni,C._Nj,1,B._data,B._Ni,_data,_Ni,0,C._data,C._Ni);
		}
		else if((_type == LowerTriangular || _type == UpperTriangular) && _Ni == _Nj)
		{
			C = B;
			cblas_dtrmm(CblasColMajor,CblasLeft,(_type == LowerTriangular)?CblasLower:CblasUpper,CblasNoTrans,CblasNonUnit,B._Ni,B._Nj,1,_data,_Ni,C._data,B._Ni);
		}
		else if((B._type == LowerTriangular || B._type == UpperTriangular) && B._Ni == B._Nj)
		{
			C = *this;
			cblas_dtrmm(CblasColMajor,CblasRight,(B._type == LowerTriangular)?CblasLower:CblasUpper,CblasNoTrans,CblasNonUnit,_Ni,_Nj,1,B._data,B._Ni,C._data,_Ni);
		}
		else if(_type == Diagonal)
		{
			if(B._type == Diagonal)
			{
				double *this_data = (double*)malloc(min(_Ni,_Nj)*sizeof(double)),
					   *B_data = (double*) malloc(min(B._Ni,B._Nj)*sizeof(double)),
					   *res = (double*)malloc(min(_Ni,B._Nj)*sizeof(double));

				vdPackI(min(_Ni,_Nj),_data,_Ni+1,this_data);
				vdPackI(min(B._Ni,B._Nj),B._data,B._Ni+1,B_data);

				vdMul(min(_Ni,B._Nj),this_data,B_data,res);

				FREE(this_data);
				FREE(B_data);

				C.Initialize(_Ni,B._Nj);

				vdUnpackI(min(_Ni,B._Nj),res,C._data,_Ni+1);

				FREE(res);
			}
			else
			{
				C.Initialize(_Ni,B._Nj);

				for(unsigned INT i =0;i<_Ni;i++)
					C.ReplaceLine(i,_data[(size_t)i+(size_t)_Ni*(size_t)i]*B.GetLine(i));
			}
		}
		else if(B._type == Diagonal)
		{
			C.Initialize(_Ni,B._Nj);

			memcpy_s(C._data,_Ni*B._Nj*sizeof(double),_data,_Ni*B._Nj*sizeof(double));
		
			double *it = C._data;
			for(unsigned INT j=0;j<B._Nj;j++)
			{
				cblas_dscal(_Ni,B._data[(size_t)j+(size_t)B._Ni*(size_t)j],it,1);
				it += _Ni;
			}
			it = 0;
		}
		else
		{
			C.Initialize(_Ni,B._Nj);
			cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,_Ni,B._Nj,_Nj,1,_data,_Ni,B._data,B._Ni,0,C._data,_Ni);
		}

		C._type = OperationResType(B);

		return C;
	}

	FMatrix FMatrix::operator,(FMatrix& B)
	{
		/*if(_Nj == 1 || _Ni == 1)
		{
			if((_Ni == 1 && _Nj != B.Nj) || (_Nj == 1 && _Ni != B.Ni))
				throw "operator,: The number of elements of the vector must be equal to the number of lines of the matrix";

			FMatrix res(B._Ni,B._Nj);

			for(INT i =0;i<B._Ni;i++)
			{
				res.Line[i] = _data[i]*B.Line[i];
			}

			res._type = _type;

			return res;
		}
		else if(B._Nj == 1 || B._Ni == 1)
		{
			if((B._Ni == 1 && _Nj != B.Nj) || (B._Nj == 1 && _Nj != B.Ni))
				throw "operator,: The number of elements of the vector must be equal to the number of lines of the matrix";

			FMatrix res(_Ni,_Nj);

			for(INT j =0;j<_Nj;j++)
			{
				res.Column[j] = (*this).Column[j]*B(j);
			}

			res._type = _type;

			return res;
		}
		else
		{*/
			if(_Nj != B._Nj || _Ni != B._Ni)
				THROWERROR(L"operator,: Matrices must have the same dimensions\r\n");

			FMatrix res(_Ni,_Nj);

			vdMul(_Ni*_Nj,_data,B._data,res._data);

			res._type = _type;

			return res;
		//}
	}

	FMatrix FMatrix::operator/(INT B) const
	{
		FMatrix res(*this);

		cblas_dscal(_Ni*_Nj,1.0/(double)B,res._data,1);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator/(unsigned INT B) const
	{
		FMatrix res(*this);

		cblas_dscal(_Ni*_Nj,1.0/(double)B,res._data,1);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator/(double B) const
	{
		FMatrix res(*this);

		cblas_dscal(_Ni*_Nj,1/B,res._data,1);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator/(float B) const
	{
		FMatrix res(*this);

		cblas_dscal(_Ni*_Nj,1/B,res._data,1);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator/(const FMatrix& B) const
	{
		if(_Nj != B._Ni)
			THROWERROR(L"operator/: Matrices must have compatible dimensions (A.Nj == B.Ni)\r\n");

		if(B._Ni == 1 && B._Nj == 1)
		{
			return (*this)/B.ToDouble();
		}


		FMatrix C(_Ni,B._Nj);
		FMatrix IB = Inverse(B);

		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,_Ni,IB._Nj,_Nj,1,_data,_Ni,IB._data,IB._Ni,0,C._data,_Ni);

		C._type = OperationResType(B);

		return C;
	}

	FMatrix FMatrix::operator|(const FMatrix& B) const
	{
		if(_Nj != B._Nj || _Ni != B._Ni)
			THROWERROR(L"operator|: Matrices must have the same dimensions\r\n");

		FMatrix res(_Ni,_Nj);

		vdDiv(_Ni*_Nj,_data,B._data,res._data);

		res._type = _type;

		return res;
	}

	FMatrix FMatrix::operator^(INT B)
	{
		if(B > 0)
		{
			if(B == 1)
			{
				return *this;
			}
			else
			{
				FMatrix C(_Ni,_Nj,1);
			
				for(INT i = 0; i < B; i++)
				{
					C = C*(*this);
				}

				return C;
			}
		}
		else if(B < 0)
		{
			if(B == -1)
			{
				return Inverse(*this);
			}
			else
			{
				FMatrix C(_Ni,_Nj,1);

				for(INT i = 0; i < -B; i++)
				{
					C = C*(*this);
				}

				return Inverse(C);
			}
		}
		else
		{
			FMatrix C(_Ni,_Nj,1);

			return C;
		}
	}

	FMatrix FMatrix::operator-() const
	{
		return (*this)*(-1.0);
	}

	FMatrix FMatrix::operator&&(const FMatrix& B) const
	{
		if(_Nj != B._Nj || _Ni != B._Ni)
			THROWERROR(L"operator&&: Matrices must have the same dimensions\r\n");

		return (BMatrix(*this)&& BMatrix(B));
	}

	FMatrix FMatrix::operator||(const FMatrix& B) const
	{
		if(_Nj != B._Nj || _Ni != B._Ni)
			THROWERROR(L"operator||: Matrices must have the same dimensions\r\n");

		return (BMatrix(*this)|| BMatrix(B));
	}
	
	FMatrix FMatrix::operator%(const FMatrix& B) const
	{
		if(_Nj != B._Nj || _Ni != B._Ni)
			THROWERROR(L"operator%: Matrices must have the same dimensions\r\n");

		FMatrix res(B._Ni,B._Nj);

		for(size_t i = 0; i<(size_t)B._Ni*(size_t)B._Nj;i++)
		{
			res._data[i] = (double)((INT)_data[i] % (INT)B._data[i]);
		}

		return res;
	}
	FMatrix FMatrix::operator>>(const FMatrix& B) const
	{
		if(_Nj != B._Nj || _Ni != B._Ni)
			THROWERROR(L"operator>>: Matrices must have the same dimensions\r\n");

		FMatrix res(B._Ni,B._Nj);

		for(size_t i = 0; i<(size_t)B._Ni*(size_t)B._Nj;i++)
		{
			res._data[i] = (double)((INT)_data[i] >> (INT)B._data[i]);
		}

		return res;
	}
	FMatrix FMatrix::operator<<(const FMatrix& B) const
	{
		if(_Nj != B._Nj || _Ni != B._Ni)
			THROWERROR(L"operator<<: Matrices must have the same dimensions\r\n");

		FMatrix res(B._Ni,B._Nj);

		for(size_t i = 0; i<(size_t)B._Ni*(size_t)B._Nj;i++)
		{
			res._data[i] = (double)((INT)_data[i] << (INT)B._data[i]);
		}

		return res;
	}

	FMatrix FMatrix::operator&(const FMatrix& B) const
	{
		if(_Nj != B._Nj || _Ni != B._Ni)
			THROWERROR(L"operator&: Matrices must have the same dimensions\r\n");

		FMatrix res(B._Ni,B._Nj);

		for(size_t i = 0; i<(size_t)B._Ni*(size_t)B._Nj;i++)
		{
			res._data[i] = (double)((INT)_data[i] & (INT)B._data[i]);
		}

		return res;
	}


	FMatrix& FMatrix::operator=(const double *Mat)
	{
		if((_Ni == 0)||(_Nj == 0))
			THROWERROR(L"operator=: Matrix must have dimensions when equaling to a double*\r\n");

		memcpy(_data,Mat,(size_t)_Ni*(size_t)_Nj*sizeof(double));

		_type = General;

		return *this;
	}

	FMatrix& FMatrix::operator= (const INT *Mat)
	{
		if((_Ni == 0)||(_Nj == 0))
			THROWERROR(L"operator=: Matrix must have dimensions when equaling to a INT*\r\n");

		for(size_t i=0;i<(size_t)_Ni*(size_t)_Nj;i++) _data[i] = (double)Mat[i];

		_type = General;

		return *this;
	}

	FMatrix& FMatrix::operator= (const FMatrix &Mat)
	{
		_Ni = Mat._Ni;
		_Nj = Mat._Nj;

		if(Mat.IsEmpty())
		{
			DestroyMatrix();
			return *this;
		}

		_data = (double*)realloc(_data,_Ni*_Nj*sizeof(double));
		memcpy_s(_data,_Ni*_Nj*sizeof(double),Mat._data,Mat._Ni*Mat._Nj*sizeof(double));

		_type = Mat._type;

		return *this;
	}
#ifndef __GNUC__
	FMatrix& FMatrix::operator= (const VARIANT Mat)
	{
		if(Mat.vt == VT_EMPTY)
		{
			DestroyMatrix();
			return *this;
		}
	
		//Dimensoes da matriz
		if(Mat.parray->cDims == 2)
		{
			_Ni = Mat.parray->rgsabound[1].cElements;
			_Nj = Mat.parray->rgsabound[0].cElements;
		}
		else if(Mat.parray->cDims == 1)
		{
			_Ni = Mat.parray->rgsabound[0].cElements;
			_Nj = 1;
		}
		//
	
		double *data = 0;
		SafeArrayAccessData(Mat.parray,(void**)&data);

		FREE_DATA;
		_data = (double*)malloc(_Ni*_Nj*sizeof(double));
		memcpy(_data,data,_Ni*_Nj*sizeof(double));

		SafeArrayUnaccessData(Mat.parray);

		data = 0;

		_type = General;

		return *this;
	}
#endif

	FMatrix& FMatrix::operator=(const CMatrix<double>& Mat)
	{
		if(Mat.GetLength() == 0)
		{
			Clear();
			return *this;
		}

		if(_Ni != Mat.GetNRows() || _Nj != Mat.GetNColumns())
		{
			_Ni = Mat.GetNRows();
			_Nj = Mat.GetNColumns();

			_data = (double*)realloc(_data,_Ni*_Nj*sizeof(double));
		}

		memcpy_s(_data,_Ni*_Nj*sizeof(double),Mat.GetDataPtr(),_Ni*_Nj*sizeof(double));

		_type = General;

		return *this;
	}

	FMatrix& FMatrix::operator= (const double& Val)
	{
		if(_Ni != 1 || _Nj != 1)
		{
			_Ni = 1;
			_Nj = 1;

			_data = (double*)realloc(_data,_Ni*_Nj*sizeof(double));
		}

		_data[0] = Val;

		_type = General;

		return *this;
	}

	FMatrix& FMatrix::operator= (const float& Val)
	{
		if(_Ni != 1 || _Nj != 1)
		{
			_Ni = 1;
			_Nj = 1;

			_data = (double*)realloc(_data,_Ni*_Nj*sizeof(double));
		}

		_data[0] = Val;

		_type = General;

		return *this;
	}

	FMatrix& FMatrix::operator= (const INT& Val)
	{
		if(_Ni != 1 || _Nj != 1)
		{
			_Ni = 1;
			_Nj = 1;

			_data = (double*)realloc(_data,_Ni*_Nj*sizeof(double));
		}

		_data[0] = (double)Val;

		_type = General;

		return *this;
	}

	FMatrix& FMatrix::operator= (const unsigned INT& Val)
	{
		if(_Ni != 1 || _Nj != 1)
		{
			_Ni = 1;
			_Nj = 1;

			_data = (double*)realloc(_data,_Ni*_Nj*sizeof(double));
		}

		_data[0] = (double)Val;

		_type = General;

		return *this;
	}

	FMatrix& FMatrix::operator+= (const FMatrix& B)
	{
		if(_Ni == 0 || _Nj == 0)
		{
			FMatrix::operator =(B);
		}
		else if (B._Ni == 0 || B._Nj == 0)
		{
			return *this;
		}
		else
		{
			if(_Ni != B._Ni || _Nj != B._Nj)
				THROWERROR(L"operator+=: Matrices must have the same dimensions\r\n");

			vdAdd(_Ni*_Nj,_data,B._data,_data);

			_type = OperationResType(B);
		}

		return *this;
	}
	FMatrix& FMatrix::operator-=(const FMatrix& B)
	{
		if(_Ni == 0 || _Nj == 0)
		{
			FMatrix::operator =(-B);
		}
		else
		{
			if(_Ni != B._Ni || _Nj != B._Nj)
				THROWERROR(L"operator-=: Matrices must have the same dimensions\r\n");

			vdSub(_Ni*_Nj,_data,B._data,_data);

			_type = OperationResType(B);
		}

		return *this;
	}
	FMatrix& FMatrix::operator*=(FMatrix& B)
	{
		if(_Nj != B._Ni)
			THROWERROR(L"operator*=: Matrices square matrices with the same dimensions\r\n");

		*this = *this * B;

		_type = OperationResType(B);

		return *this;
	}

	FMatrix& FMatrix::operator /=(double B)
	{
		*this = *this/B;

		return *this;
	}

#ifndef __GNUC__
	FMatrix::operator VARIANT() const
	{
		VARIANT res;

		if(_Ni > 1 && _Nj >1)
		{
			SAFEARRAYBOUND dims[2];

			dims[0].lLbound =0;
			dims[0].cElements = (ULONG)_Ni;
			dims[1].lLbound =0;
			dims[1].cElements = (ULONG)_Nj;

			SAFEARRAY FAR* safearray;

			safearray = SafeArrayCreate(VT_R8,2,dims);

			double* dados;
			SafeArrayAccessData(safearray,(void**)&dados);

			memcpy(dados,_data,sizeof(double)*_Ni*_Nj);

			res.parray = safearray;
			res.vt = VT_ARRAY|VT_R8|VT_R8;

			SafeArrayUnaccessData(safearray);

			safearray = 0;
			dados = 0;

			return res;
		}
		else
		{
			SAFEARRAYBOUND dims[1];

			dims[0].lLbound =0;
			dims[0].cElements = (ULONG)((_Nj==1)?_Ni:_Nj);

			SAFEARRAY FAR* safearray;

			safearray = SafeArrayCreate(VT_R8,1,dims);

			double* dados;
			SafeArrayAccessData(safearray,(void**)&dados);

			memcpy(dados,_data,sizeof(double)*_Ni*_Nj);

			res.parray = safearray;
			res.vt = VT_ARRAY|VT_R8|VT_R8;
		
			SafeArrayUnaccessData(safearray);

			safearray = 0;
			dados = 0;

			return res;
		}
	}
#endif

	FMatrix::operator CMatrix<double>() const
	{
		return CMatrix<double>(_Ni,_Nj,_data);
	}

	void FMatrix::DestroyMatrix()
	{
		_Ni = 0;
		_Nj = 0;

		FREE(_data);

		_type = General;
	}

	bool FMatrix::CheckType(MatrixType type)
	{
		if(type == UpperTriangular)
		{
			for(unsigned INT j = 0; j < _Nj; j++)
			{
				for(unsigned INT i = j+1 ; i < _Ni; i++)
				{
					if(fabs(_data[(size_t)i+(size_t)j*(size_t)_Ni]) > 1e-25)
						return false;
				}
			}
		}
		else if(type == LowerTriangular)
		{
			for(unsigned INT i = 0; i < _Ni; i++)
			{
				for(unsigned INT j = i+1 ; j < _Nj; j++)
				{
					if(fabs(_data[(size_t)i+(size_t)j*(size_t)_Ni]) > 1e-25)
						return false;
				}
			}
		}
		else if(type == Diagonal)
		{
			for(unsigned INT i = 0; i < _Ni; i++)
			{
				for(unsigned INT j = i+1 ; j < _Nj; j++)
				{
					if(i == j)
						continue;

					if(fabs(_data[(size_t)i+(size_t)j*(size_t)_Ni]) > 1e-25)
						return false;
				}
			}
		}

		return true;
	}

	MatrixType FMatrix::OperationResType(const FMatrix& B) const
	{
		if(_type == B._type)
			return _type;
		else if(_type == General || B._type == General)
			return General;
		else if((_type == LowerTriangular && B._type == UpperTriangular) || (_type == UpperTriangular && B._type == LowerTriangular))
		{
			bool sym = true;
			FMatrix TB = T(B);
			for(unsigned INT i=0;i<B.GetLength();i++)
			{
				if(_data[i] != TB(i))
				{
					sym = false;
					break;
				}
			}
			
			if(sym)
				return Symmetric;
			else
				return General;
		}
		else if(_type == Diagonal)
		{
			if(B._type == Symmetric)
				return General;
			else
				return B._type;
		}
		else if(B._type == Diagonal)
		{
			if(_type == Symmetric)
				return General;
			else
				return _type;
		}
		else
			return General;
	}

	//Calculates the Inverse of matrix A
	FMatrix Inverse(const FMatrix &A)
	{
		if(A._Ni != A._Nj)
			THROWERROR(L"Inverse: The matrix A is not a square matrix\r\n");

		INT info = 0;

		if(A._type == General || A._type == Symmetric)
		{
			double* LU = (double*)malloc(A._Ni*A._Nj*sizeof(double));
			memcpy_s(LU,A._Ni*A._Nj*sizeof(double),A._data,A._Ni*A._Nj*sizeof(double));
			INT *ipiv = (INT*)malloc(min(A._Ni,A._Nj)*sizeof(INT));

			INT lwork = A._Ni*64;
			double *work = (double*)malloc(lwork*sizeof(double));

			INT m = A._Ni,
				n = A._Nj;

			dgetrf(&m,&n,LU,&m,ipiv,&info);
			
			FMatrix C(A._Ni,A._Nj,LU);
			INT CNi = C._Ni, CNj = C._Nj;

			dgetri(&CNi,C._data,&CNi,ipiv,work,&lwork,&info);

			free(work);
			work = 0;

			free(LU);
			LU = 0;

			free(ipiv);
			ipiv = 0;

			if(info > 0)
			{
				THROWERROR(L"Inverse: A is a Singular Matrix\r\n");
			}

			return C;
		}
		else if(A._type == UpperTriangular ||A. _type == LowerTriangular)
		{
			FMatrix C(A);
			INT CNi = C._Ni, CNj = C._Nj;

			char uplo = (A._type == UpperTriangular)?'U':'L',
				 diag = 'N';

			dtrtri(&uplo,&diag,&CNi,C._data,&CNi,&info);

			if(info > 0)
			{
				THROWERROR(L"Inverse: A is a Singular Matrix\r\n");
			}

			return C;
		}
		else if(A._type == Diagonal)
		{
			FMatrix C(A._Ni,A._Nj);
		
			for(unsigned INT i =0;i<A._Ni;i++)
			{
				if(A(i,i) == 0)
					THROWERROR(L"Solve: A is a singular matrix\r\n");

				C(i,i) = 1/A(i,i);

			}		

			C._type = Diagonal;

			return C;
		}
		else
		{
			return EmptyMatrix;
		}
	}

	//Calculates the determinant of A
	double Det(const FMatrix &A)
	{
		if(A._Ni != A._Nj)
			THROWERROR(L"Det: The matrix A is not a square matrix\r\n");

		double det = 1;

		if(A._type == General)
		{
			INT info = 0,
				m = A._Ni,
				n = A._Nj;

			double* LU = (double*)malloc(A._Ni*A._Nj*sizeof(double));
			memcpy_s(LU,A._Ni*A._Nj*sizeof(double),A._data,A._Ni*A._Nj*sizeof(double));

			INT *ipiv = (INT*)malloc(min(A._Ni,A._Nj)*sizeof(INT));

			dgetrf(&m,&n,LU,&m,ipiv,&info);

			for(unsigned INT i =0; i < A._Ni; i++)
			{
				det *= LU[i+A._Ni*i];

				if(ipiv[i] != (i+1))
					det *= -1;
			}


			FREE(LU);
			FREE(ipiv);
		}
		else if(A._type == UpperTriangular || A._type == LowerTriangular || A._type == Diagonal)
		{
			for(unsigned INT i =0; i < A._Ni; i++)
			{
				det *= A(i,i);
			}
		}

		return det;
	}

	double ConditionNumber(const FMatrix &A, bool infinity_norm)
	{
		if(A._Ni != A._Nj)
			THROWERROR(L"ConditionNumber: Matrix A must be a square Matrix\r\n");


		INT info = 0;
		INT m = A._Ni,
			n = A._Nj,
			lda = A._Ni;

		double *LU = (double*)malloc((size_t)A._Ni*(size_t)A._Nj*sizeof(double));
		memcpy_s(LU,(size_t)A._Ni*(size_t)A._Nj*sizeof(double),A._data,(size_t)A._Ni*(size_t)A._Nj*sizeof(double));

		INT *ipiv = (INT*)malloc(min(A._Ni,A._Nj)*sizeof(INT));
		INT *iwork = (INT*)malloc(m*sizeof(INT));
		double *work = (double*)malloc(4*m*sizeof(double));

		dgetrf(&m,&n,LU,&lda,ipiv,&info);

		double anorm = Max(Sum(abs(A),!infinity_norm));

		char norm = (infinity_norm)?'O':'I';


		double rcond = 0;

		dgecon(&norm,&m,LU,&lda,&anorm,&rcond,work,iwork,&info);

		FREE(LU);
		FREE(ipiv);
		FREE(iwork);
		FREE(work);

		return rcond;

	}


	void LUDecomposition(FMatrix &A,FMatrix *P, FMatrix *L, FMatrix *U)
	{
		INT info = 0;

		double *LU = (double*)malloc((size_t)A._Ni*(size_t)A._Nj*sizeof(double));
		memcpy_s(LU,(size_t)A._Ni*(size_t)A._Nj*sizeof(double),A._data,(size_t)A._Ni*(size_t)A._Nj*sizeof(double));

		INT *ipiv = (INT*)malloc(min(A._Ni,A._Nj)*sizeof(INT));

		INT ANi = A._Ni, ANj = A._Nj;
		dgetrf(&ANi,&ANj,LU,&ANi,ipiv,&info);

		if(A._Ni == A._Nj)
		{
			P->Initialize(A._Ni,A._Ni);
			L->Initialize(A._Ni,A._Nj);
			U->Initialize(A._Ni,A._Nj);

			for(unsigned INT i=0; i < A._Ni; i++)
			{
				for(unsigned INT j =0; i<A._Nj; j++)
				{
					if(i == j)
					{
						(*L)(i,j) = 1;
						(*U)(i,j) = LU[i+A._Ni*j];
					}
					else if(i > j)
					{
						(*L)(i,j) = LU[i+A._Ni*j];
						(*U)(i,j) = 0;
					}
					else
					{
						(*L)(i,j) = 0;
						(*U)(i,j) = LU[i+A._Ni*j];
					}
				}
			}

		
		}
		else if(A._Ni > A._Nj)
		{
			P->Initialize(A._Ni,A._Nj);
			L->Initialize(A._Ni,A._Nj);
			U->Initialize(A._Nj,A._Nj);
		}
		else
		{
			P->Initialize(A._Ni,A._Nj);
			L->Initialize(A._Ni,A._Ni);
			U->Initialize(A._Ni,A._Nj);
		}

		FREE(LU);
		FREE(ipiv);
	}

	void CholeskyDecomposition(const FMatrix& A, FMatrix& L)
	{
		L = A;

		INT info = 0;
		INT n = A._Ni, lda = A._Ni;
		char uplo = 'L';

		dpotrf(&uplo,&n,L._data,&lda,&info);

		if(info > 0)
		{
			L.Initialize(0,0);
			return;
		}

		for(unsigned INT i=0;i<L._Ni;i++)
		{
			for(unsigned INT j =i+1;j<L._Nj;j++)
			{
				L(i,j) = 0;
			}
		}

		L._type = LowerTriangular;
	}

	void QRDecomposition(const FMatrix& A, FMatrix* Q, FMatrix* R, bool simple)
	{
		if(Q == 0 && R == 0)
			return;

		FMatrix QR = A;

		INT m = A._Ni;
		INT n = A._Nj;
		INT lda = A._Ni;
		INT info = 0;
		INT k = min(A._Nj,A._Ni);

		double* tau = (double*)calloc(min(A._Nj,A._Ni),sizeof(double));
		double* work =(double*)calloc(A._Ni*64,sizeof(double));
		INT lwork = A._Ni*64;

		dgeqrf(&m,&n,QR._data,&lda,tau,work,&lwork,&info);

		if(R != 0)
		{
			*R = A;

			char side = 'L';
			char trans = 'T';

			dormqr(&side,&trans,&m,&n,&k,QR._data,&lda,tau,R->_data,&lda,work,&lwork,&info);

			R->_type = UpperTriangular;

			for(unsigned INT j=0;j<R->_Nj-1;j++)
			{
				for(unsigned INT i = j+1;i<R->_Ni;i++)
				{
					(*R)(i,j) = 0;
				}
			}
		}

		if(Q != 0)
		{
			dorgqr(&m,&k,&k,QR._data,&lda,tau,work,&lwork,&info);

			*Q = QR.GetRange(0,m-1,0,k-1);
		}

		if(m > n)
		{
			if(simple)
			{
				FMatrix Rtemp = R->GetRange(0,k-1,0,k-1);
				
				*R = Rtemp;
			}
			else
			{
				FMatrix Qt = *Q;
				INT l = n;
				for(INT i=n; i < m; i++)
				{
					FMatrix Qp = Qt.GetRange(0,l-1,0,l-1);
					FMatrix Qpp = Qt.GetRange(l,m-1,0,l-1);

					FMatrix Qlpp(m-l,1);
					for(unsigned INT j=0;j<Qlpp.GetNRows();j++)
					{
						Qlpp(j) = (double)(j+1);
					}

					FMatrix Qlp = Solve(T(Qp),-T(Qpp)*Qlpp,Auto);

					FMatrix Ql = Normalize(VCat(Qlp,Qlpp));

					Qt = HCat(Qt,Ql);
					l++;
				}

				*Q = Qt;
			}
		}

		free(tau);
		tau = 0;

		free(work);
		work = 0;
	}

	FMatrix Eigenvalues(const FMatrix& A)
	{
		char jobvl = 'N', jobvr = 'N';

		FMatrix A_temp(A),eigR(A._Ni,1),eigI(A._Ni,1);

		INT n = A._Ni,
			lda = A._Ni,
			ldvl = max(1,n),
			ldvr = max(1,n),
			info = 0;

		INT lwork = 3*n;
		double *work = new double[lwork];

		dgeev(&jobvl, &jobvr, &n, A_temp._data, &lda, eigR._data, eigI._data, 0, &ldvl, 0, &ldvr, work, &lwork, &info);

		delete[] work;
		work = 0;

		return HCat(eigR,eigI);
	}

	void EigenSystem(const FMatrix& A, FMatrix& LeftEV, FMatrix& EigenValues, FMatrix& RightEV)
	{
		char jobvl = 'V', jobvr = 'V';

		FMatrix A_temp(A),eigR(A._Ni,1),eigI(A._Ni,1);

		INT n = A._Ni,
			lda = A._Ni,
			ldvl = max(1,n),
			ldvr = max(1,n),
			info = 0;

		LeftEV.Initialize(n,n);
		RightEV.Initialize(n,n);

		INT lwork = 4*n;
		double *work = new double[lwork];

		dgeev(&jobvl, &jobvr, &n, A_temp._data, &lda, eigR._data, eigI._data, LeftEV._data, &ldvl, RightEV._data, &ldvr, work, &lwork, &info);

		delete[] work;
		work = 0;

		EigenValues =  HCat(eigR,eigI);
	}

	MatrixDefinition CheckDefinition(const FMatrix& A, FMatrix *eigenvalues)
	{
		FMatrix eig = Eigenvalues(A);
		INT type;

		if(Norm(eig.GetColumn(1)) > 1e-15)
			return Indefinite;

		//types: 0 -> semidefinite, 1 -> positivedefinite, 2 -> negativedefinite, 3-> Psemidefinite, 4-> Nsemidefinite, 5 -> indefinite

		if(eig(0) > 0)
			type = 1;
		else if(eig(0) < 0)
			type = 2;
		else
			type = 0;

		for(unsigned INT i = 1; i < eig._Ni; i++)
		{
			if(eig(i) > 0 && (type == 2 || type == 4))
			{
				type = 5;
				break;
			}
			else if(eig(i) > 0 && type == 0)
			{
				type = 3;
			}
			else if(eig(i) < 0 && (type == 1 || type == 3))
			{
				type = 5;
				break;
			}
			else if(eig(i) < 0 && type == 0)
			{
				type = 4;
			}
			else if(fabs(eig(0)) < 1e-15 && type == 1)
			{
				type = 3;
			}
			else if(fabs(eig(0)) < 1e-15 && type == 2)
			{
				type = 4;
			}
		}

		if(eigenvalues != 0)
			*eigenvalues = eig;

		switch(type)
		{
			case 1:
				return PositiveDefinite;
			case 2:
				return NegativeDefinite;
			case 3:
				return PositiveSemiDefinite;
			case 4:
				return NegativeSemiDefinite;
			case 5:
				return Indefinite;
			default:
				return Singular;
		}
	}

	FMatrix operator*(INT A, const FMatrix& B)
	{
		return B*A;
	}

	FMatrix operator*(double A, FMatrix& B)
	{
		return B*A;
	}

	FMatrix operator*(double A, const FMatrix& B)
	{
		return B*A;
	}

	FMatrix operator*(float A, const FMatrix& B)
	{
		return B*A;
	}

	FMatrix operator|(INT A,const FMatrix& B)
	{
		FMatrix a(B._Ni,B._Nj,(double)A,false);

		return a|B;
	}

	FMatrix operator|(double A,const FMatrix& B)
	{
		FMatrix a(B._Ni,B._Nj,A,false);

		return a|B;
	}

	FMatrix operator+(INT A, const FMatrix& B)
	{
		return B+A;
	}

	FMatrix operator+(double A,const FMatrix& B)
	{
		return B+A;
	}

	FMatrix operator-(INT A, const FMatrix& B)
	{
		FMatrix a(B._Ni,B._Nj,(double)A,false);

		return a-B;
	}
	FMatrix operator-(double A, const FMatrix& B)
	{
		FMatrix a(B._Ni,B._Nj,A,false);

		return a-B;
	}

	//Matrix transpose.
	FMatrix T(const FMatrix& A)
	{
		FMatrix C(A._Nj,A._Ni);

		if(A._Ni != 1 && A._Nj != 1)
		{
			for(unsigned INT j=0;j<C.GetNColumns();j++)
				for(unsigned INT i=0;i<C.GetNRows();i++)
					C(i,j) = A(j,i);

			if(A._type == UpperTriangular)
				C._type = LowerTriangular;
			else if(A._type == LowerTriangular)
				C._type = UpperTriangular;
			else
				C._type = A._type;
		}
		else
		{
			memcpy_s(C._data,(size_t)A._Ni*(size_t)A._Nj*sizeof(double),A._data,(size_t)A._Ni*(size_t)A._Nj*sizeof(double));
		}

		return C;
	}

	FMatrix sin(const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdSin(n,_X._data,res._data);
		
		res._type = _X._type;

		return res;
	}

	FMatrix cos(const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdCos(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix tan(const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdTan(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix asin(FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdAsin(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix acos(FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdAcos(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix atan(FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdAtan(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix atan2(const FMatrix& _Y,const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdAtan2(n,_Y._data,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	//Hyperbolic tangent of the elements of _X
	FMatrix tanh(FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdTanh(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	//Absolute of the elements of _X
	FMatrix abs(const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdAbs(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	//Exponential of the elements of _X
	FMatrix exp(FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdExp(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix log(FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X._Ni*_X._Nj;

		vdLn(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	//Logistic sigma of the elements of _X
	FMatrix logsigma(FMatrix& _X)
	{
		FMatrix res=_X;

		res = 1.0|(1.0+exp(res));

		res._type = _X._type;
	
		return res;
	}

	//Element-wise square-root of a matrix
	FMatrix sqrt(FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X.GetLength();

		vdSqrt(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix cosh(const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X.GetLength();

		vdCosh(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	FMatrix sinh(const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		INT n = _X.GetLength();

		vdSinh(n,_X._data,res._data);

		res._type = _X._type;

		return res;
	}

	double rms(const FMatrix& _X)
	{
		return std::sqrt(Sum((_X,_X))/_X.GetLength());
	}

	double Max(const FMatrix& _X)
	{
		if(_X.IsEmpty())
			return 0;

		double max_val = _X(0);

		for(unsigned INT i=1;i<_X.GetLength();i++)
		{
			max_val = max(max_val,_X(i));
		}

		return max_val;
	}

	FMatrix Max(const FMatrix& _X, bool byColumns)
	{
		if(_X.IsEmpty())
			return EmptyMatrix;

		FMatrix res;
		if(byColumns)
		{
			res.Initialize(1,_X._Nj);

			for(unsigned INT j=0;j<_X._Nj;j++)
			{
				res(j) = Max(_X.GetColumn(j));
			}

			return res;
		}
		else
		{
			res.Initialize(_X._Ni,1);

			for(unsigned INT i=0;i<_X._Ni;i++)
			{
				res(i) = Max(_X.GetLine(i));
			}

			return res;
		}
	}

	double Min(const FMatrix& _X)
	{
		if(_X.IsEmpty())
			return 0;

		double min_val = _X(0);

		for(unsigned INT i=1;i<_X.GetLength();i++)
		{
			min_val = min(min_val,_X(i));
		}

		return min_val;
	}

	FMatrix Min(const FMatrix& _X, bool byColumns)
	{
		if(_X.IsEmpty())
			return EmptyMatrix;

		FMatrix res;
		if(byColumns)
		{
			res.Initialize(1,_X._Nj);

			for(unsigned INT j=0;j<_X._Nj;j++)
			{
				res(j) = Min(_X.GetColumn(j));
			}

			return res;
		}
		else
		{
			res.Initialize(_X._Ni,1);

			for(unsigned INT i=0;i<_X._Ni;i++)
			{
				res(i) = Min(_X.GetLine(i));
			}

			return res;
		}
	}

	//Computes the screw matrix from a 3D vector
	FMatrix Screw(const FMatrix& A)
	{
		FMatrix C(3,3);

		if(A.GetNRows()*A.GetNColumns() == 3)
		{

			C(1,0) = A(2);
			C(2,0) = -A(1);

			C(0,1) = -A(2);
			C(2,1) = A(0);

			C(0,2) = A(1);
			C(1,2) = -A(0);
		}

		return C;
	}

	//Computes the 3D vector from a screw matrix
	FMatrix UnScrew(const FMatrix& A)
	{
		FMatrix res(3,1);

		res(0,0) = A(2,1);
		res(1,0) = A(0,2);
		res(2,0) = A(1,0);

		return res;
	}

	//Dot product of two vectors
	double Dot(const FMatrix& A,const  FMatrix& B)
	{
		if(A._Ni*A._Nj != B._Ni*B._Nj)
			THROWERROR(L"The two vectors must have the same length\r\n");

		return cblas_ddot(A._Ni*A._Nj,A._data,1,B._data,1);
	}

	FMatrix Cross(const FMatrix& A,const  FMatrix& B)
	{
		if(A._Ni*A._Nj != B._Ni*B._Nj)
			THROWERROR(L"The two vectors must have the same length\r\n");
		if(A.GetLength() != 3 || B.GetLength() != 3)
			THROWERROR(L"The two vectors must be 3D vectors\r\n");

		FMatrix cross(3,1);

		cross(0) = A(1)*B(2)-A(2)*B(1);
		cross(1) = A(2)*B(0)-A(0)*B(2);
		cross(2) = A(0)*B(1)-A(1)*B(0);

		return cross;
	}

	//Euclidean norm of a vector
	double Norm(const FMatrix& A)
	{
		return cblas_dnrm2(A._Ni*A._Nj,A._data,1);
	}

	FMatrix Normalize(const FMatrix& A)
	{
		FMatrix res = A/Norm(A);

		return res;
	}

	ostream& operator<<(ostream& in,FMatrix &Mat)
	{
		ostream& out = in;

		char temp[20];

		out<<endl;
		for(unsigned INT i=0;i<Mat._Ni;i++)
		{
			for(unsigned INT j=0;j<Mat._Nj;j++)
			{
				sprintf_s(temp,20,"%s%6e\t",(Mat(i,j)<0)?"-":" ",fabs(Mat(i,j)));
				out<<temp;
			}

			out<<endl;
		}

		return out;

	}

	ofstream& operator<<(ofstream& out, FMatrix &Mat)
	{
		out<<"Ni= "<<Mat._Ni<<" Nj= "<<Mat._Nj<<" ";

		out.precision(16);
		for(unsigned INT i=0;i<Mat._Ni*Mat._Nj;i++)
		{
			out<<scientific<<Mat(i)<<" ";
		}

		return out;
	}

	wofstream& operator<<(wofstream& out,FMatrix &Mat)
	{
		out<<"Ni= "<<Mat._Ni<<" Nj= "<<Mat._Nj<<" ";

		out.precision(16);
		for(unsigned INT i=0;i<Mat._Ni*Mat._Nj;i++)
		{
			out<<scientific<<Mat(i)<<" ";
		}

		return out;
	}

	ifstream& operator>>(ifstream& in,FMatrix &Mat)
	{
		char temp[20];

		INT ni = 0,no = 0;

		in>>temp>>ni>>temp>>no;

		double *data = (double*)malloc(ni*no*sizeof(double));

		for(INT i = 0; i < ni*no; i++)
		{
			in>>data[i];
		}

		Mat = FMatrix(ni,no,data);

		free(data);
		data = 0;

		return in;
	}

	wifstream& operator>>(wifstream& in,FMatrix &Mat)
	{
		wchar_t temp[20];

		INT ni = 0,no = 0;

		in>>temp>>ni>>temp>>no;

		double *data = (double*)malloc(ni*no*sizeof(double));

		for(INT i = 0; i < ni*no; i++)
		{
			in>>data[i];
		}

		Mat = FMatrix(ni,no,data);

		free(data);
		data = 0;

		return in;
	}

	void EquilibrateMatrix(const FMatrix& A,FMatrix& R, FMatrix& C, bool PowerRadix)
	{
		INT m = A._Ni,
			n = A._Nj,
			lda = m,
			info = 0;

		double rowcnd = 0,
			   colcnd = 0,
			   amax = 0;

		double *r = (double*)malloc(m*sizeof(double)),
			   *c = (double*)malloc(n*sizeof(double));

		if(!PowerRadix)
			dgeequ(&m,&n,A._data,&lda,r,c,&rowcnd,&colcnd,&amax,&info);
		else
			dgeequb(&m,&n,A._data,&lda,r,c,&rowcnd,&colcnd,&amax,&info);

		if(info > 0)
		{
			if(info <= m)
			{
				wchar_t msg[100];
				swprintf_s(msg,100,L"EquilibateMatrix: row %d of A is exactly zero\r\n",(int)info-1);
				FREE(r);
				FREE(c);
				THROWERROR(msg);
			}
			else
			{
				wchar_t msg[100];
				swprintf_s(msg,100,L"EquilibateMatrix: column %d of A is exactly zero\r\n",(int)(info-m));
				FREE(r);
				FREE(c);
				THROWERROR(msg);
			}
		}

		if(rowcnd < 0.1)
		{
			R.Initialize(m,m);
			for(INT i=0;i<m;i++)
				R(i,i) = r[i];

			R._type = Diagonal;
		}
		else
			R = EmptyMatrix;

		if(colcnd < 0.1)
		{
			C.Initialize(n,n);
			for(INT i=0;i<n;i++)
				C(i,i) = c[i];

			C._type = Diagonal;
		}
		else
			C = EmptyMatrix;

		FREE(r);
		FREE(c);
	}

	FMatrix Solve(const FMatrix& A,const FMatrix& B, unsigned INT Options)
	{
		bool auto_select = false,
			 force_LU = false,
			 force_QR = false,
			 force_CL = false,
			 equilibrate = false,
			 square_matrix = (A._Ni == A._Nj)?true:false;

		INT refine_level = 0; //0 no refinement, 1 simple refine, 2 extra refine

		INT m = A._Ni,
			n = A._Nj,
			k = B._Nj,
			lda = m,
			info = 0;

		char trans = 'N',
			 diag = 'N',
			 equed = 'N';

		double rowcnd = 1,
			   colcnd = 1,
			   amax = 0,
			   rcond = 0;

		double *r = 0,
			   *c = 0,
			   *LU = 0;

		INT *ipiv = 0;

		FMatrix R,C,Y(B),X;

		if((Options & Auto) != 0)
		{
			auto_select = true;

			if(square_matrix)
			{
				force_LU = true;

				if((Options & Refine) != 0)
					refine_level = 1;

				if((Options & ExtraRefine) != 0)
					refine_level = 2;
			}
			else
			{
				force_QR = true;

				if((Options & Equilibrate) != 0)
				equilibrate = true;
			}
		}
		else
		{
			if((Options & LUdecomposition) != 0)
			{
				force_LU = true;

				if((Options & DECOMP_METHODS) > LUdecomposition)
					THROWERROR(L"Solve: Multiple decomposition methods chosen\r\n");
			}
			else if((Options & QRdecomposition) != 0)
			{
				force_QR = true;

				if((Options & DECOMP_METHODS) > QRdecomposition)
					THROWERROR(L"Solve: Multiple decomposition methods chosen\r\n");
			}
			else if((Options & Choleskydecomposition) != 0)
			{
				force_CL = true;

				if((Options & DECOMP_METHODS) > Choleskydecomposition)
					THROWERROR(L"Solve: Multiple decomposition methods chosen\r\n");
			}

			if(force_LU && force_QR)
				THROWERROR(L"Solve: Only one type of decomposition can be chosen (LU or QR)\r\n")
			else if(force_LU && !square_matrix)
				THROWERROR(L"Solve: LU decompositon is only available for square system matrices\r\n")
			else if(!force_LU && !force_QR)
			{
				if(square_matrix)
					force_LU = true;
				else
					force_QR = true;
			}

			if((Options & Equilibrate) != 0)
				equilibrate = true;

			if((Options & Refine) != 0)
				refine_level = 1;

			if((Options & ExtraRefine) != 0)
				refine_level = 2;
		}

		if(force_LU)
		{
			//Equilibrate matrix
			if(equilibrate || auto_select)
			{
				r = (double*)malloc(m*sizeof(double));
				c = (double*)malloc(n*sizeof(double));

				dgeequb(&m,&n,A._data,&lda,r,c,&rowcnd,&colcnd,&amax,&info);

				if(info > 0)
				{
					if(info <= m)
					{
						wchar_t msg[100];
						swprintf_s(msg,100,L"Solve: row %d of A is exactly zero\r\n",(int)info-1);
						FREE(r);
						FREE(c);
						THROWERROR(msg);
					}
					else
					{
						wchar_t msg[100];
						swprintf_s(msg,100,L"Solve: column %d of A is exactly zero\r\n",(int)(info-m));
						FREE(r);
						FREE(c);
						THROWERROR(msg);
					}
				}

				R.Initialize(m,1,r);
				C.Initialize(n,1,c);
				R = VectorToDiagonalMatrix(R);
				C = VectorToDiagonalMatrix(C);

				FREE(r);
				FREE(c);
			}
			//

			//Solve System
			if(A._type == General || A._type == Diagonal || A._type == Symmetric)
			{
				//Decompose matrix
				LU = (double*)malloc(A._Ni*A._Nj*sizeof(double));
				//memcpy_s(LU,A._Ni*A._Nj*sizeof(double),A._data,A._Ni*A._Nj*sizeof(double));

				if(rowcnd < 0.1 && colcnd < 0.1)
				{
					/*for(INT i = 0; i<A._Ni; i++)
					{
						for(INT j =0; j<A._Nj; j++)
							LU[i+j*A._Ni] = R(i,i)*LU[i+j*A._Ni]*C(j,j);

						Y.Line[i] = R(i,i)*Y.Line[i];
					}*/
					FMatrix Atemp = R*A*C;
					Y = R*Y;

					memcpy_s(LU,A._Ni*A._Nj*sizeof(double),Atemp._data,A._Ni*A._Nj*sizeof(double));
				}
				else if(rowcnd > 0.1 && colcnd < 0.1)
				{
					/*for(INT j = 0; j<A._Nj; j++)
					{
						for(INT i =0; i<A._Ni; i++)
							LU[i+j*A._Ni] = LU[i+j*A._Ni]*C(j,j);
					}*/
					FMatrix Atemp = A*C;

					memcpy_s(LU,A._Ni*A._Nj*sizeof(double),Atemp._data,A._Ni*A._Nj*sizeof(double));
				}
				else if(rowcnd < 0.1 && colcnd > 0.1)
				{
					/*for(INT i = 0; i<A._Ni; i++)
					{
						for(INT j =0; j<A._Nj; j++)
							LU[i+j*A._Ni] = R(i,i)*LU[i+j*A._Ni];

						Y.Line[i] = R(i,i)*Y.Line[i];
					}*/
					FMatrix Atemp = R*A;
					Y = R*Y;

					memcpy_s(LU,A._Ni*A._Nj*sizeof(double),Atemp._data,A._Ni*A._Nj*sizeof(double));
				}
				else
				{
					memcpy_s(LU,A._Ni*A._Nj*sizeof(double),A._data,A._Ni*A._Nj*sizeof(double));
				}

				ipiv = (INT*)malloc(min(A._Ni,A._Nj)*sizeof(INT));

				dgetrf(&m,&n,LU,&lda,ipiv,&info);
				dgetrs(&trans,&m,&k,LU,&n,ipiv,Y._data,&n,&info);

				if(isNaN(Sum(Y)))
				{
					FREE(LU);
					FREE(ipiv);

					return EmptyMatrix;
				}

				if(colcnd < 0.1)
				{
					X.Initialize(Y._Ni,Y._Nj);
					for(unsigned INT i = 0; i<A._Ni; i++)
						X.ReplaceLine(i,C(i,i)*Y.GetLine(i));
				}
				else
				{
					X = Y;
				}
			}
			else if(A._type == UpperTriangular || A._type == LowerTriangular)
			{
				char uplo = (A._type == UpperTriangular)?'U':'L';

				if(rowcnd < 0.1 && colcnd < 0.1)
				{
					FMatrix Atemp = R*A*C;
					Y = R*Y;
					dtrtrs(&uplo,&trans,&diag,&m,&k,Atemp._data,&n,Y._data,&m,&info);
					X = C*Y;

					Atemp.Clear();
				}
				else if(rowcnd > 0.1 && colcnd < 0.1)
				{
					FMatrix Atemp = A*C;
					dtrtrs(&uplo,&trans,&diag,&m,&k,Atemp._data,&n,Y._data,&m,&info);
					X = C*Y;
					Atemp.Clear();
				}
				else if(rowcnd < 0.1 && colcnd > 0.1)
				{
					FMatrix Atemp = R*A;
					Y = R*B;
					dtrtrs(&uplo,&trans,&diag,&m,&k,Atemp._data,&n,X._data,&m,&info);
					Atemp.Clear();

					X = Y;
				}
				else
				{
					dtrtrs(&uplo,&trans,&diag,&m,&k,A._data,&n,Y._data,&m,&info);
					X=Y;
				}

				if(isNaN(Sum(X)))
				{
					return EmptyMatrix;
				}
			}

			INT *iwork = (INT*)malloc(m*sizeof(INT));
			double *work = (double*)malloc(5*m*sizeof(double));

			//Find the condition number of A
			if(auto_select && refine_level == 0)
			{
				char norm = 'O';

				if(A._type == General|| A._type == Diagonal|| A._type == Symmetric)
				{
					//determine the 1-norm of A
					double anorm = 0;

					if(rowcnd < 0.1 && colcnd < 0.1)
						anorm = Max(Sum(abs(R*A*C),true));
					else if(rowcnd > 0.1 && colcnd < 0.1)
						anorm = Max(Sum(abs(A*C),true));
					else if(rowcnd < 0.1 && colcnd > 0.1)
						anorm = Max(Sum(abs(R*A),true));
					else
						anorm = Max(Sum(abs(A),true));

					dgecon(&norm,&m,LU,&lda,&anorm,&rcond,work,iwork,&info);
				}
				else if(A._type == UpperTriangular || A._type == LowerTriangular)
				{
					char uplo = (A._type == UpperTriangular)?'U':'L';

					dtrcon(&norm,&uplo,&diag,&m,A._data,&m,&rcond,work,iwork,&info);
				}

				if(rcond >= 1)						//Well conditioned system
					refine_level = 0;
				else if(rcond < 1 && rcond > 1e-14)	//Ill conditioned system
					refine_level = 1;
				else if(rcond < 1e-14)				//Severely ill conditioned system
					refine_level = 2;
			}
			//

			//Refine solution
			if(A._type == General|| A._type == Diagonal|| A._type == Symmetric)
			{
				if(refine_level == 1)//refine
				{
					double ferr = 0,
						   berr = 0;

					if(rowcnd < 0.1 && colcnd < 0.1)
					{
						FMatrix Atemp = R*A*C,
								Btemp = R*B;
						dgerfs(&trans,&m,&k,Atemp._data,&lda,LU,&lda,ipiv,Btemp._data,&m,Y._data,&m,&ferr,&berr,work,iwork,&info);
						Atemp.Clear();
						Btemp.Clear();
						X = C*Y;
					}
					else if(rowcnd > 0.1 && colcnd < 0.1)
					{
						FMatrix Atemp = A*C;
						dgerfs(&trans,&m,&k,Atemp._data,&lda,LU,&lda,ipiv,B._data,&m,Y._data,&m,&ferr,&berr,work,iwork,&info);
						Atemp.Clear();
						X = C*Y;
					}
					else if(rowcnd < 0.1 && colcnd > 0.1)
					{
						FMatrix Atemp = R*A,
								Btemp = R*B;
						dgerfs(&trans,&m,&k,Atemp._data,&lda,LU,&lda,ipiv,Btemp._data,&m,X._data,&m,&ferr,&berr,work,iwork,&info);
						Atemp.Clear();
						Btemp.Clear();
					}
					else
						dgerfs(&trans,&m,&k,A._data,&lda,LU,&lda,ipiv,B._data,&m,X._data,&m,&ferr,&berr,work,iwork,&info);

				}
				else if(refine_level == 2)//Extra refine
				{
					double ferr = 0,
						   berr = 0;

					double error = 0,
						   ref1_error = 0,
						   ref2_error = 0;

					INT n_err_bnds = 3,
						nparams = 0;

					FMatrix err_bnds_norm(k,n_err_bnds),
							err_bnds_comp(k,n_err_bnds),
							params(3,1);

					params(0) = 1;
					params(1) = 100;
					params(2) = 1;

					FMatrix X_ref1 = Y, X_ref2 = Y;

					if(rowcnd < 0.1 && colcnd < 0.1)
					{
						FMatrix Atemp = R*A*C,
								Btemp = R*B;
						dgerfs(&trans,&m,&k,Atemp._data,&lda,LU,&lda,ipiv,Btemp._data,&m,X_ref1._data,&m,&ferr,&berr,work,iwork,&info);
						X_ref1 = C*X_ref1;

						dgerfsx(&trans,&equed,&m,&k,Atemp._data,&m,LU,&m,ipiv,0,0,Btemp._data,&m,X_ref2._data,&m,&rcond,&berr,&n_err_bnds,err_bnds_norm._data,err_bnds_comp._data,&nparams,params._data,work,iwork,&info);
						X_ref2 = C*X_ref2;

						Atemp.Clear();
						Btemp.Clear();
					}
					else if(rowcnd > 0.1 && colcnd < 0.1)
					{
						FMatrix Atemp = A*C;
						dgerfs(&trans,&m,&k,Atemp._data,&lda,LU,&lda,ipiv,B._data,&m,X_ref1._data,&m,&ferr,&berr,work,iwork,&info);
						X_ref1 = C*X_ref1;

						dgerfsx(&trans,&equed,&m,&k,Atemp._data,&m,LU,&m,ipiv,0,0,B._data,&m,X_ref2._data,&m,&rcond,&berr,&n_err_bnds,err_bnds_norm._data,err_bnds_comp._data,&nparams,params._data,work,iwork,&info);
						X_ref2 = C*X_ref2;

						Atemp.Clear();
					}
					else if(rowcnd < 0.1 && colcnd > 0.1)
					{
						FMatrix Atemp = R*A,
								Btemp = R*B;

						dgerfs(&trans,&m,&k,Atemp._data,&lda,LU,&lda,ipiv,Btemp._data,&m,X_ref1._data,&m,&ferr,&berr,work,iwork,&info);

						dgerfsx(&trans,&equed,&m,&k,Atemp._data,&m,LU,&m,ipiv,0,0,Btemp._data,&m,X_ref2._data,&m,&rcond,&berr,&n_err_bnds,err_bnds_norm._data,err_bnds_comp._data,&nparams,params._data,work,iwork,&info);

						Atemp.Clear();
						Btemp.Clear();
					}
					else
					{
						dgerfs(&trans,&m,&k,A._data,&lda,LU,&lda,ipiv,B._data,&m,X_ref1._data,&m,&ferr,&berr,work,iwork,&info);

						dgerfsx(&trans,&equed,&m,&k,A._data,&m,LU,&m,ipiv,0,0,B._data,&m,X_ref2._data,&m,&rcond,&berr,&n_err_bnds,err_bnds_norm._data,err_bnds_comp._data,&nparams,params._data,work,iwork,&info);
					}

					error = rms(A*X-B);
					ref1_error = rms(A*X_ref1 - B);
					ref2_error = rms(A*X_ref2 - B);

					double min_error = error;

					if(ref1_error < min_error)
					{
						min_error = ref1_error;
						X = X_ref1;
					}

					if(ref2_error < min_error)
					{
						min_error = ref2_error;
						X = X_ref2;
					}
				}
			}
			else if(A._type == UpperTriangular || A._type == LowerTriangular && refine_level > 0)
			{
				double ferr = 0,
					   berr = 0;

				char uplo = (A._type == UpperTriangular)?'U':'L';

				if(rowcnd < 0.1 && colcnd < 0.1)
				{
					FMatrix Atemp = R*A*C,
							Btemp = R*B;

					dtrrfs(&uplo,&trans,&diag,&m,&k,Atemp._data,&m,Btemp._data,&m,Y._data,&m,&ferr,&berr,work,iwork,&info);
					X = C*Y;

					Atemp.Clear();
					Btemp.Clear();
				}
				else if(rowcnd > 0.1 && colcnd < 0.1)
				{
					FMatrix Atemp = A*C;

					dtrrfs(&uplo,&trans,&diag,&m,&k,Atemp._data,&m,B._data,&m,Y._data,&m,&ferr,&berr,work,iwork,&info);
					X = C*Y;

					Atemp.Clear();
				}
				else if(rowcnd < 0.1 && colcnd > 0.1)
				{
					FMatrix Atemp = R*A,
							Btemp = R*B;

					dtrrfs(&uplo,&trans,&diag,&m,&k,Atemp._data,&m,Btemp._data,&m,X._data,&m,&ferr,&berr,work,iwork,&info);

					Atemp.Clear();
					Btemp.Clear();
				}
				else
					dtrrfs(&uplo,&trans,&diag,&m,&k,A._data,&m,B._data,&m,X._data,&m,&ferr,&berr,work,iwork,&info);

			}
			//

			FREE(iwork);
			FREE(work);

			FREE(LU);
			FREE(ipiv);

			return X;
		}
		else if(force_QR)
		{
			if(m >= n)	//Rectangular matrix A[m x n] -> m >= n
			{
			
				FMatrix QR(A), RPT(A);

				INT* jpvt = (INT*)calloc(A._Nj,sizeof(INT));
				double* tau = (double*)calloc(min(A._Nj,A._Ni),sizeof(double));
				double* work =(double*)calloc(A._Ni*64,sizeof(double));
				INT lwork = A._Ni*64;
				INT info = 0,
					m = A._Ni,
					n = A._Nj;

				dgeqp3(&m,&n,QR._data,&m,jpvt,tau,work,&lwork,&info); //QR factorization

				char side = 'L',
					 trans = 'T';

				INT k = min(A._Nj,A._Ni);

				INT RPTNi = RPT._Ni, RPTNj = RPT._Nj;
				dormqr(&side,&trans,&RPTNi,&RPTNj,&k,QR._data,&RPTNi,tau,RPT._data,&RPTNi,work,&lwork,&info); //T(Q)*A = R*T(P)

				FMatrix Q(QR);

				dorgqr(&m,&k,&k,Q._data,&m,tau,work,&lwork,&info); //Q1

				FMatrix C = T(Q)*B;

				if((Options & QRdecomposition) != 0) //remove force QR
				{
					Options = Options & REMOVE_QR;
				}

				X = Solve(RPT.GetRange(0,RPT._Ni - 1- (A._Ni-A._Nj),0,RPT._Nj-1),C,Options); //R*T(P)*X = T(Q1)*B

				FREE(jpvt);
				FREE(tau);
				FREE(work);
			}
			else	//Rectangular matrix A[m x n] -> m < n
			{
				FMatrix QR(A), RPT(A), C(B);

				INT* jpvt = (INT*)calloc(A._Nj,sizeof(INT));
				double* tau = (double*)calloc(min(A._Nj,A._Ni),sizeof(double));
				double* work =(double*)calloc(max(A._Ni,A._Nj)*64,sizeof(double));
				INT lwork = max(A._Ni,A._Nj)*64;
				INT info = 0,
					m = A._Ni,
					n = A._Nj;

				dgeqp3(&m,&n,QR._data,&m,jpvt,tau,work,&lwork,&info); //QR factorization

				char side = 'L',
					 trans = 'T';

				INT k = min(A._Nj,A._Ni);
				INT RPTNi = RPT._Ni, RPTNj = RPT._Nj, CNi = C._Ni, CNj = C._Nj;

				dormqr(&side,&trans,&RPTNi,&RPTNj,&k,QR._data,&RPTNi,tau,RPT._data,&RPTNi,work,&lwork,&info); //T(Q)*A = R*T(P)

				dormqr(&side,&trans,&CNi,&CNj,&k,QR._data,&m,tau,C._data,&CNi,work,&lwork,&info); //C = T(Q)*B

				if((Options & QRdecomposition) != 0) //remove force QR
				{
					Options = Options & REMOVE_QR;
				}

				FMatrix X_temp = Solve(RPT.GetRange(0,RPT._Ni-1,0,RPT._Nj -1 - (A._Nj-A._Ni)),C,Options); //R*T(P)*X = T(Q)*B

				X.Initialize(A._Nj,B._Nj);
				for(unsigned INT i=0;i<X_temp._Ni;i++)
				{
					for(unsigned INT j=0;j<X_temp._Nj;j++)
					{
						X(i,j) = X_temp(i,j);
					}
				}

				FREE(jpvt);
				FREE(tau);
				FREE(work);
			}

			return X;
		}

		return EmptyMatrix;
	}

	//dominante por diagonal e definida positiva
	void SORSolve(const FMatrix &A, const FMatrix &B, FMatrix& X)
	{
		FMatrix error(X._Ni,X._Nj);
				
		error = (A*X - B);
		double merror = Mean(error), merror_ant = merror;

		FMatrix phi(X._Ni,X._Nj),
				phi_ant(X);

		double w = 1.4,sigma=0;

		while(Mean(error) > 1e-10)
		{
			for(unsigned INT i =0; i<phi._Ni;i++)
			{
				sigma = 0;

				for(unsigned INT j =0; j<i;j++)
				{
					sigma += A(i,j)*phi(j);
				}

				for(unsigned INT j =i+1; j<phi._Ni;j++)
				{
					sigma += A(i,j)*phi_ant(j);
				}

				sigma = (B(i) - sigma)/(A(i,i));

				phi(i) = phi_ant(i) + w*(sigma-phi_ant(i));
			}

			phi_ant = phi;

			error = (A*phi - B);
			merror = Mean(error);
		}

		X = phi;
	}

	//Simetricas definidas positivas
	void ConjugateGradientSolve(const FMatrix& A, const FMatrix& B, FMatrix& X)
	{
		FMatrix r,r_ant,p,p_ant;

		r = B - A*X;
		p = r;

		p_ant = p;
		r_ant = r;

		double meanr = Mean(r),a,b;

		while(1)
		{
			a = Dot(r_ant,r_ant)/Dot(p_ant,A*p_ant);
			X = X + a*p_ant;
			r = r - a*A*p_ant;

			meanr = Mean(r);

			if(meanr < 1e-10)
				break;

			b = Dot(r,r)/Dot(r_ant,r_ant);

			p = r + b*p_ant;

			p_ant = p;
			r_ant = r;
		}
	}

	//Reshapes the matrix A to match the specified new line and column size
	FMatrix Reshape(const FMatrix& A,INT NewNi, INT NewNj)
	{
		FMatrix res = A;

		res.Reshape(NewNi,NewNj);

		return res;
	}

	//Gets the submatrix defined by beg_line, end_line, beg_column and end_column
	FMatrix Range(const FMatrix& A, INT beg_line, INT end_line, INT beg_col, INT end_col)
	{
		FMatrix res = A;
		return res.GetRange(beg_line, end_line, beg_col, end_col);
	}

	FMatrix Reverse(const FMatrix& A, bool Columns)
	{
		FMatrix res(A._Ni,A._Nj);

		if(Columns)
		{
			for(unsigned INT j = 0; j<A._Nj;j++)
				res.ReplaceColumn(j,A.GetColumn(A._Nj-j-1));
		}
		else
		{
			for(unsigned INT i = 0; i<A._Ni;i++)
				res.ReplaceLine(i,A.GetLine(A._Ni-i-1));
		}

		return res;
	}

	//Vertical concatenation of matrix A and B
	FMatrix VCat(const FMatrix& A,const  FMatrix& B)
	{
		if(A.IsEmpty())
		{
			FMatrix res(B);

			return B;
		}

		if(A._Nj != B._Nj)
			THROWERROR(L"VCat: When concatenating vertically, the matrices must have the same number of columns\r\n");

		FMatrix res(A._Ni+B._Ni,A._Nj);
		double *tempA = A._data, *tempB = B._data, *tempRes = res._data;

		for(unsigned INT i =0; i<A._Nj;i++)
		{
			memcpy_s(tempRes,A._Ni*sizeof(double),tempA,A._Ni*sizeof(double));
			tempRes += A._Ni;
			tempA += A._Ni;

			memcpy_s(tempRes,B._Ni*sizeof(double),tempB,B._Ni*sizeof(double));
			tempRes += B._Ni;
			tempB += B._Ni;
		}

		return res;
	}

	//Horizontal concatenation of matrix A and B
	FMatrix HCat(const FMatrix& A,const FMatrix& B)
	{
		if(A.IsEmpty())
		{
			FMatrix res(B);

			return B;
		}

		if(A._Ni != B._Ni)
			THROWERROR(L"When concatenating horizontally, the matrices must have the same number of lines\r\n");

		FMatrix res(A._Ni,A._Nj+B._Nj);
		double* temp = res._data;

		memcpy_s(temp,A._Ni*A._Nj*sizeof(double),A._data,A._Ni*A._Nj*sizeof(double));
		temp += A._Ni*A._Nj;

		memcpy_s(temp,B._Ni*B._Nj*sizeof(double),B._data,B._Ni*B._Nj*sizeof(double));

		return res;
	}

	//Sum of all elements of matrix A
	double Sum(const FMatrix& A)
	{
		double res = 0;
		INT len = A._Ni*A._Nj;
		for(INT i = 0; i<len;i++)
			res += A._data[i];

		return res;
	}

	FMatrix Sum(const FMatrix& A, bool Rows)
	{
		if(Rows)
		{
			FMatrix res(1,A._Nj);

			for(unsigned INT j=0;j<A._Nj;j++)
			{
				res(j) = Sum(A.GetColumn(j));
			}

			return res;
		}
		else
		{
			FMatrix res(A._Ni,1);

			for(unsigned INT i=0;i<A._Ni;i++)
			{
				res(i) = Sum(A.GetLine(i));
			}

			return res;
		}
	}

	double Product(const FMatrix& A)
	{
		double res = 1;
		unsigned INT i = 0;
		unsigned INT n = A.GetLength();

		for(i = 0; i < n;i++)
		{
			res *= A(i);
		}

		return res;
	}

	//Converts a scalar FMatrix to double;
	double ToDouble(const FMatrix& A)
	{
		if(A._Ni !=1 || A._Nj != 1)
			THROWERROR(L"The supplied FMatrix must be a scalar to be converted to double\r\n");

		return A(0);
	}

	char* ToString(const FMatrix& A)
	{
		char* res = 0;

		if(A._Ni*A._Nj != 0)
		{
			INT nchars = 4;
			res = (char*)malloc(nchars*sizeof(char));

			char (*temp)[20] = new char[A._Ni*A._Nj][20];

			INT iter = 0;
			for(unsigned INT i=0;i<A._Ni;i++)
			{
				for(unsigned INT j=0;j<A._Nj;j++)
				{
					sprintf_s(temp[iter],20,"%s%6e\t",(A(i,j)<0)?"-":" ",fabs(A(i,j)));
					iter++;
				}
			}

		}

		return res;
	}

	//Returns a read-only poINTer to the data stored in the matrix.
	const double* ToDoublePtr(const FMatrix& A)
	{
		return A._data;
	}

	double Mean(const FMatrix& A)
	{
		double res = Sum(A);

		return res/A.GetLength();
	}

	FMatrix Mean(const FMatrix& A,bool Rows)
	{
		if(Rows)
		{
			FMatrix res(1,A._Nj);

			for(unsigned INT j = 0; j< A._Nj; j++)
			{
				res(j) = Sum(A.GetColumn(j));
			}

			return res/A._Ni;
		}
		else
		{
			FMatrix res(A._Ni,1);

			for(unsigned INT i = 0; i< A._Ni; i++)
			{
				res(i) = Sum(A.GetLine(i));
			}

			return res/A._Nj;
		}
	}

	FMatrix pow(FMatrix _X, INT _Y, bool elementwise)
	{
		if(elementwise)
		{
			if(_Y == 0)
			{
				return FMatrix(_X._Ni,_X._Nj,1,false);
			}

			FMatrix res(_X._Ni,_X._Nj);
			INT n = res.GetLength();
			double b = (double)_Y;

			vdPowx(n,_X._data,b,res._data);

			return res;
		}
		else
		{
			return _X^_Y;
		}
	}

	FMatrix FromDouble(double val)
	{
		FMatrix res(1,1,val);

		return res;
	}

	FMatrix Pack(const FMatrix& A, INT increment)
	{
		if(A._Ni != 1 && A._Nj != 1)
		{
			THROWERROR(L"Pack: this function \"Pack(const FMatrix& A,INT increment)\" is only valid for vectors\r\n"); 
		}

		FMatrix res(A);

		res.Pack(increment);

		return res;
	}

	FMatrix Diff(const FMatrix& A)
	{
		FMatrix res(A._Ni-1,A._Nj);

		for(unsigned INT i=0;i<A._Ni-1;i++)
		{
			res.ReplaceLine(i,A.GetLine(i+1) - A.GetLine(i));
		}

		return res;
	}

	FMatrix RepMat(const FMatrix& A,unsigned INT n, unsigned INT m)
	{
		FMatrix line,res;

		INT k = 0;

		if(m < 100)
		{
			for(unsigned INT j=0; j <m;j++)
			{
				line = HCat(line,A);
			}
		}
		else
		{
			line.Initialize(A.GetNRows(),m*A.GetNColumns());
			#pragma omp parallel private(k) shared(A,m,line)
			{
				#pragma omp for
				for(k=0;k<(INT)m;k++)
				{
					line.ReplaceColumn(k*A.GetNColumns(),(k+1)*A.GetNColumns()-1,A);
				}
			}
		}

		if(n < 100)
		{
			for(unsigned INT i=0; i <n;i++)
			{
				res = VCat(res,line);
			}
		}
		else
		{
			res.Initialize(A.GetNRows()*n,line.GetNColumns());
			#pragma omp parallel private(k) shared(A,n,line,res)
			{
				#pragma omp for
				for(k=0;k<(INT)n;k++)
				{
					res.ReplaceLine(k*line.GetNRows(),(k+1)*line.GetNRows()-1,line);
				}
			}
		}

		return res;
	}

	FMatrix VectorToDiagonalMatrix(const FMatrix& A)
	{
		if(A._Ni != 1 && A._Nj != 1)
			THROWERROR(L"VectorToDiagonalMatrix: Input must be a vector\r\n");

		FMatrix res(A.GetLength(),A.GetLength());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i,i) = A(i);

		res._type = Diagonal;

		return res;
	}

	FMatrix floor(const FMatrix& A)
	{
		FMatrix res(A._Ni,A._Nj);

		vdFloor( A._Ni*A._Nj, A._data, res._data );

		return res;
	}

	FMatrix ceil(const FMatrix& A)
	{
		FMatrix res(A._Ni,A._Nj);

		vdCeil( A._Ni*A._Nj, A._data, res._data );

		return res;
	}

	FMatrix round(const FMatrix& A)
	{
		FMatrix res(A._Ni,A._Nj);

		vdRound(A._Ni*A._Nj, A._data, res._data );

		return res;
	}

	FMatrix trunc(const FMatrix& A)
	{
		FMatrix res(A._Ni,A._Nj);

		vdTrunc(A._Ni*A._Nj, A._data, res._data );

		return res;
	}

	FMatrix sign(const FMatrix& _X)
	{
		FMatrix res(_X._Ni,_X._Nj);

		for(size_t i =0;i<(size_t)res._Ni*(size_t)res._Nj; i++)
		{
			res._data[i] = (double)sign(_X._data[i]);
		}

		return res;
	}

	CMATRIXLIB_API FMatrix Eye(INT N)
	{
		FMatrix eye(N,N,1);

		return eye;
	}

	CMATRIXLIB_API FMatrix Zero(INT N)
	{
		FMatrix zero(N,N);

		return zero;
	}

	CMATRIXLIB_API FMatrix Matrix(INT Ni, INT Nj, ...)
	{
		va_list argptr;
		va_start(argptr, Nj);

		FMatrix res(Ni,Nj);

		for(INT i = 0; i<Ni*Nj;i++)
		{
			res(i) = va_arg(argptr,double);
		}
		va_end(argptr);

		return res;
	}

	CMATRIXLIB_API FMatrix Zero(INT Ni, INT Nj)
	{
		FMatrix zero(Ni,Nj);

		return zero;
	}

	FMatrix Ones(INT Ni, INT Nj)
	{
		FMatrix ones(Ni,Nj,1,false);

		return ones;
	}

	FMatrix HomogeneousTranslation(double X, double Y, double Z)
	{
		FMatrix res(4,4,1);

		res(3,0) = X;
		res(3,1) = Y;
		res(3,2) = Z;

		return res;
	}

	FMatrix HomogeneousRotationX(double angle)
	{
		FMatrix res(4,4,1);

		res(1,1) = std::cos(angle);
		res(2,2) = res(1,1);

		res(1,2) = -std::sin(angle);
		res(2,1) = -res(1,2);

		return res;
	}

	FMatrix HomogeneousRotationY(double angle)
	{
		FMatrix res(4,4,1);

		res(0,0) = std::cos(angle);
		res(2,2) = res(1,1);

		res(0,2) = -std::sin(angle);
		res(2,0) = -res(1,2);

		return res;
	}

	FMatrix HomogeneousRotationZ(double angle)
	{
		FMatrix res(4,4,1);

		res(0,0) = std::cos(angle);
		res(1,1) = res(1,1);

		res(0,1) = -std::sin(angle);
		res(1,0) = -res(1,2);

		return res;
	}

	FMatrix HomogeneousScale(double sX, double sY, double sZ)
	{
		FMatrix res(4,4);

		res(0,0) = sX;
		res(1,1) = sY;
		res(2,2) = sZ;
		res(3,3) = 1;

		return res;
	}

	FMatrix FromCompressedString(const string& Data)
	{
		const char* data = Data.c_str(), *head = 0, *tail = 0;
		char buff[100];
		memset(buff, 0, sizeof(buff));

		char div = 3;
		char dec = 1;
		char sgn = 4;
		char exp = 2;

		head = data;
		tail = strchr(head, div);
		strncpy_s(buff, sizeof(buff), head, (tail - head) / sizeof(char));
		head = tail + 1;

		unsigned INT Ni = (unsigned INT)FromCompressedString(buff, (INT)strlen(buff));

		tail = strchr(head, div);
		strncpy_s(buff, sizeof(buff), head, (tail - head) / sizeof(char));
		head = tail + 1;
		unsigned INT Nj = (unsigned INT)FromCompressedString(buff, (INT)strlen(buff));

		double *vals = (double*)malloc((size_t)Ni*(size_t)Nj * sizeof(double));

		for (size_t i = 0; i<(size_t)Ni*(size_t)Nj; i++)
		{
			tail = strchr(head, dec);
			strncpy_s(buff, sizeof(buff), head, (tail - head) / sizeof(char));
			head = tail + 1;

			INT INT_sign = 0;

			char* ptr = buff;

			if (buff[0] == sgn)
			{
				ptr++;
				INT_sign = -1;
			}
			else
			{
				INT_sign = 1;
			}

			INT INT_part = (INT)FromCompressedString(ptr, (INT)strlen(ptr));

			tail = strchr(head, exp);
			strncpy_s(buff, sizeof(buff), head, (tail - head) / sizeof(char));
			head = tail + 1;
			unsigned long long dec_part = FromCompressedString(buff, (INT)strlen(buff));

			double dval = INT_sign * (INT_part + dec_part * std::pow(10.0, -16));

			tail = strchr(head, div);
			strncpy_s(buff, sizeof(buff), head, (tail - head) / sizeof(char));
			head = tail + 1;
			INT exp_sign = 0;

			ptr = buff;

			if (buff[0] == sgn)
			{
				ptr++;
				exp_sign = -1;
			}
			else
			{
				exp_sign = 1;
			}

			INT exp_part = (INT)FromCompressedString(ptr, (INT)strlen(ptr));

			vals[i] = dval * std::pow(10.0, exp_sign*exp_part);

		}

		FMatrix res(Ni, Nj, vals);

		free(vals);
		data = 0;

		return res;
	}

#pragma region BMatrix

	BMatrix::BMatrix()
	{
		_Ni = 0;
		_Nj = 0;

		_data = 0;
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;
	}

	BMatrix::BMatrix(INT Ni, INT Nj)
	{
		_Ni = 0;
		_Nj = 0;

		_data = 0;
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		PrivateInitialization(Ni,Nj);
	}

	BMatrix::BMatrix(INT Ni, INT Nj, const bool *data)
	{
		_Ni = 0;
		_Nj = 0;

		_data = 0;
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		PrivateInitialization(Ni,Nj,data);
	}

	BMatrix::BMatrix(INT Ni, INT Nj, bool val, bool diag_only)
	{
		_Ni = 0;
		_Nj = 0;

		_data = 0;
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		PrivateInitialization(Ni,Nj,val,diag_only);
	}

	BMatrix::BMatrix(const BMatrix &Mat)
	{
		_Ni = 0;
		_Nj = 0;

		_data = 0;
		_dims[0] = &_Ni;
		_dims[1] = &_Nj;

		PrivateCopy(Mat);
	}

	BMatrix::BMatrix(const FMatrix &Mat)
	{
		_Ni = 0;
		_Nj = 0;

		_data = 0;

		_dims[0] = &_Ni;
		_dims[1] = &_Nj;
		PrivateInitialization(Mat);
	}

	BMatrix::~BMatrix()
	{
		Clear();
	}

	void BMatrix::Initialize(INT Ni, INT Nj)
	{
		PrivateInitialization(Ni,Nj);
	}

	void BMatrix::Initialize(INT Ni, INT Nj, const bool *data)
	{
		PrivateInitialization(Ni,Nj,data);
	}

	void BMatrix::Initialize(INT Ni, INT Nj, bool val, bool diag_only)
	{
		PrivateInitialization(Ni,Nj,val,diag_only);
	}

	void BMatrix::Initialize(const BMatrix &Mat)
	{
		PrivateCopy(Mat);
	}

	void BMatrix::Initialize(const FMatrix &Mat)
	{
		PrivateInitialization(Mat);
	}

	void BMatrix::Clear()
	{
		FREE(_data);

		_Ni = 0;
		_Nj = 0;
	}

	void BMatrix::ReplaceColumn(INT j,const BMatrix& NewCol)
	{
		if(NewCol._Nj > 1)
			THROWERROR(L"NewCol must be a column vector\r\n");

		if(NewCol._Ni != _Ni)
			THROWERROR(L"NewCol must have the same number of lines as the matrix\r\n");

		bool *temp = &_data[_Ni*j];

		memcpy_s(temp,sizeof(bool)*_Ni,NewCol._data,sizeof(bool)*_Ni);
	}

	void BMatrix::ReplaceColumn(INT j0, INT j1, const BMatrix& NewCol)
	{
		INT NJ = 0;

		if(j0 == j1)
			NJ = 1;
		else
			NJ = j1-j0+1;

		if(NewCol._Nj != NJ)
			THROWERROR(L"The rhs must have the same number of columns as specified\r\n");

		if(NewCol._Ni != _Ni)
			THROWERROR(L"NewCol must have the same number of lines as the matrix\r\n");

		bool *temp = &_data[_Ni*j0];

		memcpy_s(temp,sizeof(bool)*(_Ni*NJ),NewCol._data,sizeof(bool)*(_Ni*NJ));
	}

	void BMatrix::ReplaceLine(INT i,const BMatrix& NewLine)
	{
		if(NewLine._Ni > 1)
			THROWERROR(L"NewLine must be a line vector\r\n");

		if(NewLine._Nj != _Nj)
			THROWERROR(L"NewLine must have the same number of columns as the matrix\r\n");

		for(INT j = 0; j < _Nj; j++)
		{
			(*this)(i,j) = NewLine(j);
		}
	}

	void BMatrix::ReplaceLine(INT i0, INT i1, const BMatrix& NewLine)
	{
		INT NI = 0;

		if(i0 == i1)
			NI = 1;
		else
			NI = i1-i0+1;

		if(NewLine._Ni != NI)
			THROWERROR(L"The rhs must have the same number of columns as specified\r\n");

		if(NewLine._Nj != _Nj)
			THROWERROR(L"NewLine must have the same number of columns as the matrix\r\n");

		for(INT i = 0; i < NI; i++)
		{
			for(INT j = 0; j < _Nj; j++)
			{
				(*this)(i+i0,j) = NewLine(i,j);
			}
		}
	}


	BMatrix BMatrix::GetColumn(INT j) const
	{
		BMatrix res(_Ni,1);

		memcpy_s(res._data,sizeof(bool)*_Ni,&_data[_Ni*j],sizeof(bool)*_Ni);

		return res;
	}

	BMatrix BMatrix::GetColumn(INT j0, INT j1) const
	{
		if(j0 > j1)
			THROWERROR(L"FMatrix::GetColumn - The index of the last column must be larger than the firt one\r\n");

		if(j0 < 0)
			THROWERROR(L"FMatrix::GetColumn - The lower index is outside the matrix bounds\r\n");

		if(j1 >= _Nj)
			THROWERROR(L"FMatrix::GetColumn - The upper index is outside the matrix bounds\r\n");

		INT NJ = 0;

		if(j0 == j1)
			NJ = 1;
		else
			NJ = j1-j0+1;

		BMatrix res(_Ni,NJ);

		memcpy_s(res._data,sizeof(bool)*(_Ni*NJ),&_data[_Ni*j0],sizeof(bool)*(_Ni*NJ));

		return res;
	}

	BMatrix BMatrix::GetLine(INT i) const
	{
		BMatrix res(1,_Nj);

		for(INT j = 0; j < _Nj; j++)
			res(0,j) = _data[i+_Ni*j];

		return res;
	}

	BMatrix BMatrix::GetLine(INT i0, INT i1) const
	{
		if(i0 < 0)
			THROWERROR(L"GetLine: i0 must be greater or equal to 0\r\n")
		else if(i1 > _Ni-1)
			THROWERROR(L"GetLine: i1 is outside the matrix limits\r\n")
		else if(i0 > i1)
			THROWERROR(L"GetLine: i1 must be greater or equal than i0\r\n");

		INT NI = 0;

		if(i0 == i1)
			NI = 1;
		else
			NI = i1-i0+1;

		BMatrix res(NI,_Nj);


		for(INT i = 0; i < NI; i++)
		{
			for(INT j = 0; j<_Nj; j++)
			{
				res(i,j) = _data[i0+i+_Ni*j];
			}
		}

		return res;
	}

	void BMatrix::Reshape(INT NewNi, INT NewNj)
	{
		if(NewNi*NewNj != _Ni*_Nj)
			THROWERROR(L"Reshape: Matrix must have the same number of elements\r\n");

		_Ni = NewNi;
		_Nj = NewNj;
	}


	BMatrix BMatrix::GetRange(INT beg_line, INT end_line, INT beg_col, INT end_col) const
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

		BMatrix temp(_Ni,NJ), res(NI,NJ);

	
		for(INT j=0; j < NJ; j++)
		{
			temp.ReplaceColumn(j,this->GetColumn(j+beg_col));
		}

		for(INT i =0; i < NI; i++)
		{
			res.ReplaceLine(i,temp.GetLine(i+beg_line));
		}

		return res;
	}

	void BMatrix::SetRange(INT beg_line, INT end_line, INT beg_col, INT end_col,const BMatrix& newMat)
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

		bool *temp = &_data[beg_line+_Ni*beg_col];
		for(INT j=0;j<NJ;j++)
		{
			memcpy_s(temp,NI*sizeof(bool),newMat.GetColumn(j)._data,NI*sizeof(bool));
			temp += _Ni;
		}
	}

	bool BMatrix::IsEmpty() const
	{
		if(_Ni*_Nj == 0)
			return true;
		else
			return false;
	}

	bool& BMatrix::operator ()(INT i, INT j)
	{
		if(i >= _Ni || j >= _Nj)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _data[i+_Ni*j];
	}

	bool BMatrix::operator ()(INT i, INT j) const
	{
		if(i >= _Ni || j >= _Nj)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _data[i+_Ni*j];
	}

	bool& BMatrix::operator ()(INT i)
	{
		if(i >= _Ni*_Nj)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _data[i];
	}

	bool BMatrix::operator ()(INT i) const
	{
		if(i >= _Ni*_Nj)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _data[i];
	}

	BMatrix BMatrix::operator &&(const BMatrix &B)
	{
		if(B.IsEmpty())
			return *this;
		else if(IsEmpty())
			return B;

		if(_Ni != B._Ni || _Nj != B._Nj)
			THROWERROR(L"operator&&: Matrices must have the same dimensions\r\n");

		BMatrix res(_Ni,_Nj);

		for(INT i=0;i<_Ni*_Nj;i++)
			res(i) = _data[i]&&B(i);

		return res;
	}

	BMatrix BMatrix::operator ||(const BMatrix &B)
	{
		if(B.IsEmpty())
			return *this;
		else if(IsEmpty())
			return B;

		if(_Ni != B._Ni || _Nj != B._Nj)
			THROWERROR(L"operator||: Matrices must have the same dimensions\r\n");

		BMatrix res(_Ni,_Nj);

		for(INT i=0;i<_Ni*_Nj;i++)
			res(i) = _data[i]||B(i);

		return res;
	}

	BMatrix BMatrix::operator !() const
	{
		BMatrix res(_Ni,_Nj);

		for(INT i=0;i<_Ni*_Nj;i++)
			res(i) = !_data[i];

		return res;
	}

	BMatrix& BMatrix::operator =(const bool *data)
	{
		PrivateInitialization(_Ni,_Nj,data);
		return *this;
	}

	BMatrix& BMatrix::operator= (const CMatrix<bool>& Mat)
	{
		PrivateInitialization(Mat.GetNRows(),Mat.GetNColumns(),Mat.GetDataPtr());
		return *this;
	}

	BMatrix& BMatrix::operator =(const BMatrix &Mat)
	{
		PrivateCopy(Mat);
		return *this;
	}

	BMatrix& BMatrix::operator =(const FMatrix &Mat)
	{
		PrivateInitialization(Mat);
		return *this;
	}

	BMatrix::operator FMatrix() const
	{
		FMatrix res(_Ni,_Nj);
		for(INT i=0;i<_Ni*_Nj;i++)
			res(i)=(_data[i])?1.0:0.0;

		return res;
	}

	BMatrix::operator CMatrix<bool>() const
	{
		CMatrix<bool> res(_Ni,_Nj);
		for(INT i=0;i<_Ni*_Nj;i++)
			res(i)=_data[i];

		return res;
	}

	BMatrix::operator bool() const
	{
		return All(*this);
	}

	void BMatrix::PrivateInitialization(INT Ni, INT Nj)
	{
		if(Ni == 0 || Nj == 0)
		{
			_Ni = 0;
			_Nj = 0;
			_data = 0;
			return;
		}
		_data = (bool*)realloc(_data,Ni*Nj*sizeof(bool));
		if(errno == ENOMEM || _data == 0)
			THROWERROR(L"BMarix: Not enough memory\r\n");
		memset(_data,0,Ni*Nj*sizeof(bool));

		_Ni = Ni;
		_Nj = Nj;
	}

	void BMatrix::PrivateInitialization(INT Ni, INT Nj, const bool* data)
	{
		if(Ni == 0 || Nj == 0)
		{
			_Ni = 0;
			_Nj = 0;
			_data = 0;
			return;
		}

		PrivateInitialization(Ni,Nj);
		memcpy_s(_data,_Ni*_Nj*sizeof(bool),data,Ni*Nj*sizeof(bool));
	}

	void BMatrix::PrivateInitialization(INT Ni, INT Nj, bool val, bool diag_only)
	{
		if(Ni == 0 || Nj == 0)
		{
			_Ni = 0;
			_Nj = 0;
			_data = 0;
			return;
		}

		PrivateInitialization(Ni,Nj);

		if(diag_only)
		{
			memset(_data,0,Ni*Nj*sizeof(bool));
			for(INT i=0;i<Ni;i++)
				_data[i*_Ni+i] = val;
		}
		else
			memset(_data,val,Ni*Nj*sizeof(bool));

	}

	void BMatrix::PrivateInitialization(const FMatrix& Mat)
	{
		if(Mat.GetNRows() == 0 || Mat.GetNColumns() == 0)
		{
			_Ni = 0;
			_Nj = 0;
			_data = 0;
			return;
		}

		PrivateInitialization(Mat.GetNRows(),Mat.GetNColumns());
		for(INT i=0;i<_Ni*_Nj;i++)
			_data[i] = (Mat(i)<1)?false:true;
	}
		
	void BMatrix::PrivateCopy(const BMatrix& Mat)
	{
		if(Mat.IsEmpty())
		{
			Clear();
			return;
		}
		PrivateInitialization(Mat._Ni,Mat._Nj,Mat._data);
	}

	BMatrix operator<(const FMatrix& A, double B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)<B;

		return res;
	}

	BMatrix operator<(const FMatrix& A, float B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = (float)A(i)<B;

		return res;
	}

	BMatrix operator<(const FMatrix& A, const FMatrix& B)
	{
		if(A.GetNRows() != B.GetNRows() || A.GetNColumns() != B.GetNColumns())
			THROWERROR(L"operator<: The input matrices must have the same dimensions\r\n");
	
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)<B(i);

		return res;
	}

	BMatrix operator<=(const FMatrix& A, double B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)<=B;

		return res;
	}

	BMatrix operator<=(const FMatrix& A, float B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = (float)A(i)<=B;

		return res;
	}

	BMatrix operator<=(const FMatrix& A, const FMatrix& B)
	{
		if(A.GetNRows() != B.GetNRows() || A.GetNColumns() != B.GetNColumns())
			THROWERROR(L"operator<=: The input matrices must have the same dimensions\r\n");
	
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)<=B(i);

		return res;
	}

	BMatrix operator>(const FMatrix& A, double B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)>B;

		return res;
	}

	BMatrix operator>(const FMatrix& A, float B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = (float)A(i)>B;

		return res;
	}

	BMatrix operator>(const FMatrix& A, const FMatrix& B)
	{
		if(A.GetNRows() != B.GetNRows() || A.GetNColumns() != B.GetNColumns())
			THROWERROR(L"operator>: The input matrices must have the same dimensions\r\n");
	
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)>B(i);

		return res;
	}

	BMatrix operator>=(const FMatrix& A, double B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)>=B;

		return res;
	}

	BMatrix operator>=(const FMatrix& A, float B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = ((float)A(i)>=B);

		return res;
	}

	BMatrix operator>=(const FMatrix& A, const FMatrix& B)
	{
		if(A.GetNRows() != B.GetNRows() || A.GetNColumns() != B.GetNColumns())
			THROWERROR(L"operator>=: The input matrices must have the same dimensions\r\n");
	
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)>=B(i);

		return res;
	}

	BMatrix operator==(const FMatrix& A, double B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)==B;

		return res;
	}

	BMatrix operator==(const FMatrix& A, float B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = (float)A(i)==B;

		return res;
	}

	BMatrix operator==(const FMatrix& A, const FMatrix& B)
	{
		if(A.GetNRows() != B.GetNRows() || A.GetNColumns() != B.GetNColumns())
			THROWERROR(L"operator==: The input matrices must have the same dimensions\r\n");
	
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)==B(i);

		return res;
	}

	BMatrix operator!=(const FMatrix& A, double B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)!=B;

		return res;
	}

	BMatrix operator!=(const FMatrix& A, float B)
	{
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = (float)A(i)!=B;

		return res;
	}

	BMatrix operator!=(const FMatrix& A, const FMatrix& B)
	{
		if(A.GetNRows() != B.GetNRows() || A.GetNColumns() != B.GetNColumns())
			THROWERROR(L"operator!=: The input matrices must have the same dimensions\r\n");
	
		BMatrix res(A.GetNRows(),A.GetNColumns());

		for(unsigned INT i=0;i<A.GetLength();i++)
			res(i) = A(i)!=B(i);

		return res;
	}

	bool Any(const BMatrix& A)
	{
		if(Sum(A) > 0)
			return 1;
		else
			return 0;
	}

	bool All(const BMatrix& A)
	{
		if(Sum(A) == A.GetLength())
			return 1;
		else
			return 0;
	}

	CMatrix<unsigned INT> Find(const BMatrix& A)
	{
		INT n = (INT)Sum(A);

		CMatrix<unsigned INT> res(n,1);

		INT j =0;
		for(unsigned INT i =0;i < (unsigned INT)A.GetLength();i++)
		{
			if(A(i) == 1)
			{
				res(j) = i;
				j++;
			}
		}

		return res;
	}

	wofstream& operator<<(wofstream& out,BMatrix &Mat)
	{
		out<<"Ni= "<<Mat._Ni<<" Nj= "<<Mat._Nj<<" ";

		for(INT i=0;i<Mat._Ni*Mat._Nj;i++)
		{
			out<<Mat(i)<<" ";
		}

		return out;
	}

	wifstream& operator>>(wifstream& in,BMatrix &Mat)
	{
		wchar_t temp[20];

		INT ni = 0,no = 0;

		in>>temp>>ni>>temp>>no;

		double *data = (double*)malloc(ni*no*sizeof(double));

		for(INT i = 0; i < ni*no; i++)
		{
			in>>data[i];
		}

		Mat = FMatrix(ni,no,data);

		free(data);
		data = 0;

		return in;
	}

	CMATRIXLIB_API const BMatrix EmptyBMatrix;

#pragma endregion //BMatrix

#pragma region RMatrix
	RMatrix::RMatrix()
	{
		_NRows = 0;
		_NColumns = 0;

		_orig = 0;

		_startR = 0;
		_startC = 0;

		_length = 0;
	}

	RMatrix::RMatrix(FMatrix& Mat)
	{
		PrivateCopy(Mat);
	}

	RMatrix::RMatrix(const RMatrix& RMat)
	{
		PrivateCopy(RMat);
	}

	RMatrix::RMatrix(FMatrix* Orig, unsigned INT NRows, unsigned INT NColumns, unsigned INT StartR, unsigned INT StartC)
	{
		_NRows = NRows;
		_NColumns = NColumns;
		_length = _NRows * _NColumns;

		_orig = Orig;

		_startR = StartR;
		_startC = StartC;
	}

	RMatrix::~RMatrix()
	{
		Clear();
	}

	void RMatrix::Clear()
	{
		_NRows = 0;
		_NColumns = 0;

		_orig = 0;

		_startR = 0;
		_startC = 0;

		_length = 0;
	}

	void RMatrix::CopyInto(const FMatrix& Data)
	{
		if (Data.GetNRows() != _NRows || Data.GetNColumns() != _NColumns)
			THROWERROR(L"RMatrix::CopyInto: Data to be copied must have the same dimensions of the reference.");

		double* rdata = _orig->_data;
		double* mdata = Data._data;

		INT length = _length;

#pragma omp parallel for shared(mdata,rdata,length)
		for (int I = 0; I < length; I++)
		{
			unsigned INT i = I % _NRows;
			unsigned INT j = (I - i) / _NRows;

			rdata[(i + _startR) + (j + _startC)*_orig->_Ni] = mdata[I];
		}
	}

	void RMatrix::PrivateCopy(FMatrix& Mat)
	{
		_NRows = Mat._Ni;
		_NColumns = Mat._Nj;
		_length = _NRows * _NColumns;

		_orig = &Mat;

		_startR = 0;
		_startC = _NRows - 1;
	}

	void RMatrix::PrivateCopy(const RMatrix& RMat)
	{
		_NRows = RMat._NRows;
		_NColumns = RMat._NColumns;
		_length = RMat._length;

		_orig = RMat._orig;

		_startR = RMat._startR;
		_startC = RMat._startC;
	}

	double& RMatrix::operator() (unsigned INT i, unsigned INT j)
	{
		if (i >= _NRows || j >= _NColumns)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _orig->_data[(i + _startR) + (j + _startC)*_orig->_Ni];
	}

	double RMatrix::operator() (unsigned INT i, unsigned INT j) const
	{
		if (i >= _NRows || j >= _NColumns)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		return _orig->_data[(i + _startR) + (j + _startC)*_orig->_Ni];
	}

	double& RMatrix::operator() (unsigned INT I)
	{
		if (I >= (size_t)_NRows*(size_t)_NColumns)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		unsigned INT i = I % _NRows;
		unsigned INT j = (I - i) / _NRows;

		return _orig->_data[(i + _startR) + (j + _startC)*_orig->_Ni];
	}

	double RMatrix::operator() (unsigned INT I) const
	{
		if (I >= (size_t)_NRows*(size_t)_NColumns)
			THROWERROR(L"operator(): index outside of matrix dimensions\r\n");

		unsigned INT i = I % _NRows;
		unsigned INT j = (I - i) / _NRows;

		return _orig->_data[(i + _startR) + (j + _startC)*_orig->_Ni];
	}


	RMatrix & RMatrix::operator=(FMatrix& Mat)
	{
		Clear();
		PrivateCopy(Mat);

		return *this;
	}

	RMatrix& RMatrix::operator=(const RMatrix& RMat)
	{
		Clear();
		PrivateCopy(RMat);


		return *this;
	}

	RMatrix::operator FMatrix() const
	{
		FMatrix res(_NRows, _NColumns);

		if (_orig == 0)
			return res;

		double* rdata = _orig->_data;
		double* mdata = res._data;

		INT length = _length;

		#pragma omp parallel for shared(mdata,rdata,length)
		for (int I = 0; I < length; I++)
		{
			unsigned INT i = I % _NRows;
			unsigned INT j = (I - i) / _NRows;

			mdata[I] = rdata[(i + _startR) + (j + _startC)*_orig->_Ni];
		}
		
		return res;

	}
#pragma endregion //RMatrix

	#if defined _DEBUG && !defined __GNUC__
	CMATRIXLIB_API void DumpMemoryLeaks()																
	{
		_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
		_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_DEBUG );
	}
	#endif

}
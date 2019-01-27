#pragma once
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

using std::endl;
using std::cout;
using std::istream;

class Matrix
{
private:
	float**p;
	__host__ __device__ inline void initialize();//初始化矩阵

public:
	int rows_num, cols_num;
	//声明一个全0矩阵
	__host__ __device__ Matrix(int rows, int cols)
	{
		rows_num = rows;
		cols_num = cols;
		initialize();
		for (int i = 0; i < rows_num; i++)
		{
			for (int j = 0; j < cols_num; j++)
			{
				p[i][j] = 0;
			}
		}
	}
	//声明一个值全部为value的矩阵
	__host__ __device__ Matrix(int rows, int cols, float value)
	{
		rows_num = rows;
		cols_num = cols;
		initialize();
		for (int i = 0; i < rows_num; i++)
		{
			for (int j = 0; j < cols_num; j++)
			{
				p[i][j] = value;
			}
		}
	}
	__host__ __device__ ~Matrix()
	{
		delete[] p;
	}
	__host__ __device__ inline Matrix& operator=(const Matrix&);//矩阵的复制
	__host__ __device__ inline Matrix& operator=(float*);//将数组的值传给矩阵
	__host__ __device__ inline Matrix& operator+=(const Matrix&);//矩阵的+=操作
	__host__ __device__ inline Matrix& operator-=(const Matrix&);//-=
	__host__ __device__ inline Matrix& operator*=(const Matrix&);//*=
	__host__ __device__ inline Matrix operator*(const Matrix& m)const;
	inline void Show() const;//矩阵显示
	__host__ __device__ inline void swapRows(int, int);
	__host__ __device__ inline float point(int i, int j) const;
	__host__ __device__ inline static Matrix eye(int);//制造一个单位矩阵
	__host__ __device__ inline static Matrix T(const Matrix& m);//矩阵转置的实现,且不改变矩阵
};

class Matrix3x3
{
private:
	float(*p)[3][3];
	__host__ __device__ inline void initialize();//初始化矩阵

public:
	//声明一个全0矩阵
	__host__ __device__  Matrix3x3()
	{
		initialize();
	}
	//声明一个值全部为value的矩阵
	__host__ __device__  Matrix3x3(float value)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				*p[i][j] = value;
			}
		}
	}
	__host__ __device__ ~Matrix3x3()
	{
		delete[] p;
	}
	__host__ __device__ inline Matrix3x3& operator=(const Matrix3x3&);//矩阵的复制
	__host__ __device__ inline Matrix3x3& operator=(float*);//将数组的值传给矩阵
	__host__ __device__ inline Matrix3x3& operator+=(const Matrix3x3&);//矩阵的+=操作
	__host__ __device__ inline Matrix3x3& operator-=(const Matrix3x3&);//-=
	__host__ __device__ inline Matrix3x3& operator*=(const Matrix3x3&);//*=
	__host__ __device__ inline Matrix3x3 operator*(const Matrix3x3& m)const;
	inline void Show() const;//矩阵显示
	__host__ __device__ inline float point(int i, int j) const;
	__host__ __device__ inline static Matrix3x3 eye();//制造一个单位矩阵
	__host__ __device__ inline static Matrix3x3 T(const Matrix3x3& m);//矩阵转置的实现,且不改变矩阵
};

class Matrix4x4
{
public:	
#pragma region fields
	// Value at row 1, column 1 of the matrix.
	float M11;
	// Value at row 1, column 2 of the matrix.
	float M12;
	// Value at row 1, column 3 of the matrix.
	float M13;
	// Value at row 1, column 4 of the matrix.
	float M14;
	// Value at row 2, column 1 of the matrix.
	float M21;
	// Value at row 2, column 2 of the matrix.
	float M22;
	// Value at row 2, column 3 of the matrix.
	float M23;
	// Value at row 2, column 4 of the matrix.
	float M24;
	// Value at row 3, column 1 of the matrix.
	float M31;
	// Value at row 3, column 2 of the matrix.
	float M32;
	// Value at row 3, column 3 of the matrix.
	float M33;
	// Value at row 3, column 4 of the matrix.
	float M34;
	// Value at row 4, column 1 of the matrix.
	float M41;
	// Value at row 4, column 2 of the matrix.
	float M42;
	// Value at row 4, column 3 of the matrix.
	float M43;
	// Value at row 4, column 4 of the matrix.
	float M44;
#pragma endregion
	__host__ __device__ inline Matrix4x4(float m11, float m12, float m13, float m14,
		float m21, float m22, float m23, float m24,
		float m31, float m32, float m33, float m34,
		float m41, float m42, float m43, float m44);
	//声明一个全0矩阵
	__host__ __device__ inline Matrix4x4()
	{
	}
	//声明一个值全部为value的矩阵
	__host__ __device__ inline Matrix4x4(float value);
	//__host__ __device__ inline Matrix4x4& operator+=(const Matrix4x4&);//矩阵的+=操作
	//__host__ __device__ inline Matrix4x4& operator-=(const Matrix4x4&);//-=
	//__host__ __device__ inline Matrix4x4& operator*=(const Matrix4x4&);//*=
	__host__ __device__ inline static Matrix4x4 invert(const Matrix4x4 matrix);
	__host__ __device__ inline static Matrix4x4 eye();//制造一个单位矩阵
	__host__ __device__ inline static Matrix4x4 transpose(const Matrix4x4& m);//矩阵转置的实现,且不改变矩阵
};

#pragma region Matrix

inline void Matrix::initialize()
{
	//初始化矩阵大小
	p = new float*[rows_num];//分配rows_num个指针
	for (int i = 0; i < rows_num; ++i)
	{
		p[i] = new float[cols_num];//为p[i]进行动态内存分配，大小为cols
	}
}

//实现矩阵的复制
Matrix& Matrix::operator=(const Matrix& m)
{
	if (this ==&m)
	{
		return*this;
	}

	if (rows_num != m.rows_num || cols_num != m.cols_num)
	{
		for (int i = 0; i < rows_num; ++i)
		{
			delete[] p[i];
		}
		delete[] p;

		rows_num = m.rows_num;
		cols_num = m.cols_num;
		initialize();
	}

	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			p[i][j] = m.p[i][j];
		}
	}
	return*this;
}

//将数组的值传递给矩阵(要求矩阵的大小已经被声明过了)
Matrix& Matrix::operator=(float*a)
{
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			p[i][j] =*(a + i* cols_num + j);
		}
	}
	return*this;
}

//+=操作
Matrix& Matrix::operator+=(const Matrix& m)
{
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			p[i][j] += m.p[i][j];
		}
	}
	return*this;
}

//实现-=
Matrix& Matrix::operator-=(const Matrix& m)
{
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			p[i][j] -= m.p[i][j];
		}
	}
	return*this;
}

//实现*=
Matrix& Matrix::operator*=(const Matrix& m)
{
	Matrix temp(rows_num, m.cols_num);//若C=AB,则矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
	for (int i = 0; i < temp.rows_num; i++)
	{
		for (int j = 0; j < temp.cols_num; j++)
		{
			for (int k = 0; k < cols_num; k++)
			{
				temp.p[i][j] += (p[i][k]* m.p[k][j]);
			}
		}
	}
	*this = temp;
	return*this;
}

//实现矩阵的乘法
Matrix Matrix::operator*(const Matrix& m)const
{
	Matrix ba_M(rows_num, m.cols_num, 0.0);
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < m.cols_num; j++)
		{
			for (int k = 0; k < cols_num; k++)
			{
				ba_M.p[i][j] += (p[i][k]* m.p[k][j]);
			}
		}
	}
	return ba_M;
}

//矩阵显示
void Matrix::Show() const
{
	//cout << rows_num <<" "<<cols_num<< endl;//显示矩阵的行数和列数
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			cout << p[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

//实现行变换
void Matrix::swapRows(int a, int b)
{
	a--;
	b--;
	float*temp = p[a];
	p[a] = p[b];
	p[b] = temp;
}

//返回矩阵第i行第j列的数
float Matrix::point(int i, int j) const
{
	return this->p[i][j];
}

//制造一个单位矩阵
Matrix Matrix::eye(int n)
{
	Matrix A(n, n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == j)
			{
				A.p[i][j] = 1;
			}
			else
			{
				A.p[i][j] = 0;
			}
		}
	}
	return A;
}

//实现矩阵的转置
Matrix Matrix::T(const Matrix& m)
{
	int col_size = m.cols_num;
	int row_size = m.rows_num;
	Matrix mt(col_size, row_size);
	for (int i = 0; i < row_size; i++)
	{
		for (int j = 0; j < col_size; j++)
		{
			mt.p[j][i] = m.p[i][j];
		}
	}
	return mt;
}

#pragma endregion

#pragma region Matrix3x3

void Matrix3x3::initialize()
{
	p = { 0 };
}

//实现矩阵的复制
Matrix3x3& Matrix3x3::operator=(const Matrix3x3& m)
{
	if (this ==&m)
	{
		return*this;
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] =*m.p[i][j];
		}
	}
	return*this;
}

//将数组的值传递给矩阵(要求矩阵的大小已经被声明过了)
Matrix3x3& Matrix3x3::operator=(float*a)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] =*(a + i* 3 + j);
		}
	}
	return*this;
}

//+=操作
Matrix3x3& Matrix3x3::operator+=(const Matrix3x3& m)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] +=*m.p[i][j];
		}
	}
	return*this;
}

//实现-=
Matrix3x3& Matrix3x3::operator-=(const Matrix3x3& m)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] -=*m.p[i][j];
		}
	}
	return*this;
}

//实现*=
Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& m)
{
	Matrix3x3 temp = Matrix3x3();//若C=AB,则矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				*temp.p[i][j] += (*p[i][k]**m.p[k][j]);
			}
		}
	}
	*this = temp;
	return*this;
}

//实现矩阵的乘法
Matrix3x3 Matrix3x3::operator*(const Matrix3x3& m)const
{
	Matrix3x3 ba_M = Matrix3x3();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				*ba_M.p[i][j] += (*p[i][k]**m.p[k][j]);
			}
		}
	}
	return ba_M;
}

//矩阵显示
void Matrix3x3::Show() const
{
	//cout << 3 <<" "<<3<< endl;//显示矩阵的行数和列数
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			cout << p[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

//返回矩阵第i行第j列的数
float Matrix3x3::point(int i, int j) const
{
	return*p[i][j];
}


//制造一个单位矩阵
Matrix3x3 Matrix3x3::eye()
{
	Matrix3x3 A = Matrix3x3();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (i == j)
			{
				*A.p[i][j] = 1;
			}
			else
			{
				*A.p[i][j] = 0;
			}
		}
	}
	return A;
}

//实现矩阵的转置
Matrix3x3 Matrix3x3::T(const Matrix3x3& m)
{
	Matrix3x3 mt = Matrix3x3();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*mt.p[j][i] =*m.p[i][j];
		}
	}
	return mt;
}

#pragma endregion

#pragma region Matrix4x4

Matrix4x4::Matrix4x4(float m11, float m12, float m13, float m14,
	float m21, float m22, float m23, float m24,
	float m31, float m32, float m33, float m34,
	float m41, float m42, float m43, float m44)
{
	M11 = m11;
	M12 = m12;
	M13 = m13;
	M14 = m14;

	M21 = m21;
	M22 = m22;
	M23 = m23;
	M24 = m24;

	M31 = m31;
	M32 = m32;
	M33 = m33;
	M34 = m34;

	M41 = m41;
	M42 = m42;
	M43 = m43;
	M44 = m44;
}

Matrix4x4::Matrix4x4(float value)
{
	M11 = value;
	M12 = value;
	M13 = value;
	M14 = value;

	M21 = value;
	M22 = value;
	M23 = value;
	M24 = value;

	M31 = value;
	M32 = value;
	M33 = value;
	M34 = value;

	M41 = value;
	M42 = value;
	M43 = value;
	M44 = value;
}

//+=操作
//Matrix4x4& Matrix4x4::operator+=(const Matrix4x4& m)
//{
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 4; j++)
//		{
//			*p[i][j] +=*m.p[i][j];
//		}
//	}
//	return*this;
//}

//实现-=
//Matrix4x4& Matrix4x4::operator-=(const Matrix4x4& m)
//{
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 4; j++)
//		{
//			*p[i][j] -=*m.p[i][j];
//		}
//	}
//	return*this;
//}

//实现*=
//Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& m)
//{
//	Matrix4x4 temp = Matrix4x4();//若C=AB,则矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 4; j++)
//		{
//			for (int k = 0; k < 4; k++)
//			{
//				*temp.p[i][j] += (*p[i][k]**m.p[k][j]);
//			}
//		}
//	}
//	*this = temp;
//	return*this;
//}

//实现矩阵的乘法
__host__ __device__ inline Matrix4x4 operator*(const Matrix4x4& value1, const Matrix4x4& value2)
{
	Matrix4x4 m;

	// First row
	m.M11 = value1.M11 * value2.M11 + value1.M12 * value2.M21 + value1.M13 * value2.M31 + value1.M14 * value2.M41;
	m.M12 = value1.M11 * value2.M12 + value1.M12 * value2.M22 + value1.M13 * value2.M32 + value1.M14 * value2.M42;
	m.M13 = value1.M11 * value2.M13 + value1.M12 * value2.M23 + value1.M13 * value2.M33 + value1.M14 * value2.M43;
	m.M14 = value1.M11 * value2.M14 + value1.M12 * value2.M24 + value1.M13 * value2.M34 + value1.M14 * value2.M44;

	// Second row
	m.M21 = value1.M21 * value2.M11 + value1.M22 * value2.M21 + value1.M23 * value2.M31 + value1.M24 * value2.M41;
	m.M22 = value1.M21 * value2.M12 + value1.M22 * value2.M22 + value1.M23 * value2.M32 + value1.M24 * value2.M42;
	m.M23 = value1.M21 * value2.M13 + value1.M22 * value2.M23 + value1.M23 * value2.M33 + value1.M24 * value2.M43;
	m.M24 = value1.M21 * value2.M14 + value1.M22 * value2.M24 + value1.M23 * value2.M34 + value1.M24 * value2.M44;

	// Third row
	m.M31 = value1.M31 * value2.M11 + value1.M32 * value2.M21 + value1.M33 * value2.M31 + value1.M34 * value2.M41;
	m.M32 = value1.M31 * value2.M12 + value1.M32 * value2.M22 + value1.M33 * value2.M32 + value1.M34 * value2.M42;
	m.M33 = value1.M31 * value2.M13 + value1.M32 * value2.M23 + value1.M33 * value2.M33 + value1.M34 * value2.M43;
	m.M34 = value1.M31 * value2.M14 + value1.M32 * value2.M24 + value1.M33 * value2.M34 + value1.M34 * value2.M44;

	// Fourth row
	m.M41 = value1.M41 * value2.M11 + value1.M42 * value2.M21 + value1.M43 * value2.M31 + value1.M44 * value2.M41;
	m.M42 = value1.M41 * value2.M12 + value1.M42 * value2.M22 + value1.M43 * value2.M32 + value1.M44 * value2.M42;
	m.M43 = value1.M41 * value2.M13 + value1.M42 * value2.M23 + value1.M43 * value2.M33 + value1.M44 * value2.M43;
	m.M44 = value1.M41 * value2.M14 + value1.M42 * value2.M24 + value1.M43 * value2.M34 + value1.M44 * value2.M44;

	return m;
}

__host__ __device__ inline Matrix4x4 operator*(const Matrix4x4& value1, float value2)
{
	Matrix4x4 m;

	m.M11 = value1.M11 * value2;
	m.M12 = value1.M12 * value2;
	m.M13 = value1.M13 * value2;
	m.M14 = value1.M14 * value2;
	m.M21 = value1.M21 * value2;
	m.M22 = value1.M22 * value2;
	m.M23 = value1.M23 * value2;
	m.M24 = value1.M24 * value2;
	m.M31 = value1.M31 * value2;
	m.M32 = value1.M32 * value2;
	m.M33 = value1.M33 * value2;
	m.M34 = value1.M34 * value2;
	m.M41 = value1.M41 * value2;
	m.M42 = value1.M42 * value2;
	m.M43 = value1.M43 * value2;
	m.M44 = value1.M44 * value2;
	return m;
}

Matrix4x4 Matrix4x4::invert(Matrix4x4 matrix)
{
	//                                       -1
	// If you have matrix M, inverse Matrix M   can compute
	//
	//     -1       1      
	//    M   = --------- A
	//            det(M)
	//
	// A is adjugate (adjoint) of M, where,
	//
	//      T
	// A = C
	//
	// C is Cofactor matrix of M, where,
	//           i + j
	// C   = (-1)      * det(M  )
	//  ij                    ij
	//
	//     [ a b c d ]
	// M = [ e f g h ]
	//     [ i j k l ]
	//     [ m n o p ]
	//
	// First Row
	//           2 | f g h |
	// C   = (-1)  | j k l | = + ( f ( kp - lo ) - g ( jp - ln ) + h ( jo - kn ) )
	//  11         | n o p |
	//
	//           3 | e g h |
	// C   = (-1)  | i k l | = - ( e ( kp - lo ) - g ( ip - lm ) + h ( io - km ) )
	//  12         | m o p |
	//
	//           4 | e f h |
	// C   = (-1)  | i j l | = + ( e ( jp - ln ) - f ( ip - lm ) + h ( in - jm ) )
	//  13         | m n p |
	//
	//           5 | e f g |
	// C   = (-1)  | i j k | = - ( e ( jo - kn ) - f ( io - km ) + g ( in - jm ) )
	//  14         | m n o |
	//
	// Second Row
	//           3 | b c d |
	// C   = (-1)  | j k l | = - ( b ( kp - lo ) - c ( jp - ln ) + d ( jo - kn ) )
	//  21         | n o p |
	//
	//           4 | a c d |
	// C   = (-1)  | i k l | = + ( a ( kp - lo ) - c ( ip - lm ) + d ( io - km ) )
	//  22         | m o p |
	//
	//           5 | a b d |
	// C   = (-1)  | i j l | = - ( a ( jp - ln ) - b ( ip - lm ) + d ( in - jm ) )
	//  23         | m n p |
	//
	//           6 | a b c |
	// C   = (-1)  | i j k | = + ( a ( jo - kn ) - b ( io - km ) + c ( in - jm ) )
	//  24         | m n o |
	//
	// Third Row
	//           4 | b c d |
	// C   = (-1)  | f g h | = + ( b ( gp - ho ) - c ( fp - hn ) + d ( fo - gn ) )
	//  31         | n o p |
	//
	//           5 | a c d |
	// C   = (-1)  | e g h | = - ( a ( gp - ho ) - c ( ep - hm ) + d ( eo - gm ) )
	//  32         | m o p |
	//
	//           6 | a b d |
	// C   = (-1)  | e f h | = + ( a ( fp - hn ) - b ( ep - hm ) + d ( en - fm ) )
	//  33         | m n p |
	//
	//           7 | a b c |
	// C   = (-1)  | e f g | = - ( a ( fo - gn ) - b ( eo - gm ) + c ( en - fm ) )
	//  34         | m n o |
	//
	// Fourth Row
	//           5 | b c d |
	// C   = (-1)  | f g h | = - ( b ( gl - hk ) - c ( fl - hj ) + d ( fk - gj ) )
	//  41         | j k l |
	//
	//           6 | a c d |
	// C   = (-1)  | e g h | = + ( a ( gl - hk ) - c ( el - hi ) + d ( ek - gi ) )
	//  42         | i k l |
	//
	//           7 | a b d |
	// C   = (-1)  | e f h | = - ( a ( fl - hj ) - b ( el - hi ) + d ( ej - fi ) )
	//  43         | i j l |
	//
	//           8 | a b c |
	// C   = (-1)  | e f g | = + ( a ( fk - gj ) - b ( ek - gi ) + c ( ej - fi ) )
	//  44         | i j k |
	//
	// Cost of operation
	// 53 adds, 104 muls, and 1 div.
	Matrix4x4 result;

	float a = matrix.M11, b = matrix.M12, c = matrix.M13, d = matrix.M14;
	float e = matrix.M21, f = matrix.M22, g = matrix.M23, h = matrix.M24;
	float i = matrix.M31, j = matrix.M32, k = matrix.M33, l = matrix.M34;
	float m = matrix.M41, n = matrix.M42, o = matrix.M43, p = matrix.M44;

	float kp_lo = k * p - l * o;
	float jp_ln = j * p - l * n;
	float jo_kn = j * o - k * n;
	float ip_lm = i * p - l * m;
	float io_km = i * o - k * m;
	float in_jm = i * n - j * m;

	float a11 = +(f * kp_lo - g * jp_ln + h * jo_kn);
	float a12 = -(e * kp_lo - g * ip_lm + h * io_km);
	float a13 = +(e * jp_ln - f * ip_lm + h * in_jm);
	float a14 = -(e * jo_kn - f * io_km + g * in_jm);

	float det = a * a11 + b * a12 + c * a13 + d * a14;

	if (abs(det) < FLT_EPSILON)
	{
		result = Matrix4x4(NAN, NAN, NAN, NAN,
			NAN, NAN, NAN, NAN,
			NAN, NAN, NAN, NAN,
			NAN, NAN, NAN, NAN);
		return false;
	}

	float invDet = 1.0f / det;

	result.M11 = a11 * invDet;
	result.M21 = a12 * invDet;
	result.M31 = a13 * invDet;
	result.M41 = a14 * invDet;

	result.M12 = -(b * kp_lo - c * jp_ln + d * jo_kn) * invDet;
	result.M22 = +(a * kp_lo - c * ip_lm + d * io_km) * invDet;
	result.M32 = -(a * jp_ln - b * ip_lm + d * in_jm) * invDet;
	result.M42 = +(a * jo_kn - b * io_km + c * in_jm) * invDet;

	float gp_ho = g * p - h * o;
	float fp_hn = f * p - h * n;
	float fo_gn = f * o - g * n;
	float ep_hm = e * p - h * m;
	float eo_gm = e * o - g * m;
	float en_fm = e * n - f * m;

	result.M13 = +(b * gp_ho - c * fp_hn + d * fo_gn) * invDet;
	result.M23 = -(a * gp_ho - c * ep_hm + d * eo_gm) * invDet;
	result.M33 = +(a * fp_hn - b * ep_hm + d * en_fm) * invDet;
	result.M43 = -(a * fo_gn - b * eo_gm + c * en_fm) * invDet;

	float gl_hk = g * l - h * k;
	float fl_hj = f * l - h * j;
	float fk_gj = f * k - g * j;
	float el_hi = e * l - h * i;
	float ek_gi = e * k - g * i;
	float ej_fi = e * j - f * i;

	result.M14 = -(b * gl_hk - c * fl_hj + d * fk_gj) * invDet;
	result.M24 = +(a * gl_hk - c * el_hi + d * ek_gi) * invDet;
	result.M34 = -(a * fl_hj - b * el_hi + d * ej_fi) * invDet;
	result.M44 = +(a * fk_gj - b * ek_gi + c * ej_fi) * invDet;

	return true;
}

//制造一个单位矩阵
Matrix4x4 Matrix4x4::eye()
{
	Matrix4x4 A = Matrix4x4();
	A.M11 = 1;
	A.M22 = 1;
	A.M33 = 1;
	A.M44 = 1;
	return A;
}

//实现矩阵的转置
Matrix4x4 Matrix4x4::transpose(const Matrix4x4& matrix)
{
	Matrix4x4 result;
	result.M11 = matrix.M11;
	result.M12 = matrix.M21;
	result.M13 = matrix.M31;
	result.M14 = matrix.M41;
	result.M21 = matrix.M12;
	result.M22 = matrix.M22;
	result.M23 = matrix.M32;
	result.M24 = matrix.M42;
	result.M31 = matrix.M13;
	result.M32 = matrix.M23;
	result.M33 = matrix.M33;
	result.M34 = matrix.M43;
	result.M41 = matrix.M14;
	result.M42 = matrix.M24;
	result.M43 = matrix.M34;
	result.M44 = matrix.M44;
	return result;
}

#pragma endregion
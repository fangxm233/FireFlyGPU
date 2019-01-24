#include "matrix.h"

using std::endl;
using std::cout;
using std::istream;

const double EPS = 1e-10;

#pragma region Matrix

void Matrix::initialize()
{
	//初始化矩阵大小
	p = new double*[rows_num];//分配rows_num个指针
	for (int i = 0; i < rows_num; ++i)
	{
		p[i] = new double[cols_num];//为p[i]进行动态内存分配，大小为cols
	}
}

///声明一个全0矩阵
Matrix::Matrix(int rows, int cols)
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

///声明一个值全部为value的矩阵
Matrix::Matrix(int rows, int cols, double value)
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

///析构函数
Matrix::~Matrix()
{

}

///实现矩阵的复制
Matrix& Matrix::operator=(const Matrix& m)
{
	if (this == &m)
	{
		return *this;
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
	return *this;
}

///将数组的值传递给矩阵(要求矩阵的大小已经被声明过了)
Matrix& Matrix::operator=(double *a)
{
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			p[i][j] = *(a + i * cols_num + j);
		}
	}
	return *this;
}

///+=操作
Matrix& Matrix::operator+=(const Matrix& m)
{
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			p[i][j] += m.p[i][j];
		}
	}
	return *this;
}

///实现-=
Matrix& Matrix::operator-=(const Matrix& m)
{
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < cols_num; j++)
		{
			p[i][j] -= m.p[i][j];
		}
	}
	return *this;
}

///实现*=
Matrix& Matrix::operator*=(const Matrix& m)
{
	Matrix temp(rows_num, m.cols_num);//若C=AB,则矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
	for (int i = 0; i < temp.rows_num; i++)
	{
		for (int j = 0; j < temp.cols_num; j++)
		{
			for (int k = 0; k < cols_num; k++)
			{
				temp.p[i][j] += (p[i][k] * m.p[k][j]);
			}
		}
	}
	*this = temp;
	return *this;
}

///实现矩阵的乘法
Matrix Matrix::operator*(const Matrix & m)const
{
	Matrix ba_M(rows_num, m.cols_num, 0.0);
	for (int i = 0; i < rows_num; i++)
	{
		for (int j = 0; j < m.cols_num; j++)
		{
			for (int k = 0; k < cols_num; k++)
			{
				ba_M.p[i][j] += (p[i][k] * m.p[k][j]);
			}
		}
	}
	return ba_M;
}

///矩阵显示
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

///实现行变换
void Matrix::swapRows(int a, int b)
{
	a--;
	b--;
	double *temp = p[a];
	p[a] = p[b];
	p[b] = temp;
}

///返回矩阵第i行第j列的数
double Matrix::Point(int i, int j) const
{
	return this->p[i][j];
}


///制造一个单位矩阵
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

///实现矩阵的转置
Matrix Matrix::T(const Matrix & m)
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

///声明一个全0矩阵
Matrix3x3::Matrix3x3()
{
	initialize();
}

///声明一个值全部为value的矩阵
Matrix3x3::Matrix3x3(double value)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] = value;
		}
	}
}

///析构函数
Matrix3x3::~Matrix3x3()
{

}

///实现矩阵的复制
Matrix3x3& Matrix3x3::operator=(const Matrix3x3& m)
{
	if (this == &m)
	{
		return *this;
	}

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] = *m.p[i][j];
		}
	}
	return *this;
}

///将数组的值传递给矩阵(要求矩阵的大小已经被声明过了)
Matrix3x3& Matrix3x3::operator=(double *a)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] = *(a + i * 3 + j);
		}
	}
	return *this;
}

///+=操作
Matrix3x3& Matrix3x3::operator+=(const Matrix3x3& m)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] += *m.p[i][j];
		}
	}
	return *this;
}

///实现-=
Matrix3x3& Matrix3x3::operator-=(const Matrix3x3& m)
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*p[i][j] -= *m.p[i][j];
		}
	}
	return *this;
}

///实现*=
Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& m)
{
	Matrix3x3 temp = Matrix3x3();//若C=AB,则矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				*temp.p[i][j] += (*p[i][k] * *m.p[k][j]);
			}
		}
	}
	*this = temp;
	return *this;
}

///实现矩阵的乘法
Matrix3x3 Matrix3x3::operator*(const Matrix3x3 & m)const
{
	Matrix3x3 ba_M = Matrix3x3();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				*ba_M.p[i][j] += (*p[i][k] * *m.p[k][j]);
			}
		}
	}
	return ba_M;
}

///矩阵显示
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

///返回矩阵第i行第j列的数
double Matrix3x3::Point(int i, int j) const
{
	return *p[i][j];
}


///制造一个单位矩阵
Matrix3x3 Matrix3x3::eye(int n)
{
	Matrix3x3 A = Matrix3x3();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
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

///实现矩阵的转置
Matrix3x3 Matrix3x3::T(const Matrix3x3 & m)
{
	Matrix3x3 mt = Matrix3x3();
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			*mt.p[j][i] = *m.p[i][j];
		}
	}
	return mt;
}

#pragma endregion

#pragma region Matrix4x4

void Matrix4x4::initialize()
{
	p = { 0 };
}

///声明一个全0矩阵
Matrix4x4::Matrix4x4()
{
	initialize();
}

///声明一个值全部为value的矩阵
Matrix4x4::Matrix4x4(double value)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			*p[i][j] = value;
		}
	}
}

///析构函数
Matrix4x4::~Matrix4x4()
{

}

///实现矩阵的复制
Matrix4x4& Matrix4x4::operator=(const Matrix4x4& m)
{
	if (this == &m)
	{
		return *this;
	}

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			*p[i][j] = *m.p[i][j];
		}
	}
	return *this;
}

///将数组的值传递给矩阵(要求矩阵的大小已经被声明过了)
Matrix4x4& Matrix4x4::operator=(double *a)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			*p[i][j] = *(a + i * 4 + j);
		}
	}
	return *this;
}

///+=操作
Matrix4x4& Matrix4x4::operator+=(const Matrix4x4& m)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			*p[i][j] += *m.p[i][j];
		}
	}
	return *this;
}

///实现-=
Matrix4x4& Matrix4x4::operator-=(const Matrix4x4& m)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			*p[i][j] -= *m.p[i][j];
		}
	}
	return *this;
}

///实现*=
Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& m)
{
	Matrix4x4 temp = Matrix4x4();//若C=AB,则矩阵C的行数等于矩阵A的行数，C的列数等于B的列数。
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				*temp.p[i][j] += (*p[i][k] * *m.p[k][j]);
			}
		}
	}
	*this = temp;
	return *this;
}

///实现矩阵的乘法
Matrix4x4 Matrix4x4::operator*(const Matrix4x4 & m)const
{
	Matrix4x4 ba_M = Matrix4x4();
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				*ba_M.p[i][j] += (*p[i][k] * *m.p[k][j]);
			}
		}
	}
	return ba_M;
}

///矩阵显示
void Matrix4x4::Show() const
{
	//cout << 4 <<" "<<4<< endl;//显示矩阵的行数和列数
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			cout << *p[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

///返回矩阵第i行第j列的数
double Matrix4x4::Point(int i, int j) const
{
	return *p[i][j];
}


///制造一个单位矩阵
Matrix4x4 Matrix4x4::eye(int n)
{
	Matrix4x4 A = Matrix4x4();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
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

///实现矩阵的转置
Matrix4x4 Matrix4x4::T(const Matrix4x4 & m)
{
	Matrix4x4 mt = Matrix4x4();
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			*mt.p[j][i] = *m.p[i][j];
		}
	}
	return mt;
}

#pragma endregion

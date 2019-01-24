#include "matrix.h"

using std::endl;
using std::cout;
using std::istream;

const double EPS = 1e-10;

#pragma region Matrix

void Matrix::initialize()
{
	//��ʼ�������С
	p = new double*[rows_num];//����rows_num��ָ��
	for (int i = 0; i < rows_num; ++i)
	{
		p[i] = new double[cols_num];//Ϊp[i]���ж�̬�ڴ���䣬��СΪcols
	}
}

///����һ��ȫ0����
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

///����һ��ֵȫ��Ϊvalue�ľ���
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

///��������
Matrix::~Matrix()
{

}

///ʵ�־���ĸ���
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

///�������ֵ���ݸ�����(Ҫ�����Ĵ�С�Ѿ�����������)
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

///+=����
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

///ʵ��-=
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

///ʵ��*=
Matrix& Matrix::operator*=(const Matrix& m)
{
	Matrix temp(rows_num, m.cols_num);//��C=AB,�����C���������ھ���A��������C����������B��������
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

///ʵ�־���ĳ˷�
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

///������ʾ
void Matrix::Show() const
{
	//cout << rows_num <<" "<<cols_num<< endl;//��ʾ���������������
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

///ʵ���б任
void Matrix::swapRows(int a, int b)
{
	a--;
	b--;
	double *temp = p[a];
	p[a] = p[b];
	p[b] = temp;
}

///���ؾ����i�е�j�е���
double Matrix::Point(int i, int j) const
{
	return this->p[i][j];
}


///����һ����λ����
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

///ʵ�־����ת��
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

///����һ��ȫ0����
Matrix3x3::Matrix3x3()
{
	initialize();
}

///����һ��ֵȫ��Ϊvalue�ľ���
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

///��������
Matrix3x3::~Matrix3x3()
{

}

///ʵ�־���ĸ���
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

///�������ֵ���ݸ�����(Ҫ�����Ĵ�С�Ѿ�����������)
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

///+=����
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

///ʵ��-=
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

///ʵ��*=
Matrix3x3& Matrix3x3::operator*=(const Matrix3x3& m)
{
	Matrix3x3 temp = Matrix3x3();//��C=AB,�����C���������ھ���A��������C����������B��������
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

///ʵ�־���ĳ˷�
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

///������ʾ
void Matrix3x3::Show() const
{
	//cout << 3 <<" "<<3<< endl;//��ʾ���������������
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

///���ؾ����i�е�j�е���
double Matrix3x3::Point(int i, int j) const
{
	return *p[i][j];
}


///����һ����λ����
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

///ʵ�־����ת��
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

///����һ��ȫ0����
Matrix4x4::Matrix4x4()
{
	initialize();
}

///����һ��ֵȫ��Ϊvalue�ľ���
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

///��������
Matrix4x4::~Matrix4x4()
{

}

///ʵ�־���ĸ���
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

///�������ֵ���ݸ�����(Ҫ�����Ĵ�С�Ѿ�����������)
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

///+=����
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

///ʵ��-=
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

///ʵ��*=
Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& m)
{
	Matrix4x4 temp = Matrix4x4();//��C=AB,�����C���������ھ���A��������C����������B��������
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

///ʵ�־���ĳ˷�
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

///������ʾ
void Matrix4x4::Show() const
{
	//cout << 4 <<" "<<4<< endl;//��ʾ���������������
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

///���ؾ����i�е�j�е���
double Matrix4x4::Point(int i, int j) const
{
	return *p[i][j];
}


///����һ����λ����
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

///ʵ�־����ת��
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

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <cmath>

class Matrix
{
private:
	double **p;
	void initialize();//初始化矩阵

public:
	int rows_num, cols_num;
	Matrix(int, int);
	Matrix(int, int, double);//预配分空间
	virtual ~Matrix();//析构函数应当是虚函数，除非此类不用做基类
	Matrix& operator=(const Matrix&);//矩阵的复制
	Matrix& operator=(double *);//将数组的值传给矩阵
	Matrix& operator+=(const Matrix&);//矩阵的+=操作
	Matrix& operator-=(const Matrix&);//-=
	Matrix& operator*=(const Matrix&);//*=
	Matrix operator*(const Matrix & m)const;
	void Show() const;//矩阵显示
	void swapRows(int, int);
	double Point(int i, int j) const;
	static Matrix eye(int);//制造一个单位矩阵
	static Matrix T(const Matrix & m);//矩阵转置的实现,且不改变矩阵
};

class Matrix3x3
{
private:
	double(*p)[3][3];
	void initialize();//初始化矩阵

public:
	Matrix3x3();
	Matrix3x3(double);//预配分空间
	virtual ~Matrix3x3();//析构函数应当是虚函数，除非此类不用做基类
	Matrix3x3& operator=(const Matrix3x3&);//矩阵的复制
	Matrix3x3& operator=(double *);//将数组的值传给矩阵
	Matrix3x3& operator+=(const Matrix3x3&);//矩阵的+=操作
	Matrix3x3& operator-=(const Matrix3x3&);//-=
	Matrix3x3& operator*=(const Matrix3x3&);//*=
	Matrix3x3 operator*(const Matrix3x3 & m)const;
	void Show() const;//矩阵显示
	double Point(int i, int j) const;
	static Matrix3x3 eye(int);//制造一个单位矩阵
	static Matrix3x3 T(const Matrix3x3 & m);//矩阵转置的实现,且不改变矩阵
};

class Matrix4x4
{
private:
	double(*p)[4][4];
	void initialize();//初始化矩阵

public:
	Matrix4x4();
	Matrix4x4(double);//预配分空间
	virtual ~Matrix4x4();//析构函数应当是虚函数，除非此类不用做基类
	Matrix4x4& operator=(const Matrix4x4&);//矩阵的复制
	Matrix4x4& operator=(double *);//将数组的值传给矩阵
	Matrix4x4& operator+=(const Matrix4x4&);//矩阵的+=操作
	Matrix4x4& operator-=(const Matrix4x4&);//-=
	Matrix4x4& operator*=(const Matrix4x4&);//*=
	Matrix4x4 operator*(const Matrix4x4 & m)const;
	void Show() const;//矩阵显示
	double Point(int i, int j) const;
	static Matrix4x4 eye(int);//制造一个单位矩阵
	static Matrix4x4 T(const Matrix4x4 & m);//矩阵转置的实现,且不改变矩阵
};
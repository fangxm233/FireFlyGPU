#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <cmath>

class Matrix
{
private:
	double **p;
	void initialize();//��ʼ������

public:
	int rows_num, cols_num;
	Matrix(int, int);
	Matrix(int, int, double);//Ԥ��ֿռ�
	virtual ~Matrix();//��������Ӧ�����麯�������Ǵ��಻��������
	Matrix& operator=(const Matrix&);//����ĸ���
	Matrix& operator=(double *);//�������ֵ��������
	Matrix& operator+=(const Matrix&);//�����+=����
	Matrix& operator-=(const Matrix&);//-=
	Matrix& operator*=(const Matrix&);//*=
	Matrix operator*(const Matrix & m)const;
	void Show() const;//������ʾ
	void swapRows(int, int);
	double Point(int i, int j) const;
	static Matrix eye(int);//����һ����λ����
	static Matrix T(const Matrix & m);//����ת�õ�ʵ��,�Ҳ��ı����
};

class Matrix3x3
{
private:
	double(*p)[3][3];
	void initialize();//��ʼ������

public:
	Matrix3x3();
	Matrix3x3(double);//Ԥ��ֿռ�
	virtual ~Matrix3x3();//��������Ӧ�����麯�������Ǵ��಻��������
	Matrix3x3& operator=(const Matrix3x3&);//����ĸ���
	Matrix3x3& operator=(double *);//�������ֵ��������
	Matrix3x3& operator+=(const Matrix3x3&);//�����+=����
	Matrix3x3& operator-=(const Matrix3x3&);//-=
	Matrix3x3& operator*=(const Matrix3x3&);//*=
	Matrix3x3 operator*(const Matrix3x3 & m)const;
	void Show() const;//������ʾ
	double Point(int i, int j) const;
	static Matrix3x3 eye(int);//����һ����λ����
	static Matrix3x3 T(const Matrix3x3 & m);//����ת�õ�ʵ��,�Ҳ��ı����
};

class Matrix4x4
{
private:
	double(*p)[4][4];
	void initialize();//��ʼ������

public:
	Matrix4x4();
	Matrix4x4(double);//Ԥ��ֿռ�
	virtual ~Matrix4x4();//��������Ӧ�����麯�������Ǵ��಻��������
	Matrix4x4& operator=(const Matrix4x4&);//����ĸ���
	Matrix4x4& operator=(double *);//�������ֵ��������
	Matrix4x4& operator+=(const Matrix4x4&);//�����+=����
	Matrix4x4& operator-=(const Matrix4x4&);//-=
	Matrix4x4& operator*=(const Matrix4x4&);//*=
	Matrix4x4 operator*(const Matrix4x4 & m)const;
	void Show() const;//������ʾ
	double Point(int i, int j) const;
	static Matrix4x4 eye(int);//����һ����λ����
	static Matrix4x4 T(const Matrix4x4 & m);//����ת�õ�ʵ��,�Ҳ��ı����
};
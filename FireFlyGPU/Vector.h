#pragma once
#include <math.h>
#include <stdlib.h>
#include <iostream>

///host使用的向量
struct Vector2
{
public:
	Vector2() {}
	Vector2(float e0, float e1) { X = e0; Y = e1; }

	const Vector2& operator+() const { return *this; }
	Vector2 operator-() const { return Vector2(-X, -Y); }

	Vector2& operator+=(const Vector2 &v2);
	Vector2& operator-=(const Vector2 &v2);
	Vector2& operator*=(const Vector2 &v2);
	Vector2& operator/=(const Vector2 &v2);
	Vector2& operator*=(const float t);
	Vector2& operator/=(const float t);

	float length() const { return sqrt(X * X + Y * Y); }
	float squared_length() const { return X * X + Y * Y; }
	void Normalize();

	float X, Y;
};

///host使用的向量
struct Vector3
{
public:
	Vector3() {}
	Vector3(float x, float y, float z) { X = x; Y = y; Z = z; }

	const Vector3& operator+() const { return *this; }
	Vector3 operator-() const { return Vector3(-X, -Y, -Z); }

	Vector3& operator+=(const Vector3 &v2);
	Vector3& operator-=(const Vector3 &v2);
	Vector3& operator*=(const Vector3 &v2);
	Vector3& operator/=(const Vector3 &v2);
	Vector3& operator*=(const float t);
	Vector3& operator/=(const float t);

	float length() const { return sqrt(X * X + Y * Y + Z * Z); }
	float squared_length() const { return X * X + Y * Y + Z * Z; }
	void Normalize();

	float X, Y, Z;
};

///host使用的向量
struct Vector4
{
public:
	Vector4() {}
	Vector4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }

	const Vector4& operator+() const { return *this; }
	Vector4 operator-() const { return Vector4(-X, -Y, -Z, -W); }

	Vector4& operator+=(const Vector4 &v2);
	Vector4& operator-=(const Vector4 &v2);
	Vector4& operator*=(const Vector4 &v2);
	Vector4& operator/=(const Vector4 &v2);
	Vector4& operator*=(const float t);
	Vector4& operator/=(const float t);

	float length() const { return sqrt(X * X + Y * Y + Z * Z + W * W); }
	float squared_length() const { return X * X + Y * Y + Z * Z + W * W; }
	void Normalize();

	float X, Y, Z, W;
};

#pragma region Vector2

std::istream& operator>>(std::istream &is, Vector2 &t)
{
	is >> t.X >> t.Y;
	return is;
}

std::ostream& operator<<(std::ostream &os, const Vector2 &t)
{
	os << t.X << " " << t.Y;
	return os;
}

void Vector2::Normalize()
{
	float k = 1.0f / sqrt(X * X + Y * Y);
	X *= k; Y *= k;
}

Vector2 operator+(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X + v2.X, v1.Y + v2.Y);
}

Vector2 operator-(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X - v2.X, v1.Y - v2.Y);
}

Vector2 operator*(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X * v2.X, v1.Y * v2.Y);
}

Vector2 operator/(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X / v2.X, v1.Y / v2.Y);
}

Vector2 operator*(float t, const Vector2 &v)
{
	return Vector2(t*v.X, t*v.Y);
}

Vector2 operator/(Vector2 v, float t)
{
	return Vector2(v.X / t, v.Y / t);
}

Vector2 operator*(const Vector2 &v, float t)
{
	return Vector2(t*v.X, t*v.Y);
}

float dot(const Vector2 &v1, const Vector2 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y;
}

Vector2& Vector2::operator+=(const Vector2 &v)
{
	X += v.X;
	Y += v.Y;
	return *this;
}

Vector2& Vector2::operator*=(const Vector2 &v)
{
	X *= v.X;
	Y *= v.Y;
	return *this;
}

Vector2& Vector2::operator/=(const Vector2 &v)
{
	X /= v.X;
	Y /= v.Y;
	return *this;
}

Vector2& Vector2::operator-=(const Vector2& v)
{
	X -= v.X;
	Y -= v.Y;
	return *this;
}

Vector2& Vector2::operator*=(const float t)
{
	X *= t;
	Y *= t;
	return *this;
}

Vector2& Vector2::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	return *this;
}

Vector2 Normalize(Vector2 v)
{
	return v / v.length();
}

#pragma endregion

#pragma region Vector3

std::istream& operator>>(std::istream &is, Vector3 &t)
{
	is >> t.X >> t.Y >> t.Z;
	return is;
}

std::ostream& operator<<(std::ostream &os, const Vector3 &t)
{
	os << t.X << " " << t.Y << " " << t.Z;
	return os;
}

void Vector3::Normalize()
{
	float k = 1.0f / sqrt(X * X + Y * Y + Z * Z);
	X *= k; Y *= k; Z *= k;
}

Vector3 operator+(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z);
}

Vector3 operator-(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z);
}

Vector3 operator*(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z);
}

Vector3 operator/(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X / v2.X, v1.Y / v2.Y, v1.Z / v2.Z);
}

Vector3 operator*(float t, const Vector3 &v)
{
	return Vector3(t*v.X, t*v.Y, t*v.Z);
}

Vector3 operator/(Vector3 v, float t)
{
	return Vector3(v.X / t, v.Y / t, v.Z / t);
}

Vector3 operator*(const Vector3 &v, float t)
{
	return Vector3(t*v.X, t*v.Y, t*v.Z);
}

float dot(const Vector3 &v1, const Vector3 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

Vector3 cross(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3((v1.Y * v2.Z - v1.Z * v2.Y),
		(-(v1.X * v2.Z - v1.Z * v2.X)),
		(v1.X * v2.Y - v1.Y * v2.X));
}


Vector3& Vector3::operator+=(const Vector3 &v)
{
	X += v.X;
	Y += v.Y;
	Z += v.Z;
	return *this;
}

Vector3& Vector3::operator*=(const Vector3 &v)
{
	X *= v.X;
	Y *= v.Y;
	Z *= v.Z;
	return *this;
}

Vector3& Vector3::operator/=(const Vector3 &v)
{
	X /= v.X;
	Y /= v.Y;
	Z /= v.Z;
	return *this;
}

Vector3& Vector3::operator-=(const Vector3& v)
{
	X -= v.X;
	Y -= v.Y;
	Z -= v.Z;
	return *this;
}

Vector3& Vector3::operator*=(const float t)
{
	X *= t;
	Y *= t;
	Z *= t;
	return *this;
}

Vector3& Vector3::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	Z *= k;
	return *this;
}

Vector3 Normalize(Vector3 v)
{
	return v / v.length();
}

#pragma endregion

#pragma region Vector4

std::istream& operator>>(std::istream &is, Vector4 &t)
{
	is >> t.X >> t.Y >> t.Z;
	return is;
}

std::ostream& operator<<(std::ostream &os, const Vector4 &t)
{
	os << t.X << " " << t.Y << " " << t.Z;
	return os;
}

void Vector4::Normalize()
{
	float k = 1.0f / length();
	X *= k; Y *= k; Z *= k; W *= k;
}

Vector4 operator+(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z, v1.W + v2.W);
}

Vector4 operator-(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z, v1.W - v2.W);
}

Vector4 operator*(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z, v1.W * v2.W);
}

Vector4 operator/(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X / v2.X, v1.Y / v2.Y, v1.Z / v2.Z, v1.W / v2.W);
}

Vector4 operator*(float t, const Vector4 &v)
{
	return Vector4(t * v.X, t * v.Y, t * v.Z, t * v.W);
}

Vector4 operator/(Vector4 v, float t)
{
	return Vector4(v.X / t, v.Y / t, v.Z / t, v.W / t);
}

Vector4 operator*(const Vector4 &v, float t)
{
	return Vector4(t * v.X, t * v.Y, t * v.Z, t * v.W);
}

float dot(const Vector4 &v1, const Vector4 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

Vector4& Vector4::operator+=(const Vector4 &v)
{
	X += v.X;
	Y += v.Y;
	Z += v.Z;
	W += v.W;
	return *this;
}

Vector4& Vector4::operator*=(const Vector4 &v)
{
	X *= v.X;
	Y *= v.Y;
	Z *= v.Z;
	W *= v.W;
	return *this;
}

Vector4& Vector4::operator/=(const Vector4 &v)
{
	X /= v.X;
	Y /= v.Y;
	Z /= v.Z;
	W /= v.W;
	return *this;
}

Vector4& Vector4::operator-=(const Vector4& v)
{
	X -= v.X;
	Y -= v.Y;
	Z -= v.Z;
	W -= v.W;
	return *this;
}

Vector4& Vector4::operator*=(const float t)
{
	X *= t;
	Y *= t;
	Z *= t;
	W *= t;
	return *this;
}

Vector4& Vector4::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	Z *= k;
	W *= k;
	return *this;
}

Vector4 Normalize(Vector4 v)
{
	return v / v.length();
}

#pragma endregion
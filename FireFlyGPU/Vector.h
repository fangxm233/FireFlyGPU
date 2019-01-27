#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "Matrix.h"

struct Vector2
{
public:
	__device__ __host__ Vector2() {}
	__device__ __host__ Vector2(float e0, float e1) { X = e0; Y = e1; }

	__device__ __host__ inline const Vector2& operator+() const { return *this; }
	__device__ __host__ inline Vector2 operator-() const { return Vector2(-X, -Y); }

	__device__ __host__ inline Vector2& operator+=(const Vector2 &v2);
	__device__ __host__ inline Vector2& operator-=(const Vector2 &v2);
	__device__ __host__ inline Vector2& operator*=(const Vector2 &v2);
	__device__ __host__ inline Vector2& operator/=(const Vector2 &v2);
	__device__ __host__ inline Vector2& operator*=(const float t);
	__device__ __host__ inline Vector2& operator/=(const float t);

	__device__ __host__ inline float length() const { return sqrt(X * X + Y * Y); }
	__device__ __host__ inline float squared_length() const { return X * X + Y * Y; }
	__device__ __host__ inline void Normalize();

	float X, Y;
};

struct Vector3
{
public:
	__device__ __host__ Vector3() {}
	__device__ __host__ Vector3(float x, float y, float z) { X = x; Y = y; Z = z; }

	const Vector3& operator+() const { return *this; }
	__device__ __host__ inline Vector3 operator-() const { return Vector3(-X, -Y, -Z); }

	__device__ __host__ inline Vector3& operator+=(const Vector3 &v2);
	__device__ __host__ inline Vector3& operator-=(const Vector3 &v2);
	__device__ __host__ inline Vector3& operator*=(const Vector3 &v2);
	__device__ __host__ inline Vector3& operator/=(const Vector3 &v2);
	__device__ __host__ inline Vector3& operator*=(const float t);
	__device__ __host__ inline Vector3& operator/=(const float t);

	__device__ __host__ inline float length() const { return sqrt(X * X + Y * Y + Z * Z); }
	__device__ __host__ inline float squared_length() const { return X * X + Y * Y + Z * Z; }
	__device__ __host__ inline void Normalize();

	float X, Y, Z;
};

struct Vector4
{
public:
	__device__ __host__ Vector4() {}
	__device__ __host__ Vector4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }

	__device__ __host__ inline const Vector4& operator+() const { return *this; }
	__device__ __host__ inline Vector4 operator-() const { return Vector4(-X, -Y, -Z, -W); }

	__device__ __host__ inline Vector4& operator+=(const Vector4 &v2);
	__device__ __host__ inline Vector4& operator-=(const Vector4 &v2);
	__device__ __host__ inline Vector4& operator*=(const Vector4 &v2);
	__device__ __host__ inline Vector4& operator/=(const Vector4 &v2);
	__device__ __host__ inline Vector4& operator*=(const float t);
	__device__ __host__ inline Vector4& operator/=(const float t);

	__device__ __host__ inline float length() const { return sqrt(X * X + Y * Y + Z * Z + W * W); }
	__device__ __host__ inline float squared_length() const { return X * X + Y * Y + Z * Z + W * W; }
	__device__ __host__ inline void Normalize();

	float X, Y, Z, W;
};

#pragma region Vector2

__host__ inline std::istream& operator>>(std::istream &is, Vector2 &t)
{
	is >> t.X >> t.Y;
	return is;
}

__host__ inline std::ostream& operator<<(std::ostream &os, const Vector2 &t)
{
	os << t.X << " " << t.Y;
	return os;
}

__device__ __host__ inline void Vector2::Normalize()
{
	float k = 1.0f / sqrt(X * X + Y * Y);
	X *= k; Y *= k;
}

__device__ __host__ inline Vector2 operator+(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X + v2.X, v1.Y + v2.Y);
}

__device__ __host__ inline Vector2 operator-(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X - v2.X, v1.Y - v2.Y);
}

__device__ __host__ inline Vector2 operator*(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X * v2.X, v1.Y * v2.Y);
}

__device__ __host__ inline Vector2 operator/(const Vector2 &v1, const Vector2 &v2)
{
	return Vector2(v1.X / v2.X, v1.Y / v2.Y);
}

__device__ __host__ inline Vector2 operator*(float t, const Vector2 &v)
{
	return Vector2(t*v.X, t*v.Y);
}

__device__ __host__ inline Vector2 operator/(Vector2 v, float t)
{
	return Vector2(v.X / t, v.Y / t);
}

__device__ __host__ inline Vector2 operator*(const Vector2 &v, float t)
{
	return Vector2(t*v.X, t*v.Y);
}

__device__ __host__ inline float dot(const Vector2 &v1, const Vector2 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y;
}

__device__ __host__ inline Vector2& Vector2::operator+=(const Vector2 &v)
{
	X += v.X;
	Y += v.Y;
	return *this;
}

__device__ __host__ inline Vector2& Vector2::operator*=(const Vector2 &v)
{
	X *= v.X;
	Y *= v.Y;
	return *this;
}

__device__ __host__ inline Vector2& Vector2::operator/=(const Vector2 &v)
{
	X /= v.X;
	Y /= v.Y;
	return *this;
}

__device__ __host__ inline Vector2& Vector2::operator-=(const Vector2& v)
{
	X -= v.X;
	Y -= v.Y;
	return *this;
}

__device__ __host__ inline Vector2& Vector2::operator*=(const float t)
{
	X *= t;
	Y *= t;
	return *this;
}

__device__ __host__ inline Vector2& Vector2::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	return *this;
}

__device__ __host__ inline Vector2 Normalize(Vector2 v)
{
	return v / v.length();
}

#pragma endregion

#pragma region Vector3

__host__ inline std::istream& operator>>(std::istream &is, Vector3 &t)
{
	is >> t.X >> t.Y >> t.Z;
	return is;
}

__host__ inline std::ostream& operator<<(std::ostream &os, const Vector3 &t)
{
	os << t.X << " " << t.Y << " " << t.Z;
	return os;
}

__device__ __host__ inline void Vector3::Normalize()
{
	float k = 1.0f / sqrt(X * X + Y * Y + Z * Z);
	X *= k; Y *= k; Z *= k;
}

__device__ __host__ inline Vector3 operator+(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z);
}

__device__ __host__ inline Vector3 operator-(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z);
}

__device__ __host__ inline Vector3 operator*(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z);
}

__device__ __host__ inline Vector3 operator*(const Matrix4x4 &m, const Vector3 &v)
{
	return Vector3(
		v.X * m.M11 + v.Y * m.M12 + v.Z * m.M13 + m.M14,
		v.X * m.M21 + v.Y * m.M22 + v.Z * m.M23 + m.M24,
		v.X * m.M31 + v.Y * m.M32 + v.Z * m.M33 + m.M34);
}

__device__ __host__ inline Vector3 operator*(const Vector3 &v, const Matrix4x4 &m)
{
	return Vector3(
		v.X * m.M11 + v.Y * m.M12 + v.Z * m.M13 + m.M14,
		v.X * m.M21 + v.Y * m.M22 + v.Z * m.M23 + m.M24,
		v.X * m.M31 + v.Y * m.M32 + v.Z * m.M33 + m.M34);
}

__device__ __host__ inline Vector3 operator/(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3(v1.X / v2.X, v1.Y / v2.Y, v1.Z / v2.Z);
}

__device__ __host__ inline Vector3 operator*(float t, const Vector3 &v)
{
	return Vector3(t*v.X, t*v.Y, t*v.Z);
}

__device__ __host__ inline Vector3 operator/(Vector3 v, float t)
{
	return Vector3(v.X / t, v.Y / t, v.Z / t);
}

__device__ __host__ inline Vector3 operator*(const Vector3 &v, float t)
{
	return Vector3(t*v.X, t*v.Y, t*v.Z);
}

__device__ __host__ inline float dot(const Vector3 &v1, const Vector3 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

__device__ __host__ inline Vector3 cross(const Vector3 &v1, const Vector3 &v2)
{
	return Vector3((v1.Y * v2.Z - v1.Z * v2.Y),
		(-(v1.X * v2.Z - v1.Z * v2.X)),
		(v1.X * v2.Y - v1.Y * v2.X));
}


__device__ __host__ inline Vector3& Vector3::operator+=(const Vector3 &v)
{
	X += v.X;
	Y += v.Y;
	Z += v.Z;
	return *this;
}

__device__ __host__ inline Vector3& Vector3::operator*=(const Vector3 &v)
{
	X *= v.X;
	Y *= v.Y;
	Z *= v.Z;
	return *this;
}

__device__ __host__ inline Vector3& Vector3::operator/=(const Vector3 &v)
{
	X /= v.X;
	Y /= v.Y;
	Z /= v.Z;
	return *this;
}

__device__ __host__ inline Vector3& Vector3::operator-=(const Vector3& v)
{
	X -= v.X;
	Y -= v.Y;
	Z -= v.Z;
	return *this;
}

__device__ __host__ inline Vector3& Vector3::operator*=(const float t)
{
	X *= t;
	Y *= t;
	Z *= t;
	return *this;
}

__device__ __host__ inline Vector3& Vector3::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	Z *= k;
	return *this;
}

__device__ __host__ inline Vector3 Normalize(Vector3 v)
{
	return v / v.length();
}

#pragma endregion

#pragma region Vector4

__host__ inline std::istream& operator>>(std::istream &is, Vector4 &t)
{
	is >> t.X >> t.Y >> t.Z;
	return is;
}

__host__ inline std::ostream& operator<<(std::ostream &os, const Vector4 &t)
{
	os << t.X << " " << t.Y << " " << t.Z;
	return os;
}

__device__ __host__ inline void Vector4::Normalize()
{
	float k = 1.0f / length();
	X *= k; Y *= k; Z *= k; W *= k;
}

__device__ __host__ inline Vector4 operator+(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z, v1.W + v2.W);
}

__device__ __host__ inline Vector4 operator-(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z, v1.W - v2.W);
}

__device__ __host__ inline Vector4 operator*(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z, v1.W * v2.W);
}

__device__ __host__ inline Vector4 operator/(const Vector4 &v1, const Vector4 &v2)
{
	return Vector4(v1.X / v2.X, v1.Y / v2.Y, v1.Z / v2.Z, v1.W / v2.W);
}

__device__ __host__ inline Vector4 operator*(float t, const Vector4 &v)
{
	return Vector4(t * v.X, t * v.Y, t * v.Z, t * v.W);
}

__device__ __host__ inline Vector4 operator*(const Matrix4x4 &m, const Vector4 &v)
{
	return Vector4	(
		v.X * m.M11 + v.Y * m.M12 + v.Z * m.M13 + v.W * m.M14,
		v.X * m.M21 + v.Y * m.M22 + v.Z * m.M23 + v.W * m.M24,
		v.X * m.M31 + v.Y * m.M32 + v.Z * m.M33 + v.W * m.M34,
		v.X * m.M41 + v.Y * m.M42 + v.Z * m.M43 + v.W * m.M44);
}

__device__ __host__ inline Vector4 operator*(const Vector4 &v, const Matrix4x4 &m)
{
	return Vector4(
		v.X * m.M11 + v.Y * m.M12 + v.Z * m.M13 + v.W * m.M14,
		v.X * m.M21 + v.Y * m.M22 + v.Z * m.M23 + v.W * m.M24,
		v.X * m.M31 + v.Y * m.M32 + v.Z * m.M33 + v.W * m.M34,
		v.X * m.M41 + v.Y * m.M42 + v.Z * m.M43 + v.W * m.M44);
}

__device__ __host__ inline Vector4 operator/(Vector4 v, float t)
{
	return Vector4(v.X / t, v.Y / t, v.Z / t, v.W / t);
}

__device__ __host__ inline Vector4 operator*(const Vector4 &v, float t)
{
	return Vector4(t * v.X, t * v.Y, t * v.Z, t * v.W);
}

__device__ __host__ inline float dot(const Vector4 &v1, const Vector4 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

__device__ __host__ inline Vector4& Vector4::operator+=(const Vector4 &v)
{
	X += v.X;
	Y += v.Y;
	Z += v.Z;
	W += v.W;
	return *this;
}

__device__ __host__ inline Vector4& Vector4::operator*=(const Vector4 &v)
{
	X *= v.X;
	Y *= v.Y;
	Z *= v.Z;
	W *= v.W;
	return *this;
}

__device__ __host__ inline Vector4& Vector4::operator/=(const Vector4 &v)
{
	X /= v.X;
	Y /= v.Y;
	Z /= v.Z;
	W /= v.W;
	return *this;
}

__device__ __host__ inline Vector4& Vector4::operator-=(const Vector4& v)
{
	X -= v.X;
	Y -= v.Y;
	Z -= v.Z;
	W -= v.W;
	return *this;
}

__device__ __host__ inline Vector4& Vector4::operator*=(const float t)
{
	X *= t;
	Y *= t;
	Z *= t;
	W *= t;
	return *this;
}

__device__ __host__ inline Vector4& Vector4::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	Z *= k;
	W *= k;
	return *this;
}

__device__ __host__ inline Vector4 Normalize(Vector4 v)
{
	return v / v.length();
}

#pragma endregion
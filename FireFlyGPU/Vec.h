#pragma once
#include "device_launch_parameters.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "Vector.h"

///device使用的向量
struct Vec2
{
public:
	__device__ Vec2() {}
	__device__ Vec2(float e0, float e1) { X = e0; Y = e1; }
	__device__ Vec2(Vector2 Vec2) { X = Vec2.X; Y = Vec2.Y; }

	__device__ inline const Vec2& operator+() const { return *this; }
	__device__ inline Vec2 operator-() const { return Vec2(-X, -Y); }

	__device__ inline Vec2& operator+=(const Vec2 &v2);
	__device__ inline Vec2& operator-=(const Vec2 &v2);
	__device__ inline Vec2& operator*=(const Vec2 &v2);
	__device__ inline Vec2& operator/=(const Vec2 &v2);
	__device__ inline Vec2& operator*=(const float t);
	__device__ inline Vec2& operator/=(const float t);

	__device__ inline float length() const { return sqrt(X * X + Y * Y); }
	__device__ inline float squared_length() const { return X * X + Y * Y; }
	__device__ inline void Normalize();

	Vector2 ToVector2();

	float X, Y;
};

///device使用的向量
struct Vec3
{
public:
	__device__ Vec3() {}
	__device__ Vec3(float e0, float e1, float e2) { X = e0; Y = e1; Z = e2; }
	__device__ Vec3(Vector3 vec3) { X = vec3.X; Y = vec3.Y; Z = vec3.Z; }

	__device__ inline const Vec3& operator+() const { return *this; }
	__device__ inline Vec3 operator-() const { return Vec3(-X, -Y, -Z); }

	__device__ inline Vec3& operator+=(const Vec3 &v2);
	__device__ inline Vec3& operator-=(const Vec3 &v2);
	__device__ inline Vec3& operator*=(const Vec3 &v2);
	__device__ inline Vec3& operator/=(const Vec3 &v2);
	__device__ inline Vec3& operator*=(const float t);
	__device__ inline Vec3& operator/=(const float t);

	__device__ inline float length() const { return sqrt(X * X + Y * Y + Z * Z); }
	__device__ inline float squared_length() const { return X * X + Y * Y + Z * Z; }
	__device__ inline void Normalize();

	Vector3 ToVector3();

	float X, Y, Z;
};

///device使用的向量
struct Vec4
{
public:
	__device__ Vec4() {}
	__device__ Vec4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; }
	__device__ Vec4(float value) { X = Y = Z = W = value; }

	__device__ inline const Vec4& operator+() const { return *this; }
	__device__ inline Vec4 operator-() const { return Vec4(-X, -Y, -Z, -W); }

	__device__ inline Vec4& operator+=(const Vec4 &v2);
	__device__ inline Vec4& operator-=(const Vec4 &v2);
	__device__ inline Vec4& operator*=(const Vec4 &v2);
	__device__ inline Vec4& operator/=(const Vec4 &v2);
	__device__ inline Vec4& operator*=(const float t);
	__device__ inline Vec4& operator/=(const float t);

	__device__ inline float length() const { return sqrt(X * X + Y * Y + Z * Z + W * W); }
	__device__ inline float squared_length() const { return X * X + Y * Y + Z * Z + W * W; }
	__device__ inline void Normalize();

	Vector4 ToVector4();

	float X, Y, Z, W;
};

#pragma region Vec2

__device__ inline std::istream& operator>>(std::istream &is, Vec2 &t)
{
	is >> t.X >> t.Y;
	return is;
}

__device__ inline std::ostream& operator<<(std::ostream &os, const Vec2 &t)
{
	os << t.X << " " << t.Y;
	return os;
}

__device__ inline void Vec2::Normalize()
{
	float k = 1.0f / sqrt(X * X + Y * Y);
	X *= k; Y *= k;
}

inline Vector2 Vec2::ToVector2()
{
	return Vector2(X, Y);
}

__device__ inline Vec2 operator+(const Vec2 &v1, const Vec2 &v2)
{
	return Vec2(v1.X + v2.X, v1.Y + v2.Y);
}

__device__ inline Vec2 operator-(const Vec2 &v1, const Vec2 &v2)
{
	return Vec2(v1.X - v2.X, v1.Y - v2.Y);
}

__device__ inline Vec2 operator*(const Vec2 &v1, const Vec2 &v2)
{
	return Vec2(v1.X * v2.X, v1.Y * v2.Y);
}

__device__ inline Vec2 operator/(const Vec2 &v1, const Vec2 &v2)
{
	return Vec2(v1.X / v2.X, v1.Y / v2.Y);
}

__device__ inline Vec2 operator*(float t, const Vec2 &v)
{
	return Vec2(t*v.X, t*v.Y);
}

__device__ inline Vec2 operator/(Vec2 v, float t)
{
	return Vec2(v.X / t, v.Y / t);
}

__device__ inline Vec2 operator*(const Vec2 &v, float t)
{
	return Vec2(t*v.X, t*v.Y);
}

__device__ inline float dot(const Vec2 &v1, const Vec2 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y;
}

__device__ inline Vec2& Vec2::operator+=(const Vec2 &v)
{
	X += v.X;
	Y += v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator*=(const Vec2 &v)
{
	X *= v.X;
	Y *= v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator/=(const Vec2 &v)
{
	X /= v.X;
	Y /= v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator-=(const Vec2& v)
{
	X -= v.X;
	Y -= v.Y;
	return *this;
}

__device__ inline Vec2& Vec2::operator*=(const float t)
{
	X *= t;
	Y *= t;
	return *this;
}

__device__ inline Vec2& Vec2::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	return *this;
}

__device__ inline Vec2 Normalize(Vec2 v)
{
	return v / v.length();
}

#pragma endregion

#pragma region Vec3

__device__ inline std::istream& operator>>(std::istream &is, Vec3 &t)
{
	is >> t.X >> t.Y >> t.Z;
	return is;
}

__device__ inline std::ostream& operator<<(std::ostream &os, const Vec3 &t)
{
	os << t.X << " " << t.Y << " " << t.Z;
	return os;
}

__device__ inline void Vec3::Normalize()
{
	float k = 1.0f / sqrt(X * X + Y * Y + Z * Z);
	X *= k; Y *= k; Z *= k;
}

inline Vector3 Vec3::ToVector3()
{
	return Vector3(X, Y, Z);
}

__device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
	return Vec3(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z);
}

__device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2)
{
	return Vec3(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z);
}

__device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2)
{
	return Vec3(v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z);
}

__device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2)
{
	return Vec3(v1.X / v2.X, v1.Y / v2.Y, v1.Z / v2.Z);
}

__device__ inline Vec3 operator*(float t, const Vec3 &v)
{
	return Vec3(t*v.X, t*v.Y, t*v.Z);
}

__device__ inline Vec3 operator/(Vec3 v, float t)
{
	return Vec3(v.X / t, v.Y / t, v.Z / t);
}

__device__ inline Vec3 operator*(const Vec3 &v, float t)
{
	return Vec3(t*v.X, t*v.Y, t*v.Z);
}

__device__ inline float dot(const Vec3 &v1, const Vec3 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

__device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2)
{
	return Vec3((v1.Y * v2.Z - v1.Z * v2.Y),
		(-(v1.X * v2.Z - v1.Z * v2.X)),
		(v1.X * v2.Y - v1.Y * v2.X));
}


__device__ inline Vec3& Vec3::operator+=(const Vec3 &v)
{
	X += v.X;
	Y += v.Y;
	Z += v.Z;
	return *this;
}

__device__ inline Vec3& Vec3::operator*=(const Vec3 &v)
{
	X *= v.X;
	Y *= v.Y;
	Z *= v.Z;
	return *this;
}

__device__ inline Vec3& Vec3::operator/=(const Vec3 &v)
{
	X /= v.X;
	Y /= v.Y;
	Z /= v.Z;
	return *this;
}

__device__ inline Vec3& Vec3::operator-=(const Vec3& v)
{
	X -= v.X;
	Y -= v.Y;
	Z -= v.Z;
	return *this;
}

__device__ inline Vec3& Vec3::operator*=(const float t)
{
	X *= t;
	Y *= t;
	Z *= t;
	return *this;
}

__device__ inline Vec3& Vec3::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	Z *= k;
	return *this;
}

__device__ inline Vec3 Normalize(Vec3 v)
{
	return v / v.length();
}

#pragma endregion

#pragma region Vec4

__device__ inline std::istream& operator>>(std::istream &is, Vec4 &t)
{
	is >> t.X >> t.Y >> t.Z;
	return is;
}

__device__ inline std::ostream& operator<<(std::ostream &os, const Vec4 &t)
{
	os << t.X << " " << t.Y << " " << t.Z;
	return os;
}

__device__ inline void Vec4::Normalize()
{
	float k = 1.0f / length();
	X *= k; Y *= k; Z *= k; W *= k;
}

inline Vector4 Vec4::ToVector4()
{
	return Vector4(X, Y, Z, W);
}

__device__ inline Vec4 operator+(const Vec4 &v1, const Vec4 &v2)
{
	return Vec4(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z, v1.W + v2.W);
}

__device__ inline Vec4 operator-(const Vec4 &v1, const Vec4 &v2)
{
	return Vec4(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z, v1.W - v2.W);
}

__device__ inline Vec4 operator*(const Vec4 &v1, const Vec4 &v2)
{
	return Vec4(v1.X * v2.X, v1.Y * v2.Y, v1.Z * v2.Z, v1.W * v2.W);
}

__device__ inline Vec4 operator/(const Vec4 &v1, const Vec4 &v2)
{
	return Vec4(v1.X / v2.X, v1.Y / v2.Y, v1.Z / v2.Z, v1.W / v2.W);
}

__device__ inline Vec4 operator*(float t, const Vec4 &v)
{
	return Vec4(t * v.X, t * v.Y, t * v.Z, t * v.W);
}

__device__ inline Vec4 operator/(Vec4 v, float t)
{
	return Vec4(v.X / t, v.Y / t, v.Z / t, v.W / t);
}

__device__ inline Vec4 operator*(const Vec4 &v, float t)
{
	return Vec4(t * v.X, t * v.Y, t * v.Z, t * v.W);
}

__device__ inline float dot(const Vec4 &v1, const Vec4 &v2)
{
	return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z;
}

__device__ inline Vec4& Vec4::operator+=(const Vec4 &v)
{
	X += v.X;
	Y += v.Y;
	Z += v.Z;
	W += v.W;
	return *this;
}

__device__ inline Vec4& Vec4::operator*=(const Vec4 &v)
{
	X *= v.X;
	Y *= v.Y;
	Z *= v.Z;
	W *= v.W;
	return *this;
}

__device__ inline Vec4& Vec4::operator/=(const Vec4 &v)
{
	X /= v.X;
	Y /= v.Y;
	Z /= v.Z;
	W /= v.W;
	return *this;
}

__device__ inline Vec4& Vec4::operator-=(const Vec4& v)
{
	X -= v.X;
	Y -= v.Y;
	Z -= v.Z;
	W -= v.W;
	return *this;
}

__device__ inline Vec4& Vec4::operator*=(const float t)
{
	X *= t;
	Y *= t;
	Z *= t;
	W *= t;
	return *this;
}

__device__ inline Vec4& Vec4::operator/=(const float t)
{
	float k = 1.0f / t;

	X *= k;
	Y *= k;
	Z *= k;
	W *= k;
	return *this;
}

__device__ inline Vec4 Normalize(Vec4 v)
{
	return v / v.length();
}

#pragma endregion
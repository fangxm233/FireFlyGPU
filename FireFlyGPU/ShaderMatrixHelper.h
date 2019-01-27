#pragma once
#include <math.h>
#include "ShaderMatrix.h"
#include "Vector.h"

Vector3 CameraPosition, CameraRotation;
Vector3 ObjectPosition, ObjectRotation;
float FOV, Aspect, Near, Far, Size;
//float Width, Height;

Matrix4x4 GetRotationX(float a)
{
	float c = cos(a);
	float s = sin(a);
	return Matrix4x4(1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1);
}
Matrix4x4 GetRotationY(float a)
{
	float c = cos(a);
	float s = sin(a);
	return Matrix4x4(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1);
}
Matrix4x4 GetRotationZ(float a)
{
	float c = cos(a);
	float s = sin(a);
	return Matrix4x4(c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
}
Matrix4x4 GetRotationMatrix(Vector3 v)
{
	return GetRotationY(v.Y) * GetRotationX(v.X) * GetRotationZ(v.Z);
}
Matrix4x4 PerspectiveFieldOfView(float fov, float aspect, float _near, float _far)
{
	return Matrix4x4(1 / (tan(fov * 0.5f) * aspect), 0, 0, 0, 0, 1 / tan(fov * 0.5f),
		0, 0, 0, 0,  _far / (_far - _near), 1, 0, 0, (_near * _far) / (_near - _far), 0);
}
Matrix4x4 GetOrthographic(float size, float aspect, float _near, float _far)
{
	return Matrix4x4(1 / (aspect * size), 0, 0, 0, 0, 1 / size, 0, 0,
		0, 0, -2 / (_far - _near), -(_far + _near) / (_far - _near), 0, 0, 0, 1);
}
Matrix4x4 CreateTranslation(Vector3 position)
{
	Matrix4x4 result;

	result.M11 = 1.0f;
	result.M12 = 0.0f;
	result.M13 = 0.0f;
	result.M41 = 0.0f;
	result.M21 = 0.0f;
	result.M22 = 1.0f;
	result.M23 = 0.0f;
	result.M42 = 0.0f;
	result.M31 = 0.0f;
	result.M32 = 0.0f;
	result.M33 = 1.0f;
	result.M43 = 0.0f;

	result.M14 = position.X;
	result.M24 = position.Y;
	result.M34 = position.Z;
	result.M44 = 1.0f;

	return result;
}

void CalculateMatrixs()
{
	Object2World = CreateTranslation(ObjectPosition) * GetRotationMatrix(ObjectRotation);
	World2Object = Matrix4x4::invert(Object2World);
	V = CreateTranslation(CameraPosition) * GetRotationMatrix(CameraRotation);
	P = PerspectiveFieldOfView(FOV, Aspect, Near, Far);
	O = GetOrthographic(Size, Aspect, Near, Far);
	MV = V * Object2World;
	T_MV = Matrix4x4::transpose(MV);
	IT_MV = Matrix4x4::invert(T_MV);
	VP = P * V;
	MVP = P * (V * Object2World);
}

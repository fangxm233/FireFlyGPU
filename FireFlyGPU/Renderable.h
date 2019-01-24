#pragma once
#include <string>
#include "Vector.h"
#include "Structure.h"
#include "Matrix.h"

using std::string;

class Camera
{
public:
	Vector3 Position, Rotation;
};

class Object
{
public:
	string name;
	Vector3 position, rotation;
	Mesh mesh;
	Material material;

private:
	Matrix4x4 _matrix4x4;
};
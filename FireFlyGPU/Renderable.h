#pragma once
#include "Structure.h"
#include <string>
#include "Matrix.h"
#include "Vector.h"

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
	Mesh* mesh;
	Material* material;

	Object() {}
	Object(string name, Vector3 position, Vector3 rotation, Mesh* mesh, Material* material) 
	{
		this->name = name;
		this->position = position;
		this->rotation = rotation;
		this->mesh = mesh;
		this->material = material;
	}
private:
	Matrix4x4 _matrix4x4;
};
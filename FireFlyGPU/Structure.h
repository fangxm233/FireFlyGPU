#pragma once
#include <string>
#include "Vector.h"

using std::string;

class Material
{
public:
	string name;
	string shaderName;
	void* shader;
};

struct Vertex
{
public:
	Vector2 UV;
	Vector3 Point, Normal, Tangent, Bitangent;
	Vector4 Color;
};

using std::string;

class Mesh
{
public:
	string name;
	Vertex * vertices;
	int vertexCount;
	int * triangles;
	int triangleCount;
};
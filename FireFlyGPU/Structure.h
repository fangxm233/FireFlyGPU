#pragma once
#include <cuda_runtime.h>
#include <string>
#include <map>
#include "Vector.h"

using std::string;

class Material
{
public:
	string& name;
	string& shaderName;
	void* shader;
};

struct Vertex
{
public:
	Vector2 uv;
	Vector3 point, normal, tangent, bitangent;
	Vector4 color;

	__device__ __host__ inline Vertex operator*=(const float &v)
	{
		uv *= v;
		point *= v;
		normal *= v;
		tangent *= v;
		bitangent *= v;
		color *= v;
		return *this;
	}
};

class Mesh
{
public:
	string& name;
	Vertex* vertices;
	int vertexCount;
	int* triangles;
	int triangleCount;

	Vertex& GetVertex(int i)
	{
		return vertices[triangles[i]];
	}
};

//Ϊ�˽��ѭ���������Ķ���
class Camera;
class Object;

class Scene
{
public:
	string& name;
	Camera& camera; //Ϊ�˽��ѭ������ֻ��������
	Vector4 ambientColor;
	std::map<string, Object> objects;
};
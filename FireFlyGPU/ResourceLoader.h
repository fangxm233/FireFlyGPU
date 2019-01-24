#pragma once
#include <string>
#include <fstream>
#include "root.h"
#include "Vector.h"
#include "Structure.h"

typedef unsigned char CSbyte;

using namespace std;

/*
 * 模型文件格式
 * unsigned char 233 魔数
 * int 顶点数目
 * //位置
 * float X
 * float Y
 * float Z
 * //法线
 * float X
 * float Y
 * float Z
 * //UV
 * float U
 * float V
 * ...
 * int 三角形个数
 * int 顶点编号
 * ...
 */

Vertex ReadVertex(ifstream * readStream)
{
	Vertex vertex;
	readStream->read((char*)&vertex.Point.X, sizeof(float));
	readStream->read((char*)&vertex.Point.Y, sizeof(float));
	readStream->read((char*)&vertex.Point.Z, sizeof(float));
	readStream->read((char*)&vertex.Normal.X, sizeof(float));
	readStream->read((char*)&vertex.Normal.Y, sizeof(float));
	readStream->read((char*)&vertex.Normal.Z, sizeof(float));
	readStream->read((char*)&vertex.UV.X, sizeof(float));
	readStream->read((char*)&vertex.UV.Y, sizeof(float));
	return vertex;
}

Mesh * LoadMesh(string name)
{
	ifstream readStream(ModelPath + name + ModelExtension, ios::binary);
	if (!readStream.good())return nullptr;
	CSbyte m;
	readStream >> m;
	if (m != CSbyte(233))
	{
		printf("模型魔数不正确!");
		return nullptr;
	}
	int verCount;
	readStream.read((char *)&verCount, sizeof(int));
	Vertex * vertices = new Vertex[verCount];
	for (int i = 0; i < verCount; i++)
		vertices[i] = ReadVertex(&readStream);
	int triangleCount;
	readStream.read((char *)&triangleCount, sizeof(int));
	int * triangles = new int[triangleCount * 3];
	for (int i = 0; i < triangleCount * 3; i++)
		readStream.read((char *)(triangles + i), sizeof(int));

	return new Mesh{ name, vertices, verCount, triangles, triangleCount };
}
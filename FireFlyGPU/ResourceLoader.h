#pragma once
#include <string>
#include <fstream>
#include "root.h"
#include "Vector.h"
#include "Structure.h"

typedef unsigned char CSbyte;

using namespace std;

/*
 * ģ���ļ���ʽ
 * unsigned char 233 ħ��
 * int ������Ŀ
 * //λ��
 * float X
 * float Y
 * float Z
 * //����
 * float X
 * float Y
 * float Z
 * //UV
 * float U
 * float V
 * ...
 * int �����θ���
 * int ������
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
		printf("ģ��ħ������ȷ!");
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
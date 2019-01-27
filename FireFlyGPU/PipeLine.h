#pragma once
#include "Structure.h"
#include <GL\glew.h>

extern "C" void d_handleVerteices(Vertex* vertices, int vertexCount);
extern "C" void d_init(int width, int height, unsigned char* h_buffer, unsigned char* d_buffer);;
extern "C" void d_drawTriangles(Mesh* mesh);

class pipeLine
{
private:
	GLint bufferSize;
	GLbyte* buffer;

public:
	void init(int width, int height, GLbyte* h_buffer, unsigned char* d_buffer, GLint bufferSize);
	void handleVerteices(Vertex* vertices, int vertexCount);
	void drawTriangles(Mesh* mesh);
};

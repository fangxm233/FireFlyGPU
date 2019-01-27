#include "PipeLine.h"
#include <assert.h>

void pipeLine::init(int width, int height, GLbyte* h_buffer, unsigned char* d_buffer, GLint bufferSize)
{
	pipeLine::buffer = buffer;
	pipeLine::bufferSize = bufferSize;
	d_init(width, height, (unsigned char*)h_buffer, d_buffer);
}

void pipeLine::handleVerteices(Vertex* vertices, int vertexCount)
{
	d_handleVerteices(vertices, vertexCount);
}

void pipeLine::drawTriangles(Mesh * mesh)
{
	d_drawTriangles(mesh);
}
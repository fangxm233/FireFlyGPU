#pragma once
#include "Renderer.h"
#include "Vector.h"
#include "PipeLine.h"
#include "ShaderMatrixHelper.h"

void renderer::initRenderer(GLbyte* _h_buffer, unsigned char* _d_buffer, GLint _bufferSize)
{
	h_buffer = _h_buffer;
	d_buffer = _d_buffer;
	bufferSize = _bufferSize;
	//initDeviceMemory();
}

void renderer::loadScene(string sceneName)
{

}

void renderer::initDeviceMemory()
{
	cudaMallocManaged(reinterpret_cast<void**>(&object), sizeof(object));
	cudaMallocManaged(reinterpret_cast<void**>(&object->mesh->vertices), object->mesh->vertexCount * sizeof(Vertex));
	cudaMallocManaged(reinterpret_cast<void**>(&object->mesh->triangles), object->mesh->triangleCount * sizeof(int));
}

void renderer::renderFrame()
{
	Vertex* d_vertices;
	int* d_triangles;

	cudaMalloc(reinterpret_cast<void**>(&d_vertices), object->mesh->vertexCount * sizeof(Vertex));
	cudaMalloc(reinterpret_cast<void**>(&d_triangles), object->mesh->triangleCount * sizeof(int));

	cudaMemcpy(d_vertices, object->mesh->vertices, object->mesh->vertexCount * sizeof(Vertex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangles, object->mesh->triangles, object->mesh->triangleCount * sizeof(int), cudaMemcpyHostToDevice);

	CameraPosition = Vector3(0, 0, 0);
	CameraRotation = Vector3(0, 0, 0);
	ObjectPosition = object->position;
	ObjectRotation = object->rotation;
	Far = 500;
	Near = 0.5f;
	FOV = Pi / 2;
	Aspect = Width / Height;
	//Width = Width;
	//Height = Height;
	Size = 1;
	CalculateMatrixs();

	setMatrixs(MVP, MV, V, P, VP, T_MV, IT_MV, Object2World, World2Object, O);

	pipeLine p = pipeLine();
	p.init(Width, Height, h_buffer, d_buffer, bufferSize);
	p.handleVerteices(d_vertices, object->mesh->vertexCount);
	p.drawTriangles(object->mesh);

	//for (int i = 0; i + 2 < object->mesh.triangleCount; i += 3)
	//	if (!backFaceCulling(Object2World * object->mesh.GetVertex(i).point,
	//		Object2World * object->mesh.GetVertex(i + 1).point,
	//		Object2World * object->mesh.GetVertex(i + 2).point))
	//		pipeLine::drawTriangle(object->mesh.GetVertex(i),
	//			object->mesh.GetVertex(i + 1),
	//			object->mesh.GetVertex(i + 2));
}

bool renderer::backFaceCulling(Vector3 p1, Vector3 p2, Vector3 p3)
{
	Vector3 v1 = p2 - p1;
	Vector3 v2 = p3 - p2;
	Vector3 normal = cross(v1, v2);
	Vector3 view_dir = p1 - Vector3(0, 0, 0);
	return dot(normal, view_dir) > 0;
}

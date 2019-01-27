#pragma once
#include "Renderable.h"
#include "GlobalSetting.h"
#include <GL\glew.h>

extern "C" void setMatrixs(Matrix4x4 _MVP,
	Matrix4x4 _MV,
	Matrix4x4 _V,
	Matrix4x4 _P,
	Matrix4x4 _VP,
	Matrix4x4 _T_MV,
	Matrix4x4 _IT_MV,
	Matrix4x4 _Object2World,
	Matrix4x4 _World2Object,
	Matrix4x4 _O);

class renderer
{
private:
	GLint bufferSize;
	GLbyte* h_buffer;
	unsigned char* d_buffer;
	bool backFaceCulling(Vector3 p1, Vector3 p2, Vector3 p3);
	void initDeviceMemory();

public:	
	Object* object;
	void initRenderer(GLbyte* _h_buffer, unsigned char* d_buffer, GLint _bufferSize);
	void loadScene(string sceneName);
	void renderFrame();
};

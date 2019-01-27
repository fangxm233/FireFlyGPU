#pragma once
#include "device_launch_parameters.h"
#include "Structure.h"

// ģ��*�۲�*ͶӰ�������ڽ�����/����ʸ����ģ�Ϳռ�任�����пռ�
__device__ Matrix4x4 MVP;
// ģ��*�۲�������ڽ�����/����ʸ����ģ�Ϳռ�任���۲�ռ�
__device__ Matrix4x4 MV;
// �۲�������ڽ�����/����ʸ��������ռ�任���۲�ռ�
__device__ Matrix4x4 V;
// ͸��ͶӰ�������ڽ�����/����ʸ���ӹ۲�ռ�任�����пռ�
__device__ Matrix4x4 P;
// �۲�*ͶӰ���� ���ڽ�����/����ʸ��������ռ�任�����пռ�
__device__ Matrix4x4 VP;
// MV��ת�þ���
__device__ Matrix4x4 T_MV;
// MV����ת�þ������ڽ����ߴ�ģ�Ϳռ�任���۲�ռ�
__device__ Matrix4x4 IT_MV;
// ģ�;������ڽ�����/���ߴ�ģ�Ϳռ�任������ռ�
__device__ Matrix4x4 Object2World;
// Entity2World����������ڽ�����/���ߴ�����ռ�任��ģ�Ϳռ�
__device__ Matrix4x4 World2Object;
// ����ͶӰ�������ڽ�����/����ʸ���ӹ۲�ռ�任�����пռ�
__device__ Matrix4x4 O;

//��װ�Ƕ�����ɫ�����
struct Vertex4
{
public:
	Vector2 uv;
	Vector3 normal, tangent, bitangent;
	Vector4 point, color;
};

__device__ __host__ inline Vertex4 operator*(const float &v, const Vertex4 &v1)
{
	Vertex4 v4;
	v4.uv *= v * v1.uv;
	v4.point = v * v1.point;
	v4.normal = v * v1.normal;
	v4.tangent = v * v1.tangent;
	v4.bitangent = v * v1.bitangent;
	v4.color = v * v1.color;
	return v4;
}
__device__ __host__ inline Vertex4 operator+(const Vertex4 &v1, const Vertex4 &v2)
{
	Vertex4 v4;
	v4.uv = v1.uv + v2.uv;
	v4.point = v1.point + v2.point;
	v4.normal = v1.normal + v2.normal;
	v4.tangent = v1.tangent + v2.tangent;
	v4.bitangent = v1.bitangent + v2.bitangent;
	v4.color = v1.color + v2.color;
	return v4;
}

Vertex4* d_vertices;
Vertex4* h_vertices;
int width;
int height;
unsigned char* h_buffer;
unsigned char* d_buffer;
extern "C" void d_init(int width, int height, unsigned char* h_buffer, unsigned char* d_buffer);
extern "C" void d_handleVerteices(Vertex* vertices, int vertexCount);
extern "C"  void setMatrixs(Matrix4x4 _MVP, 
	Matrix4x4 _MV, 
	Matrix4x4 _V, 
	Matrix4x4 _P, 
	Matrix4x4 _VP, 
	Matrix4x4 _T_MV, 
	Matrix4x4 _IT_MV, 
	Matrix4x4 _Object2World, 
	Matrix4x4 _World2Object, 
	Matrix4x4 _O);
__global__ void handleVertices_g(Vertex* vertices, Vertex4* vertices4, int vertexCount);
__device__ Vertex4 vertexShader(Vertex v);
__device__ inline Vertex4 toScreen(Vertex4 v);
__device__ inline Vertex4 mulOnePerZ(Vertex4 v);
extern "C" void d_drawTriangles(Mesh* mesh);
bool backFaceCulling(Vector3 p1, Vector3 p2, Vector3 p3);
inline void calAABB(Vertex4& v1, Vertex4& v2, Vertex4& v3, Vector2& bboxmin, Vector2& bboxmax);
__global__ void drawTriangle_g(Vertex4* vertices, int i1, int i2, int i3, Vector2 bboxmin, Vector2 bboxmax, unsigned char* buffer, int width);
__device__ inline bool BarycentricFast(Vector4 a, Vector4 b, Vector4 c, Vector4 p, Vector3& result);
__device__ Vector4 fragmentShader(Vertex4 v);

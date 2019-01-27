#include "PipeLine.cuh"
#include <cuda_runtime.h>
#include "GlobalSetting.h"
#include <helper_math.h>

#define printError(){auto error = cudaGetLastError();if (error != 0)printf("error:%d\n", error);}

extern "C"
void d_init(int _width, int _height, unsigned char* _h_buffer, unsigned char* _d_buffer)
{
	width = _width;
	height = _height;
	h_buffer = _h_buffer;
	d_buffer = _d_buffer;
}

void d_handleVerteices(Vertex* vertices, int vertexCount)
{
	cudaMalloc(reinterpret_cast<void**>(&d_vertices), sizeof(Vertex4) * vertexCount);
	
	//hard code
	dim3 blockSize(256);
	dim3 gridSize((vertexCount + blockSize.x - 1) / blockSize.x); //只准多不准少

	handleVertices_g << <gridSize, blockSize >> > (vertices, d_vertices, vertexCount);
	
	printError();
}

extern "C" void setMatrixs(Matrix4x4 _MVP,
	Matrix4x4 _MV,
	Matrix4x4 _V,
	Matrix4x4 _P,
	Matrix4x4 _VP,
	Matrix4x4 _T_MV,
	Matrix4x4 _IT_MV,
	Matrix4x4 _Object2World,
	Matrix4x4 _World2Object,
	Matrix4x4 _O)
{
	//不理这里的10个错误
	cudaMemcpyToSymbol(MVP, &_MVP, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(MV, &_MV, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(V, &_V, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(P, &_P, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(VP, &_VP, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(T_MV, &_T_MV, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(IT_MV, &_IT_MV, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(Object2World, &_Object2World, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(World2Object, &_World2Object, sizeof(Matrix4x4));
	cudaMemcpyToSymbol(O, &_O, sizeof(Matrix4x4));
}

__global__ void handleVertices_g(Vertex* vertices, Vertex4* vertices4, int vertexCount)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index > vertexCount)return;

	Vertex v = vertices[index];
	Vertex4 vo = vertexShader(v);
	vo = toScreen(vo);
	vo = mulOnePerZ(vo);
	vertices4[index] = vo;
	//printf("index:%d \nsrc:\npoint X:%f\npoint Y:%f\npoint Z:%f\nresult:\npoint X:%f\npoint Y:%f\npoint Z:%f\n\n", 
	//	index, v.point.X, v.point.Y, v.point.Z, vo.point.X, vo.point.Y, vo.point.Z);
	return;
}

__device__ Vertex4 vertexShader(Vertex v)
{
	return Vertex4{
		v.uv,
		v.normal,
		v.tangent,
		v.bitangent,
		Vector4(v.point.X , v.point.Y, v.point.Z, 1) * MVP,
		v.color
	};
}
__device__ Vertex4 toScreen(Vertex4 v)
{
	v.point = Vector4((int)(v.point.X * Width / (2 * v.point.W) + Width / 2),
		(int)(v.point.Y * Height / (2 * v.point.W) + Height / 2), v.point.Z, v.point.W);
	return v;
}
__device__ Vertex4 mulOnePerZ(Vertex4 v)
{
	v.point.W = 1 / v.point.W;
	Vector4 point = v.point;
	v = v.point.W * v;
	v.point = point;
	return v;
}

void d_drawTriangles(Mesh* mesh)
{
	h_vertices = new Vertex4[mesh->vertexCount];
	cudaMemcpy(h_vertices, d_vertices, sizeof(Vertex4) * mesh->vertexCount, cudaMemcpyDeviceToHost);

	for (int i = 0; i + 2 < mesh->triangleCount; i += 3)
		if (!backFaceCulling(Object2World * mesh->GetVertex(i).point,
			Object2World * mesh->GetVertex(i + 1).point,
			Object2World * mesh->GetVertex(i + 2).point))
		{
			Vertex4& v1 = h_vertices[mesh->triangles[i]];
			Vertex4& v2 = h_vertices[mesh->triangles[i + 1]];
			Vertex4& v3 = h_vertices[mesh->triangles[i + 2]];
			Vector2 bboxmin = Vector2();
			Vector2 bboxmax = Vector2();
			calAABB(v1, v2, v3, bboxmin, bboxmax);
			//hard code
			dim3 grid((int)(bboxmax.X - bboxmin.X + 31) / 32, (int)(bboxmax.Y - bboxmin.Y + 31) / 32), block(32, 32);
			drawTriangle_g << <grid, block >> > (d_vertices, mesh->triangles[i], mesh->triangles[i + 1], mesh->triangles[i + 2], bboxmin, bboxmax, d_buffer, width);
			printError();
		}
	cudaMemcpy(h_buffer, d_buffer, sizeof(unsigned char) * width * height * 4, cudaMemcpyDeviceToHost);
}

bool backFaceCulling(Vector3 p1, Vector3 p2, Vector3 p3)
{
	//Vector3 p1 = Vector3(pp1.X, pp1.Y, pp1.Z);
	//Vector3 p2 = Vector3(pp2.X, pp2.Y, pp2.Z);
	//Vector3 p3 = Vector3(pp3.X, pp3.Y, pp3.Z);

	Vector3 v1 = p2 - p1;
	Vector3 v2 = p3 - p2;
	Vector3 normal = cross(v1, v2);
	Vector3 view_dir = p1 - Vector3(0, 0, 0);
	return dot(normal, view_dir) > 0;
}

void calAABB(Vertex4& v1, Vertex4& v2, Vertex4& v3, Vector2& bboxmin, Vector2& bboxmax)
{
	bboxmin = Vector2(INFINITY, INFINITY);
	bboxmax = Vector2(-INFINITY, -INFINITY);
	Vector2 clamp = Vector2(Width - 1, Height - 1);

	bboxmin.X = max(0.0f, min(bboxmin.X, v1.point.X));
	bboxmin.Y = max(0.0f, min(bboxmin.Y, v1.point.Y));

	bboxmax.X = min(clamp.X, max(bboxmax.X, v1.point.X));
	bboxmax.Y = min(clamp.Y, max(bboxmax.Y, v1.point.Y));

	bboxmin.X = max(0.0f, min(bboxmin.X, v2.point.X));
	bboxmin.Y = max(0.0f, min(bboxmin.Y, v2.point.Y));

	bboxmax.X = min(clamp.X, max(bboxmax.X, v2.point.X));
	bboxmax.Y = min(clamp.Y, max(bboxmax.Y, v2.point.Y));

	bboxmin.X = max(0.0f, min(bboxmin.X, v3.point.X));
	bboxmin.Y = max(0.0f, min(bboxmin.Y, v3.point.Y));

	bboxmax.X = min(clamp.X, max(bboxmax.X, v3.point.X));
	bboxmax.Y = min(clamp.Y, max(bboxmax.Y, v3.point.Y));
}

//__global__ void calAABB_g(Vertex4* d_vertices, int* triangles, int triangleCount)
//{
//	Vertex4 v1 = d_vertices[threadIdx.x];
//	Vertex4 v2 = d_vertices[threadIdx.x + 1];
//	Vertex4 v3 = d_vertices[threadIdx.x + 2];
//
//	Vector2 bboxmin = Vector2(INFINITY, INFINITY);
//	Vector2 bboxmax = Vector2(-INFINITY, -INFINITY);
//	Vector2 clamp = Vector2(Width - 1, Height - 1);
//
//	bboxmin.X = max(0.0f, min(bboxmin.X, v1.point.X));
//	bboxmin.Y = max(0.0f, min(bboxmin.Y, v1.point.Y));
//
//	bboxmax.X = min(clamp.X, max(bboxmax.X, v1.point.X));
//	bboxmax.Y = min(clamp.Y, max(bboxmax.Y, v1.point.Y));
//
//	bboxmin.X = max(0.0f, min(bboxmin.X, v2.point.X));
//	bboxmin.Y = max(0.0f, min(bboxmin.Y, v2.point.Y));
//
//	bboxmax.X = min(clamp.X, max(bboxmax.X, v2.point.X));
//	bboxmax.Y = min(clamp.Y, max(bboxmax.Y, v2.point.Y));
//
//	bboxmin.X = max(0.0f, min(bboxmin.X, v3.point.X));
//	bboxmin.Y = max(0.0f, min(bboxmin.Y, v3.point.Y));
//
//	bboxmax.X = min(clamp.X, max(bboxmax.X, v3.point.X));
//	bboxmax.Y = min(clamp.Y, max(bboxmax.Y, v3.point.Y));
//}

__global__ void drawTriangle_g(Vertex4* vertices, int i1, int i2, int i3, Vector2 bboxmin, Vector2 bboxmax, unsigned char* buffer, int width)
{
	Vertex4 v1 = vertices[i1];
	Vertex4 v2 = vertices[i2];
	Vertex4 v3 = vertices[i3];

	//hard code
	const int x = blockIdx.x * 32 + threadIdx.x + (int)bboxmin.X;
	const int y = blockIdx.y * 32 + threadIdx.y + (int)bboxmin.Y;
	if (x > bboxmax.X || y > bboxmax.Y)return;

	//if (x != 256 || y != 256)return;
	//printf("hey!!!\n");

	Vector4 currentPoint = Vector4(x, y, 0, 0);
	Vector3 barycentricCoord = Vector3();
	bool isInline = BarycentricFast(v1.point, v2.point, v3.point, currentPoint, barycentricCoord);

	if (isInline)
		return;

	float threshold = -0.000001f;
	if (barycentricCoord.X < threshold || barycentricCoord.Y < threshold || barycentricCoord.Z < threshold) return;

	float fInvW = 1.0f / (barycentricCoord.X * v1.point.W + barycentricCoord.Y * v2.point.W + barycentricCoord.Z * v3.point.W);
	Vertex4 input = fInvW * (barycentricCoord.X * v1 + barycentricCoord.Y * v2 + barycentricCoord.Z * v3);

	Vector4 result = fragmentShader(input);
	const auto i = width * 4 * y + x * 4;
	buffer[i] = result.X * 255;
	buffer[i + 1] = result.Y * 255;
	buffer[i + 2] = result.Z * 255;
	buffer[i + 3] = result.W * 255;
	//printf("R:%f G:%f B:%f A:%f\n", result.X, result.Y, result.Z, result.W);
	//SetPixel((int)X, (int)Y, Tex.Value(uv * fInvW)/*col*/);
}

__device__ bool BarycentricFast(Vector4 a, Vector4 b, Vector4 c, Vector4 p, Vector3& result)
{
	Vector4 v0 = b - a, v1 = c - a, v2 = p - a;

	float d00 = v0.X * v0.X + v0.Y * v0.Y;
	float d01 = v0.X * v1.X + v0.Y * v1.Y;
	float d11 = v1.X * v1.X + v1.Y * v1.Y;
	float d20 = v2.X * v0.X + v2.Y * v0.Y;
	float d21 = v2.X * v1.X + v2.Y * v1.Y;

	float denom = d00 * d11 - d01 * d01;
	//三角形变成了一条线
	if (abs(denom) < 0.000001)
		return true;

	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0f - v - w;

	result = Vector3(u, v, w);
	return false;
}

__device__ Vector4 fragmentShader(Vertex4 v)
{
	return Vector4(1, 1, 1, 1);
}
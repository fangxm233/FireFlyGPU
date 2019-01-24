#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdlib.h"
#include <thread>
#include <helper_cuda.h>
#include "curand_kernel.h"

#include "root.h"
#include "GLWindow.h"
#include "ResourceLoader.h"
#include "Structure.h"

using namespace std;

int FpsCount;
const int BlockSize = 32;

void InitData()
{
	PixelLength = Height * Width * 4;
	Buffer = new GLbyte[PixelLength];
	for (auto i = 0; i < PixelLength; i++)
		Buffer[i] = GLbyte(0);
}

struct BasicData
{
	int Width;
	int Height;
};

__global__ 
void Render(int width, int height, int blocksize, byte * buffer)
{
	const int x = blockIdx.x * blocksize + threadIdx.x;
	const int y = blockIdx.y * blocksize + threadIdx.y;

	const auto i = width * 4 * y + x * 4;
	int color = (x * 1.0f / width) * (y * 1.0f / height) * 255;
	buffer[i] = color;
	buffer[i + 1] = color;
	buffer[i + 2] = color;
	buffer[i + 3] = 255;
}

void RenderFrame()
{
	dim3 grid(Width / BlockSize, Height / BlockSize), block(BlockSize, BlockSize);

	//******分配线程 ******
	Render << <grid, block >> > (Width, Height, BlockSize, reinterpret_cast<byte*>(Buffer));

	//同步以获取正确的结果
	cudaDeviceSynchronize();

	FpsCount++;
}

cudaError_t InitRender()
{
	// Cuda Status for checking error
	const auto cuda_status = cudaSetDevice(0);

	BasicData renderData = { Width, Height };// Split area, 32*32/block

	cudaMallocManaged(reinterpret_cast<void**>(&renderData), sizeof(BasicData));
	cudaMallocManaged(reinterpret_cast<void**>(&Buffer), Width * Height * 4 * sizeof(byte));

	return cuda_status;
}

void RefreshFps(int value)
{
	char fps[256];
	sprintf(fps, "FireFlyGPU FPS: %d", FpsCount);
	glutSetWindowTitle(fps);
}

void PrintDeviceInfo() 
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) 
	{
		printf("没有支持CUDA的设备!\n");
		return;
	}

	for (int dev = 0; dev < deviceCount; dev++)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("设备 %d: \"%s\"\n", dev, deviceProp.name);

		char msg[256];
		sprintf_s(msg, sizeof(msg),
			"global memory大小:        %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
		printf("%s", msg);

		printf("SM数:                    %2d \n每SMCUDA核心数:           %3d \n总CUDA核心数:             %d \n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount);

		printf("静态内存大小:             %zu bytes\n",
			deviceProp.totalConstMem);
		printf("每block共享内存大小:      %zu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("每block寄存器数:          %d\n",
			deviceProp.regsPerBlock);
		printf("线程束大小:               %d\n",
			deviceProp.warpSize);
		printf("每处理器最大线程数:       %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("每block最大线程数:        %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("线程块最大维度大小        (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("网格最大维度大小          (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);

		printf("\n");
	}
	printf("设备信息打印完毕************************\n\n");
}

int main(int argc, char* argv[])
{
	PrintDeviceInfo();
	InitData();
	InitRender();
	Mesh * mesh = LoadMesh("cube");
	//InitWindow(argc, argv, GLUT_DOUBLE | GLUT_RGBA, 100, 100, Width, Height, "FireFlyGPU");
	return 0;
}



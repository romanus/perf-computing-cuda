#include "main.cuh"

using float32_t = float;

constexpr auto KERNEL_RADIUS = 2;
constexpr auto KERNEL_W = (2 * KERNEL_RADIUS + 1); // 5 x 5 kernel

__device__ __constant__ float32_t d_Kernel[KERNEL_W];

constexpr auto TILE_W = 16;
constexpr auto TILE_H = 16;

#define IMUL(a,b) __mul24(a,b)

__global__ void convolutionRowGPU(float32_t* d_Result, const float32_t* d_Data, int dataW, int dataH)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ float32_t data[3 * TILE_H * (TILE_W + KERNEL_RADIUS * 2)]; // 3 channels of TILE_H rows and (TILE_W + KERNEL_RADIUS * 2) columns

	// original image based coordinate
	const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
	const int shift = threadIdx.y * (TILE_W + KERNEL_RADIUS * 2);

	const int gLoc = threadIdx.x +
		IMUL(blockIdx.x, blockDim.x) +
		IMUL(threadIdx.y, dataW) +
		IMUL(blockIdx.y, blockDim.y) * dataW;

	// load cache (32x16 shared memory, 16x16 threads blocks)
	// each threads loads two values from global memory into shared mem
	// if in image area, get value in global mem, else 0
	int x;		// image based coordinate

	// case1: left
	x = x0 - KERNEL_RADIUS;
	if (x < 0)
	{
		data[3 * (threadIdx.x + shift) + 0] = 0;
		data[3 * (threadIdx.x + shift) + 1] = 0;
		data[3 * (threadIdx.x + shift) + 2] = 0;
	}
	else
	{
		data[3 * (threadIdx.x + shift) + 0] = d_Data[3 * (gLoc - KERNEL_RADIUS) + 0];
		data[3 * (threadIdx.x + shift) + 1] = d_Data[3 * (gLoc - KERNEL_RADIUS) + 1];
		data[3 * (threadIdx.x + shift) + 2] = d_Data[3 * (gLoc - KERNEL_RADIUS) + 2];
	}

	// case2: right
	x = x0 + KERNEL_RADIUS;
	if (x > dataW - 1)
	{
		data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 0] = 0;
		data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 1] = 0;
		data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 2] = 0;
	}
	else
	{
		data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 0] = d_Data[3 * (gLoc + KERNEL_RADIUS) + 0];
		data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 1] = d_Data[3 * (gLoc + KERNEL_RADIUS) + 1];
		data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 2] = d_Data[3 * (gLoc + KERNEL_RADIUS) + 2];
	}

	data[3 * (threadIdx.x + KERNEL_RADIUS + shift) + 0] = d_Data[3 * gLoc + 0];
	data[3 * (threadIdx.x + KERNEL_RADIUS + shift) + 1] = d_Data[3 * gLoc + 1];
	data[3 * (threadIdx.x + KERNEL_RADIUS + shift) + 2] = d_Data[3 * gLoc + 2];

	__syncthreads();

	// convolution
	float32_t sum[] = { 0, 0, 0 };
	x = threadIdx.x + KERNEL_RADIUS;
	for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
	{
		sum[0] += data[3 * (x + i + shift) + 0] * d_Kernel[i + KERNEL_RADIUS];
		sum[1] += data[3 * (x + i + shift) + 1] * d_Kernel[i + KERNEL_RADIUS];
		sum[2] += data[3 * (x + i + shift) + 2] * d_Kernel[i + KERNEL_RADIUS];
	}

	d_Result[3 * gLoc + 0] = sum[0];
	d_Result[3 * gLoc + 1] = sum[1];
	d_Result[3 * gLoc + 2] = sum[2];
}

__global__ void convolutionColGPU(float32_t* d_Result, const float32_t* d_Data, int dataW, int dataH)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ float32_t data[3 * TILE_W * (TILE_H + KERNEL_RADIUS * 2)]; // 3 channels of (TILE_H + KERNEL_RADIUS * 2) rows and TILE_W columns

	// original image based coordinate
	const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
	const int shift = threadIdx.y * TILE_W;

	// global mem address of this thread
	const int gLoc = threadIdx.x +
		IMUL(blockIdx.x, blockDim.x) +
		IMUL(threadIdx.y, dataW) +
		IMUL(blockIdx.y, blockDim.y) * dataW;

	// load cache (32x16 shared memory, 16x16 threads blocks)
	// each threads loads two values from global memory into shared mem
	// if in image area, get value in global mem, else 0
	int y;		// image based coordinate

	// case1: upper
	y = y0 - KERNEL_RADIUS;
	if (y < 0)
	{
		data[3 * (threadIdx.x + shift) + 0] = 0;
		data[3 * (threadIdx.x + shift) + 1] = 0;
		data[3 * (threadIdx.x + shift) + 2] = 0;
	}
	else
	{
		data[3 * (threadIdx.x + shift) + 0] = d_Data[3 * (gLoc - IMUL(dataW, KERNEL_RADIUS)) + 0];
		data[3 * (threadIdx.x + shift) + 1] = d_Data[3 * (gLoc - IMUL(dataW, KERNEL_RADIUS)) + 1];
		data[3 * (threadIdx.x + shift) + 2] = d_Data[3 * (gLoc - IMUL(dataW, KERNEL_RADIUS)) + 2];
	}

	// case2: lower
	y = y0 + KERNEL_RADIUS;
	const auto shift1 = shift + IMUL(2 * KERNEL_RADIUS, TILE_W);
	if (y > dataH - 1)
	{
		data[3 * (threadIdx.x + shift1) + 0] = 0;
		data[3 * (threadIdx.x + shift1) + 1] = 0;
		data[3 * (threadIdx.x + shift1) + 2] = 0;
	}
	else
	{
		data[3 * (threadIdx.x + shift1) + 0] = d_Data[3 * (gLoc + IMUL(dataW, KERNEL_RADIUS)) + 0];
		data[3 * (threadIdx.x + shift1) + 1] = d_Data[3 * (gLoc + IMUL(dataW, KERNEL_RADIUS)) + 1];
		data[3 * (threadIdx.x + shift1) + 2] = d_Data[3 * (gLoc + IMUL(dataW, KERNEL_RADIUS)) + 2];
	}

	data[3 * (threadIdx.x + shift + IMUL(TILE_W, KERNEL_RADIUS)) + 0] = d_Data[3 * gLoc + 0];
	data[3 * (threadIdx.x + shift + IMUL(TILE_W, KERNEL_RADIUS)) + 1] = d_Data[3 * gLoc + 1];
	data[3 * (threadIdx.x + shift + IMUL(TILE_W, KERNEL_RADIUS)) + 2] = d_Data[3 * gLoc + 2];

	__syncthreads();

	// convolution
	float32_t sum[] = { 0, 0, 0 };
	for (int i = 0; i <= KERNEL_RADIUS * 2; i++)
	{
		sum[0] += data[3 * (threadIdx.x + (threadIdx.y + i) * TILE_W) + 0] * d_Kernel[i];
		sum[1] += data[3 * (threadIdx.x + (threadIdx.y + i) * TILE_W) + 1] * d_Kernel[i];
		sum[2] += data[3 * (threadIdx.x + (threadIdx.y + i) * TILE_W) + 2] * d_Kernel[i];
	}

	d_Result[3 * gLoc + 0] = sum[0];
	d_Result[3 * gLoc + 1] = sum[1];
	d_Result[3 * gLoc + 2] = sum[2];
}

constexpr auto KERNEL_SIZE = static_cast<int32_t>(KERNEL_W * sizeof(float32_t));

void FilterBenchmark(const cv::Mat& image)
{
	std::cout << "----------- CONVOLUTION ------------\n";

	const auto h_Kernel = cv::getGaussianKernel(KERNEL_W, -1, CV_32F);
	cudaMemcpyToSymbol(d_Kernel, h_Kernel.data, KERNEL_SIZE);

	const auto inputImage = ImageMultiplier::Multiply(image);

	const auto dw = inputImage.cols;
	const auto dh = inputImage.rows;

	auto h_Result = cv::Mat(dh, dw, CV_32FC3);

	const dim3 blocks(TILE_W, TILE_H);
	const dim3 grids(dw / TILE_W, dh / TILE_H); // we assume that image width and height divide by TILE_W/TILE_H

	{
		const auto timeLock = MeasureTime("Computation time+load+unload");

		const DeviceAlloc d_Image(inputImage);

		DeviceAlloc d_Result(d_Image.m_size);
		DeviceAlloc d_Data(d_Image.m_size);

		{
			const auto timeLock = MeasureTime("Computation time+unload");

			{
				const auto timeLock3 = MeasureTime("Computation time");
				convolutionRowGPU<<<grids, blocks>>>((float32_t*)d_Data.m_deviceData, (const float32_t*)d_Image.m_deviceData, dw, dh);
				cudaDeviceSynchronize();
				convolutionColGPU<<<grids, blocks>>>((float32_t*)d_Result.m_deviceData, (const float32_t*)d_Data.m_deviceData, dw, dh);
				cudaDeviceSynchronize();
			}

			d_Result.CopyToHost(h_Result.data);
		}
	}

	// compare the output and OpenCV output
	auto openCV_output = cv::Mat{};
	cv::sepFilter2D(inputImage, openCV_output, CV_32F, h_Kernel, h_Kernel, cv::Point(-1, -1), 0, cv::BorderTypes::BORDER_CONSTANT);

	const auto algoOutputEqual = std::equal(h_Result.datastart, h_Result.dataend, openCV_output.datastart); // I'm really shocked that the output matches perfectly
	std::cout << "CUDA algorithm matches OpenCV: " << std::boolalpha << algoOutputEqual << std::endl;

    std::cout << "------------------------------------\n" << std::endl;
}
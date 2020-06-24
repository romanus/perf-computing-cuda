#include "main.cuh"

using float32_t = float;

constexpr auto KERNEL_RADIUS = 2;
constexpr auto KERNEL_W = (2 * KERNEL_RADIUS + 1); // 5 x 5 kernel

__device__ __constant__ float32_t d_Kernel[KERNEL_W];

constexpr auto TILE_W = 16;		// active cell width
constexpr auto TILE_H = 16;		// active cell height

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
	if (blockIdx.x == 0)
	{
		if (x0 <= KERNEL_RADIUS)
		{
			data[3 * (threadIdx.x + shift) + 0] = 0;
			data[3 * (threadIdx.x + shift) + 1] = 0;
			data[3 * (threadIdx.x + shift) + 2] = 0;
		}
	}

	// case2: right
	if (blockIdx.x == blockDim.x - 1)
	{
		if (x0 >= dataW - KERNEL_RADIUS)
		{
			data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 0] = 0;
			data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 1] = 0;
			data[3 * (threadIdx.x + 2 * KERNEL_RADIUS + shift) + 2] = 0;
		}
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

	d_Result[3 * gLoc + 0] = 32 * sum[0];
	d_Result[3 * gLoc + 1] = 32 * sum[1];
	d_Result[3 * gLoc + 2] = 32 * sum[2];
}

__global__ void convolutionColGPU(float32_t* d_Result, const float32_t* d_Data, int dataW, int dataH)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ float32_t data[3 * TILE_W * (TILE_H + KERNEL_RADIUS * 2)]; // 3 channels of TILE_H rows and (TILE_W + KERNEL_RADIUS * 2) columns

	// original image based coordinate
	const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
	const int shift = threadIdx.y * (TILE_W);

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
	const int shift1 = shift + IMUL(blockDim.y, TILE_W);
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

//Image width should be aligned to maximum coalesced read/write size
//for best global memory performance in both row and column filter.
constexpr auto KERNEL_SIZE = static_cast<int32_t>(KERNEL_W * sizeof(float32_t));

void FilterBenchmark(const cv::Mat& image)
{
	std::cout << "----------- CONVOLUTION ------------\n";

	const auto dw = image.cols;
	const auto dh = image.rows;

	const auto h_Kernel = cv::getGaussianKernel(KERNEL_SIZE, -1, CV_32F);

	cudaMemcpyToSymbol(d_Kernel, h_Kernel.data, KERNEL_SIZE);

	dim3 blocks(TILE_W, TILE_H);
	dim3 grids(dw / TILE_W, dh / TILE_H);

	auto multiplier = size_t{};
	DeviceAlloc::ComputeSize(image, &multiplier);

	auto h_Result = cv::Mat(cv::saturate_cast<int>(image.rows * multiplier), image.cols, CV_32FC3);

	{
		const auto timeLock = MeasureTime("Time computing+load+unload");

		const auto d_Image = DeviceAlloc(image);

		auto d_Result = DeviceAlloc(image);
		auto d_Data = DeviceAlloc(d_Image.m_size);

		{
			const auto timeLock = MeasureTime("Time computing+unload");

			{
				const auto timeLock3 = MeasureTime("Time computing");
				convolutionRowGPU<<<grids, blocks>>>((float32_t*)d_Data.m_deviceData, (const float32_t*)d_Image.m_deviceData, dw, dh);
				//convolutionColGPU<<<grids, blocks>>>((float32_t*)d_Result.m_deviceData, (const float32_t*)d_Data.m_deviceData, dw, dh);
			}

			cudaDeviceSynchronize();
			d_Data.CopyToHost(h_Result.data);
		}

		cudaDeviceSynchronize();
	}

	cv::imwrite("C://Users/trom/Downloads/image.png", h_Result);

    std::cout << "------------------------------------\n" << std::endl;
}
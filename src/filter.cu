#include "main.cuh"

using float32_t = float;

constexpr auto KERNEL_RADIUS = 2;
constexpr auto KERNEL_W = (2 * KERNEL_RADIUS + 1); // 5 x 5 kernel

__device__ __constant__ float32_t d_Kernel[KERNEL_W];

constexpr auto TILE_W = 16;		// active cell width
constexpr auto TILE_H = 16;		// active cell height
//constexpr auto TILE_SIZE = (TILE_W + KERNEL_RADIUS * 2) * (TILE_W + KERNEL_RADIUS * 2);

#define IMUL(a,b) __mul24(a,b)

__global__ void convolutionRowGPU(float32_t* d_Result, float32_t* d_Data, int dataW, int dataH)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ float32_t data[TILE_H * (TILE_W + KERNEL_RADIUS * 2)];

	// global mem address of this thread
	const int gLoc = threadIdx.x +
		IMUL(blockIdx.x, blockDim.x) +
		IMUL(threadIdx.y, dataW) +
		IMUL(blockIdx.y, blockDim.y) * dataW;

	// load cache (32x16 shared memory, 16x16 threads blocks)
	// each threads loads two values from global memory into shared mem
	// if in image area, get value in global mem, else 0
	int x;		// image based coordinate

	// original image based coordinate
	const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
	const int shift = threadIdx.y * (TILE_W + KERNEL_RADIUS * 2);

	// case1: left
	x = x0 - KERNEL_RADIUS;
	if (x < 0)
		data[threadIdx.x + shift] = 0;
	else
		data[threadIdx.x + shift] = d_Data[gLoc - KERNEL_RADIUS];

	// case2: right
	x = x0 + KERNEL_RADIUS;
	if (x > dataW - 1)
		data[threadIdx.x + blockDim.x + shift] = 0;
	else
		data[threadIdx.x + blockDim.x + shift] = d_Data[gLoc + KERNEL_RADIUS];

	__syncthreads();

	// convolution
	float32_t sum = 0;
	x = KERNEL_RADIUS + threadIdx.x;
	for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
		sum += data[x + i + shift] * d_Kernel[KERNEL_RADIUS + i];

	d_Result[gLoc] = sum;

}

__global__ void convolutionColGPU(float32_t* d_Result, float32_t* d_Data, int dataW, int dataH)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ float32_t data[TILE_W * (TILE_H + KERNEL_RADIUS * 2)];

	// global mem address of this thread
	const int gLoc = threadIdx.x +
		IMUL(blockIdx.x, blockDim.x) +
		IMUL(threadIdx.y, dataW) +
		IMUL(blockIdx.y, blockDim.y) * dataW;

	// load cache (32x16 shared memory, 16x16 threads blocks)
	// each threads loads two values from global memory into shared mem
	// if in image area, get value in global mem, else 0
	int y;		// image based coordinate

	// original image based coordinate
	const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
	const int shift = threadIdx.y * (TILE_W);

	// case1: upper
	y = y0 - KERNEL_RADIUS;
	if (y < 0)
		data[threadIdx.x + shift] = 0;
	else
		data[threadIdx.x + shift] = d_Data[gLoc - IMUL(dataW, KERNEL_RADIUS)];

	// case2: lower
	y = y0 + KERNEL_RADIUS;
	const int shift1 = shift + IMUL(blockDim.y, TILE_W);
	if (y > dataH - 1)
		data[threadIdx.x + shift1] = 0;
	else
		data[threadIdx.x + shift1] = d_Data[gLoc + IMUL(dataW, KERNEL_RADIUS)];

	__syncthreads();

	// convolution
	float32_t sum = 0;
	for (int i = 0; i <= KERNEL_RADIUS * 2; i++)
		sum += data[threadIdx.x + (threadIdx.y + i) * TILE_W] * d_Kernel[i];

	d_Result[gLoc] = sum;

}

//Image width should be aligned to maximum coalesced read/write size
//for best global memory performance in both row and column filter.
constexpr auto KERNEL_SIZE = static_cast<int32_t>(KERNEL_W * sizeof(float32_t));

void loadRawImage(const cv::Mat& image, void* h_DataR, void* h_DataG, void* h_DataB)
{
	std::vector<cv::Mat> mats;
	cv::split(image, mats);
	assert(mats.size() == 3);

	std::copy(mats[2].datastart, mats[2].dataend, (uint8_t*)h_DataR);
	std::copy(mats[1].datastart, mats[1].dataend, (uint8_t*)h_DataG);
	std::copy(mats[0].datastart, mats[0].dataend, (uint8_t*)h_DataB);
}

void FilterBenchmark(const cv::Mat& image)
{
    std::cout << "----------- CONVOLUTION ------------\n";

	float32_t* h_DataR, * h_DataG, * h_DataB, * h_ResultR, * h_ResultG, * h_ResultB;
	float32_t* d_DataA, * d_DataB;

	const auto dw = image.cols;
	const auto dh = image.rows;

	const auto data_size = dw * dh * sizeof(float32_t);

	const auto h_Kernel = cv::getGaussianKernel(KERNEL_SIZE, -1, CV_32F);

	h_DataR = (float32_t*)malloc(data_size);
	h_DataG = (float32_t*)malloc(data_size);
	h_DataB = (float32_t*)malloc(data_size);
	h_ResultR = (float32_t*)malloc(data_size);
	h_ResultG = (float32_t*)malloc(data_size);
	h_ResultB = (float32_t*)malloc(data_size);

	cudaMalloc((void**)&d_DataA, data_size);
	cudaMalloc((void**)&d_DataB, data_size);

	loadRawImage(image, h_DataR, h_DataG, h_DataB);

	cudaMemcpyToSymbol(d_Kernel, h_Kernel.data, KERNEL_SIZE);

	dim3 blocks(TILE_W, TILE_H);
	dim3 grids(dw / TILE_W, dh / TILE_H);

	{
		const auto timeLock = MeasureTime("Time");

		// red channel
		cudaMemcpy(d_DataA, h_DataR, data_size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		convolutionRowGPU<<<grids, blocks>>>(d_DataB, d_DataA, dw, dh);
		convolutionColGPU<<<grids, blocks>>>(d_DataA, d_DataB, dw, dh);
		cudaDeviceSynchronize();

		// read back GPU result
		cudaMemcpy(h_ResultR, d_DataA, data_size, cudaMemcpyDeviceToHost);

		// green channel
		cudaMemcpy(d_DataA, h_DataG, data_size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		convolutionRowGPU<<<grids, blocks>>>(d_DataB, d_DataA, dw, dh);
		convolutionColGPU<<<grids, blocks>>>(d_DataA, d_DataB, dw, dh);
		cudaDeviceSynchronize();

		// read back GPU result
		cudaMemcpy(h_ResultG, d_DataA, data_size, cudaMemcpyDeviceToHost);

		// blue channel
		cudaMemcpy(d_DataA, h_DataB, data_size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		convolutionRowGPU<<<grids, blocks>>>(d_DataB, d_DataA, dw, dh);
		convolutionColGPU<<<grids, blocks>>>(d_DataA, d_DataB, dw, dh);
		cudaDeviceSynchronize();

		// read back GPU result
		cudaMemcpy(h_ResultB, d_DataA, data_size, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_DataB);
	cudaFree(d_DataA);

	free(h_ResultB);
	free(h_ResultG);
	free(h_ResultR);
	free(h_DataB);
	free(h_DataG);
	free(h_DataR);

    std::cout << "------------------------------------\n" << std::endl;
}
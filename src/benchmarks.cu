#include <vector>
#include <cassert>
#include <numeric>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "main.h"

// Approaches are from the presentation "Optimizing Parallel Reduction in CUDA"
// link: http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

constexpr auto BLOCK_SIZE_1024 = 1024; // number of threads per block
constexpr auto BLOCK_SIZE_512 = 512;

#pragma region Utils

struct DeviceAlloc
{
    DeviceAlloc(const cv::Mat& image)
    {
        const auto imageSize = sizeof(uint8_t) * image.rows * image.cols * image.channels();
        m_size = IMAGE_MULTIPLIER * imageSize;
        cudaMalloc((void**)&m_deviceData, m_size);
        cudaMemcpy(m_deviceData, image.data, imageSize, cudaMemcpyHostToDevice);

        for (int i = 1; i < IMAGE_MULTIPLIER; ++i)
        {
            cudaMemcpy(m_deviceData + i * imageSize, m_deviceData, imageSize, cudaMemcpyDeviceToDevice);
        }
        m_pixelsCount = IMAGE_MULTIPLIER * image.rows * image.cols;
    }

    DeviceAlloc(size_t size)
    {
        m_size = size;
        cudaMalloc((void**)&m_deviceData, m_size);
        cudaMemset(m_deviceData, 0, m_size);
    }

    void CopyToHost(void* dst) const
    {
        cudaMemcpy(dst, m_deviceData, m_size, cudaMemcpyDeviceToHost);
    }

    ~DeviceAlloc()
    {
        cudaFree(m_deviceData);
    }

    uint8_t* m_deviceData;
    size_t m_size;
    size_t m_pixelsCount;
};

template<class T>
__device__ const T& min(const T& a, const T& b)
{
    return (b < a) ? b : a;
}

template<class ForwardIt>
__device__ ForwardIt min_element(ForwardIt first, ForwardIt last)
{
    if (first == last) return last;

    ForwardIt smallest = first;
    ++first;
    for (; first != last; ++first) {
        if (*first < *smallest) {
            smallest = first;
        }
    }
    return smallest;
}

#pragma endregion

// Reduction #1: Interleaved Addressing
__global__ void SumPixels(const uint8_t* data, uint64_t* output)
{
    __shared__ uint64_t sdata[BLOCK_SIZE_1024];

    const auto tid = threadIdx.x;
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = data[3 * i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void SumPixels2(const uint64_t* data, uint64_t* output)
{
    __shared__ uint64_t sdata[BLOCK_SIZE_512];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = data[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void SumPixelsBenchmark(const cv::Mat& image)
{
    std::cout << "---------- PIXELS SUMMATION ----------\n";

    {
        const auto timeLock = MeasureTime("Time with copy");

        const auto imageDevice = DeviceAlloc(image);

        const auto pixelsCount = imageDevice.m_pixelsCount;
        const auto gridSize = static_cast<unsigned int>(pixelsCount / BLOCK_SIZE_1024);
        const auto gridSize2 = static_cast<unsigned int>(gridSize / BLOCK_SIZE_512);
        auto sumDevice = DeviceAlloc(sizeof(uint64_t) * gridSize);
        auto sumDevice2 = DeviceAlloc(sizeof(uint64_t) * gridSize2);

        {
            const auto timeLock2 = MeasureTime("Time without copy");

            SumPixels<<<gridSize, BLOCK_SIZE_1024>>>(imageDevice.m_deviceData, (uint64_t*)sumDevice.m_deviceData);
            SumPixels2<<<gridSize2, BLOCK_SIZE_512>>>((const uint64_t*)sumDevice.m_deviceData, (uint64_t*)sumDevice2.m_deviceData);
        }

        auto sumVector = std::vector<uint64_t>(gridSize2); // ~2Kb
        sumDevice2.CopyToHost(sumVector.data());
        const volatile auto sum = std::accumulate(sumVector.begin(), sumVector.end(), uint64_t{ 0 });
    }

    std::cout << "------------------------------------\n" << std::endl;
}

__device__ void warpReduce(volatile uint8_t* sdata, int tid) {
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

// Reduction #4 & #5: First Add During Load + Unroll the Last Warp
__global__ void ReducePixels(const uint8_t* data, uint8_t* output)
{
    __shared__ uint8_t sdata[BLOCK_SIZE_512];

    const auto tid = threadIdx.x;
    const auto i = blockIdx.x * (blockDim.x * 2) + tid;
    sdata[tid] = min(data[3 * i], data[3 * (i + blockDim.x)]);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Reduction #5: Unroll the Last Warp (without First Add During Load)
__global__ void ReducePixels2(const uint8_t* data, uint8_t* output)
{
    __shared__ uint8_t sdata[BLOCK_SIZE_512];

    const auto tid = threadIdx.x;
    const auto i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = data[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void ReducePixelsBenchmark(const cv::Mat& image)
{
    std::cout << "---------- PIXELS REDUCTION ----------\n";

    {
        const auto timeLock = MeasureTime("Time with copy");
        const auto imageDevice = DeviceAlloc(image);

        const auto pixelsCount = imageDevice.m_pixelsCount;
        const auto gridSize = pixelsCount / BLOCK_SIZE_512 / 2;
        const auto gridSize2 = gridSize / BLOCK_SIZE_512;

        auto minDevice = DeviceAlloc(gridSize);
        auto minDevice2 = DeviceAlloc(gridSize2);

        {
            const auto timeLock2 = MeasureTime("Time without copy");
            ReducePixels<<<gridSize, BLOCK_SIZE_512>>>(imageDevice.m_deviceData, minDevice.m_deviceData);
            ReducePixels2<<<gridSize2, BLOCK_SIZE_512>>>(minDevice.m_deviceData, minDevice2.m_deviceData);
        }

        auto minVector = std::vector<uint8_t>(gridSize2); // ~2Kb
        minDevice2.CopyToHost(minVector.data());
        const volatile auto minValue = *std::min_element(minVector.begin(), minVector.end());
    }

    std::cout << "------------------------------------\n" << std::endl;
}

void FilterBenchmark(const cv::Mat& image)
{

}
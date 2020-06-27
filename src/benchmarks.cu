#include <vector>
#include <cassert>
#include <numeric>

#include "main.cuh"

// IMPORTANT
// Approaches are from the presentation "Optimizing Parallel Reduction in CUDA"
// link: http://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

// IMPORTANT
// The algorithms are applied 2 times to achieve better performance

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
    std::cout << "--------- PIXELS SUMMATION ---------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);

    {
        const auto timeLock = MeasureTime("Computation time+unload+load");

        const DeviceAlloc imageDevice(inputImage);

        const auto pixelsCount = imageDevice.m_pixelsCount;
        const auto gridSize = static_cast<unsigned int>(pixelsCount / BLOCK_SIZE_1024);
        const auto gridSize2 = static_cast<unsigned int>(gridSize / BLOCK_SIZE_512);
        DeviceAlloc sumDevice(sizeof(uint64_t) * gridSize);
        DeviceAlloc sumDevice2(sizeof(uint64_t) * gridSize2);

        {
            const auto timeLock2 = MeasureTime("Computation time+unload");

            {
                const auto timeLock3 = MeasureTime("Computation time");

                SumPixels << <gridSize, BLOCK_SIZE_1024 >> > (imageDevice.m_deviceData, (uint64_t*)sumDevice.m_deviceData);
                cudaDeviceSynchronize();
                SumPixels2 << <gridSize2, BLOCK_SIZE_512 >> > ((const uint64_t*)sumDevice.m_deviceData, (uint64_t*)sumDevice2.m_deviceData);
                cudaDeviceSynchronize();
            }

            auto sumVector = std::vector<uint64_t>(gridSize2); // ~2Kb
            sumDevice2.CopyToHost(sumVector.data());
            const volatile auto sum = std::accumulate(sumVector.begin(), sumVector.end(), uint64_t{ 0 });
        }
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
    std::cout << "--------- PIXELS REDUCTION ---------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);

    {
        const auto timeLock = MeasureTime("Computation time+load+unload");
        const DeviceAlloc imageDevice(inputImage);

        const auto pixelsCount = imageDevice.m_pixelsCount;
        const auto gridSize = static_cast<unsigned int>(pixelsCount / BLOCK_SIZE_512 / 2);
        const auto gridSize2 = static_cast<unsigned int>(gridSize / BLOCK_SIZE_512);

        DeviceAlloc minDevice(gridSize);
        DeviceAlloc minDevice2(gridSize2);

        {
            const auto timeLock2 = MeasureTime("Computation time+unload");

            {
                const auto timeLock3 = MeasureTime("Computation time");
                ReducePixels<<<gridSize, BLOCK_SIZE_512>>>(imageDevice.m_deviceData, minDevice.m_deviceData);
                cudaDeviceSynchronize();
                ReducePixels2<<<gridSize2, BLOCK_SIZE_512>>>(minDevice.m_deviceData, minDevice2.m_deviceData);
                cudaDeviceSynchronize();
            }
            
            auto minVector = std::vector<uint8_t>(gridSize2); // ~2Kb
            minDevice2.CopyToHost(minVector.data());
            const volatile auto minValue = *std::min_element(minVector.begin(), minVector.end());
        }
    }

    std::cout << "------------------------------------\n" << std::endl;
}
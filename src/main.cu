#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <opencv2/imgcodecs.hpp>

#include "main.h"

constexpr auto PIXELS_PER_THREAD = 400;
constexpr auto BLOCK_SIZE = 1024; // number of threads per block

// RAII-wrapper over device data
struct DeviceAlloc
{
    DeviceAlloc(const cv::Mat& image)
    {
        m_size = sizeof(uint8_t) * image.rows * image.cols * image.channels();
        cudaMalloc((void**)&m_deviceData, m_size);
        cudaMemcpy(m_deviceData, image.data, m_size, cudaMemcpyHostToDevice);
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

__global__ void SumPixels(const uint8_t* data, uint64_t* output)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    const auto threadData = data + 3 * PIXELS_PER_THREAD * id;

    auto sum = uint64_t{ 0 };
    for (size_t i = 0; i < PIXELS_PER_THREAD; ++i)
    {
        sum += threadData[3 * i];
    }
    output[id] = sum;
}

void SumPixelsBenchmark(const cv::Mat& image)
{
    std::cout << "---------- PIXELS SUMMATION ----------\n";

    const auto pixelsCount = static_cast<size_t>(image.rows) * image.cols;
    const auto threadsCount = pixelsCount / PIXELS_PER_THREAD;
    const auto gridSize = threadsCount / BLOCK_SIZE; // the image was selected so we don't care about rounding

    {
        const auto timeLock = MeasureTime("Time with copy");

        const auto imageDevice = DeviceAlloc(image);
        auto sumDeviceLock = DeviceAlloc(sizeof(uint64_t) * threadsCount);

        {
            const auto timeLock2 = MeasureTime("Time without copy");

            SumPixels<<<gridSize, BLOCK_SIZE>>>(imageDevice.m_deviceData, (uint64_t*)sumDeviceLock.m_deviceData); // sum over B (because BGR)
        }

        // "Time without copy" is not a sincere truth, because we need to copy data back to the host, which takes time.
        // But in real world, the results of such addition can be used in some other algorithm at the device, so it makes sense to benchmark 
        // per-thread sum, which was done above.

        auto sumVector = std::vector<uint64_t>(threadsCount); // ~83'000 numbers
        sumDeviceLock.CopyToHost(sumVector.data());
        const volatile auto sum = std::accumulate(sumVector.begin(), sumVector.end(), uint64_t{ 0 });
    }

    std::cout << "------------------------------------\n" << std::endl;
}

__global__ void ReducePixels(const uint8_t* data, uint8_t* output)
{
    __shared__ int mins[BLOCK_SIZE];

    const auto blockId = blockIdx.x;
    const auto threadId = threadIdx.x;
    const auto id = blockId * blockDim.x + threadId;

    const auto threadData = data + 3 * PIXELS_PER_THREAD * id;

    auto minValue = threadData[0];
    for (size_t i = 1; i < PIXELS_PER_THREAD; ++i)
    {
        minValue = min(minValue, threadData[3 * i]);
    }
    mins[threadId] = minValue;

    __syncthreads();

    if (threadId == 0)
    {
        output[blockId] = *min_element(mins, mins + BLOCK_SIZE);
    }
}

void ReducePixelsBenchmark(const cv::Mat& image)
{
    std::cout << "---------- PIXELS REDUCTION ----------\n";

    const auto pixelsCount = static_cast<size_t>(image.rows) * image.cols;
    const auto threadsCount = pixelsCount / PIXELS_PER_THREAD;
    const auto gridSize = threadsCount / BLOCK_SIZE; // the image was selected so we don't care about rounding

    {
        const auto timeLock = MeasureTime("Time with copy");
        const auto imageDevice = DeviceAlloc(image);
        auto minDevice = DeviceAlloc(sizeof(uint8_t) * gridSize);

        {
            const auto timeLock2 = MeasureTime("Time without copy");
            ReducePixels<<<gridSize, BLOCK_SIZE>>>(imageDevice.m_deviceData, minDevice.m_deviceData);
        }

        auto minVector = std::vector<uint8_t>(gridSize); // 81 items
        minDevice.CopyToHost(minVector.data());
        const volatile auto minValue = *std::min_element(minVector.begin(), minVector.end());
    }

    std::cout << "------------------------------------\n" << std::endl;
}

int main()
{
    const auto image = cv::imread(imgPath, cv::IMREAD_COLOR);
    assert(image.type() == CV_8UC3);

    SumPixelsBenchmark(image);
    ReducePixelsBenchmark(image);

    return 0;
}
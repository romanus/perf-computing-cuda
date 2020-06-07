#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <opencv2/imgcodecs.hpp>

#include "main.h"

constexpr auto PIXELS_PER_THREAD = 400;

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

__global__
void SumPixels(const uint8_t* data, uint64_t* output)
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

    const auto blockSize = 1024;
    const auto gridSize = threadsCount / blockSize; // the image was selected so we don't care about rounding

    {
        const auto timeLock = MeasureTime("Time with copy");

        const auto imageDevice = DeviceAlloc(image);
        auto sumDeviceLock = DeviceAlloc(sizeof(uint64_t) * threadsCount);

        {
            const auto timeLock2 = MeasureTime("Time without copy");

            SumPixels<<<gridSize, blockSize>>>(imageDevice.m_deviceData, (uint64_t*)sumDeviceLock.m_deviceData); // sum over B (because BGR)
        }

        // "Time without copy" is not a sincere truth, because we need to copy data back to the host, which takes time.
        // But in real world, the results of such addition can be used in some other algorithm at the device, so it makes sense to benchmark 
        // per-thread sum, which was done above.

        auto sumVector = std::vector<uint64_t>(threadsCount);
        sumDeviceLock.CopyToHost(sumVector.data());
        const volatile auto sum = std::accumulate(sumVector.begin(), sumVector.end(), uint64_t{ 0 });
    }

    std::cout << "------------------------------------\n" << std::endl;
}

int main()
{
    const auto image = cv::imread(imgPath, cv::IMREAD_COLOR);
    assert(image.type() == CV_8UC3);

    SumPixelsBenchmark(image);

    return 0;
}
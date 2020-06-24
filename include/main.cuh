#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "device_functions.h"

#include "main.h"

constexpr auto BLOCK_SIZE_1024 = uint32_t{ 1024 }; // number of threads per block
constexpr auto BLOCK_SIZE_512 = uint32_t{ 512 };

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
        m_pixelsCount = static_cast<size_t>(-1);
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
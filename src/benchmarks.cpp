#include <cassert>
#include <numeric>

#include "main.h"

uint64_t SumPixels(const uint8_t* data, size_t pixelsCount)
{
    // sum over the first channel: B (because BGR)

    auto sum = uint64_t{ 0 };
    volatile auto vi = 0;
    for (int i = 0; i < IMAGE_MULTIPLIER; ++i)
    {
        vi = i;
        for (size_t i = 0; i < pixelsCount; ++i)
        {
            sum += data[3 * i];
        }
    }
    return sum;
}

void SumPixelsBenchmark(const cv::Mat& image)
{
    std::cout << "--------- PIXELS SUMMATION ---------\n";

    const auto pixelsCount = static_cast<size_t>(image.rows) * image.cols;

    {
        const auto timeLock = MeasureTime("Time without copy");
        const volatile auto sum = SumPixels(image.data, pixelsCount);
    }

    std::cout << "------------------------------------\n" << std::endl;
}

uint8_t ReducePixels(const uint8_t* data, size_t pixelsCount)
{
    std::vector<uint8_t> minValues(IMAGE_MULTIPLIER);
    volatile auto vi = 0;

    for (int i = 0; i < IMAGE_MULTIPLIER; ++i)
    {
        vi = i;
        // this is going not through a single channel, but through 'pixelsCount' sequential pixels. But still, we can estimate timings here.
        minValues[i] = std::reduce(data, data + pixelsCount, std::numeric_limits<uint8_t>::max(), [](const auto& val1, const auto& val2) { return std::min(val1, val2); });
    }

    return std::reduce(minValues.begin(), minValues.end(), std::numeric_limits<uint8_t>::max(), [](const auto& val1, const auto& val2) { return std::min(val1, val2); });
}

void ReducePixelsBenchmark(const cv::Mat& image)
{
    std::cout << "--------- PIXELS REDUCTION ---------\n";

    const auto pixelsCount = static_cast<size_t>(image.rows) * image.cols;

    {
        const auto timeLock = MeasureTime("Time without copy");
        const volatile auto minValue = ReducePixels(image.data, pixelsCount);
    }

    std::cout << "------------------------------------\n" << std::endl;
}

void FilterBenchmark(const cv::Mat& image)
{
    std::cout << "----------- CONVOLUTION ------------\n";

    auto blurred = image.clone(); // no allocation is performed in a loop

    const auto kernel = cv::getGaussianKernel(5, -1, CV_32F);

    {
        const auto timeLock = MeasureTime("Time");

        for (int i = 0; i < IMAGE_MULTIPLIER; ++i)
        {
            cv::sepFilter2D(image, blurred, CV_32F, kernel, kernel, cv::Point(-1, -1), 0, cv::BorderTypes::BORDER_CONSTANT);
        }
    }

    std::cout << "------------------------------------\n" << std::endl;
}
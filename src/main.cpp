#include <iostream>
#include <cassert>

#include <opencv2/imgcodecs.hpp>

#include "main.h"

uint64_t SumPixels(const uint8_t* data, size_t pixelsCount)
{
    // sum over the first channel: B (because BGR)

    auto sum = uint64_t{ 0 };
    for (size_t i = 0; i < pixelsCount; ++i)
    {
        sum += data[3 * i];
    }
    return sum;
}

void SumPixelsBenchmark(const cv::Mat& image)
{
    std::cout << "---------- PIXELS SUMMATION ----------\n";

    const auto pixelsCount = static_cast<size_t>(image.rows) * image.cols;

    {
        const auto timeLock = MeasureTime("Time");
        const volatile auto sum = SumPixels(image.data, pixelsCount);
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
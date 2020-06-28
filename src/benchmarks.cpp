#include <cassert>
#include <numeric>

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
    std::cout << "--------- PIXELS SUMMATION ---------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);

    const auto pixelsCount = static_cast<size_t>(inputImage.rows) * inputImage.cols;

    auto timer = MeasureTime("Computation time");
    timer.Start();

    const volatile auto sum = SumPixels(inputImage.data, pixelsCount);

    timer.Stop();
    timer.Print();

    std::cout << "------------------------------------\n" << std::endl;
}

uint8_t ReducePixels(const uint8_t* data, size_t pixelsCount)
{
    // this is going not through a single channel, but through 'pixelsCount' sequential pixels. But still, we can estimate timings here.
    return std::reduce(data, data + pixelsCount, std::numeric_limits<uint8_t>::max(), [](const auto& val1, const auto& val2) { return std::min(val1, val2); });
}

void ReducePixelsBenchmark(const cv::Mat& image)
{
    std::cout << "--------- PIXELS REDUCTION ---------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);

    const auto pixelsCount = static_cast<size_t>(inputImage.rows) * inputImage.cols;

    auto timer = MeasureTime("Computation time");
    timer.Start();

    const auto minValue = ReducePixels(inputImage.data, pixelsCount);

    timer.Stop();
    timer.Print();

    std::cout << "------------------------------------\n" << std::endl;
}

void FilterBenchmark(const cv::Mat& image)
{
    std::cout << "----------- CONVOLUTION ------------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);
    auto blurred = inputImage.clone(); // no memory is allocated inside sepFilter2D

    const auto kernel = cv::getGaussianKernel(5, -1, CV_32F);

    auto timer = MeasureTime("Computation time");
    timer.Start();

    cv::sepFilter2D(inputImage, blurred, CV_32F, kernel, kernel, cv::Point(-1, -1), 0, cv::BorderTypes::BORDER_CONSTANT);

    timer.Stop();
    timer.Print();

    std::cout << "------------------------------------\n" << std::endl;
}
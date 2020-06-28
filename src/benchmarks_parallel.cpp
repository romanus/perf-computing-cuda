#include <cassert>
#include <numeric>
#include <thread>
#include <atomic>
#include <mutex>

#include "main.h"

void ParallelSum(const uint8_t* datastart, const uint8_t* dataend, std::atomic_uint64_t& sum)
{
    // sum over the first channel: B (because BGR)

    uint64_t threadSum = 0;
    for (auto* iter = datastart; iter < dataend; iter += 3)
    {
        threadSum += *iter;
    }
    sum += threadSum;
}

uint64_t SumPixels(const uint8_t* datastart, const uint8_t* dataend, uint8_t threadCount)
{
    auto sum = std::atomic_uint64_t{ 0 };

    auto* threads = new std::thread[threadCount];

    const auto pixelsPerThread = (dataend - datastart) / threadCount;
    for (int threadIdx = 0; threadIdx < threadCount; ++threadIdx)
    {
        const auto* begin = datastart + threadIdx * pixelsPerThread;
        const auto* end = begin + pixelsPerThread;
        threads[threadIdx] = std::thread(ParallelSum, begin, end, std::ref(sum));
    }

    for (int threadIdx = 0; threadIdx < threadCount; ++threadIdx)
    {
        threads[threadIdx].join();
    }

    delete[] threads;

    return sum;
}

void SumPixelsBenchmark(const cv::Mat& image)
{
    std::cout << "--------- PIXELS SUMMATION ---------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);

    const auto pixelsCount = static_cast<size_t>(inputImage.rows) * inputImage.cols;

    for (const uint8_t threadCount : { 1, 2, 4, 8, 16, 32 })
    {
        auto timer = MeasureTime("Computation time (" + std::to_string(threadCount) + " threads)");
        timer.Start();

        const volatile auto sum = SumPixels(inputImage.datastart, inputImage.dataend, threadCount);

        timer.Stop();
        timer.Print();
    }

    std::cout << "------------------------------------\n" << std::endl;
}

void ParallelReduce(const uint8_t* datastart, const uint8_t* dataend, uint8_t& minVal)
{
    // sum over the first channel: B (because BGR)

    static std::mutex mu;

    auto threadMinVal = std::numeric_limits<uint8_t>::max();
    for (auto* iter = datastart; iter < dataend; iter += 3)
    {
        threadMinVal = std::min(threadMinVal, *iter);
    }

    std::lock_guard<std::mutex> lock(mu);
    minVal = std::min(minVal, threadMinVal);
}

uint8_t ReducePixels(const uint8_t* datastart, const uint8_t* dataend, uint8_t threadCount)
{
    auto minVal = std::numeric_limits<uint8_t>::max();

    auto* threads = new std::thread[threadCount];

    const auto pixelsPerThread = (dataend - datastart) / threadCount;
    for (int threadIdx = 0; threadIdx < threadCount; ++threadIdx)
    {
        const auto* begin = datastart + threadIdx * pixelsPerThread;
        const auto* end = begin + pixelsPerThread;
        threads[threadIdx] = std::thread(ParallelReduce, begin, end, std::ref(minVal));
    }

    for (int threadIdx = 0; threadIdx < threadCount; ++threadIdx)
    {
        threads[threadIdx].join();
    }

    delete[] threads;

    return minVal;
}

void ReducePixelsBenchmark(const cv::Mat& image)
{
    std::cout << "--------- PIXELS REDUCTION ---------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);

    const auto pixelsCount = static_cast<size_t>(inputImage.rows) * inputImage.cols;

    for (const uint8_t threadCount : { 1, 2, 4, 8, 16, 32 })
    {
        auto timer = MeasureTime("Computation time (" + std::to_string(threadCount) + " threads)");
        timer.Start();

        const auto minValue = ReducePixels(inputImage.datastart, inputImage.dataend, threadCount);

        timer.Stop();
        timer.Print();
    }

    std::cout << "------------------------------------\n" << std::endl;
}

void ParallelFilter(const cv::Mat input, const cv::Mat kernel, cv::Mat& output, int rowBegin, int rowEnd)
{
    // we don't use any locks here because writes don't lock each other
    const auto inputRanged = input.rowRange(rowBegin, rowEnd);
    const auto outputRanged = output.rowRange(rowBegin, rowEnd);

    cv::sepFilter2D(inputRanged, outputRanged, CV_32F, kernel, kernel, cv::Point(-1, -1), 0, cv::BorderTypes::BORDER_CONSTANT);
}

void Filter(const cv::Mat input, const cv::Mat kernel, cv::Mat& output, uint8_t threadCount)
{
    output.create(input.size(), input.type()); // assure we have enough memory allocated

    auto* threads = new std::thread[threadCount];

    const auto rowsPerThread = input.rows / threadCount;
    for (int threadIdx = 0; threadIdx < threadCount; ++threadIdx)
    {
        const auto begin = threadIdx * rowsPerThread;
        const auto end = begin + rowsPerThread;
        threads[threadIdx] = std::thread(ParallelFilter, input, kernel, output, begin, end);
    }

    for (int threadIdx = 0; threadIdx < threadCount; ++threadIdx)
    {
        threads[threadIdx].join();
    }

    delete[] threads;
}

void FilterBenchmark(const cv::Mat& image)
{
    std::cout << "----------- CONVOLUTION ------------\n";

    const auto inputImage = ImageMultiplier::Multiply(image);
    auto blurred = inputImage.clone(); // no memory is allocated inside sepFilter2D

    const auto kernel = cv::getGaussianKernel(5, -1, CV_32F);

    for (const uint8_t threadCount : { 1, 2, 4, 8, 16, 32 })
    {
        auto timer = MeasureTime("Computation time (" + std::to_string(threadCount) + " threads)");
        timer.Start();

        Filter(inputImage, kernel, blurred, threadCount);

        timer.Stop();
        timer.Print();
    }

    // compare the output and OpenCV output
    auto openCV_output = cv::Mat{};
    cv::sepFilter2D(inputImage, openCV_output, CV_32F, kernel, kernel, cv::Point(-1, -1), 0, cv::BorderTypes::BORDER_CONSTANT);

    const auto algoOutputEqual = std::equal(blurred.datastart, blurred.dataend, openCV_output.datastart);
    std::cout << "Parallel algorithm matches OpenCV: " << std::boolalpha << algoOutputEqual << std::endl;

    std::cout << "------------------------------------\n" << std::endl;
}
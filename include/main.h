#include <iostream>
#include <chrono>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

constexpr auto imgPath = "../resources/scenery.jpg";

struct MeasureTime
{
    MeasureTime(const std::string& message)
    {
        m_message = message;
    }

    void Start()
    {
        m_begin = std::chrono::steady_clock::now();
    }

    void Stop()
    {
        m_end = std::chrono::steady_clock::now();
    }

    void Print() const
    {
        const auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds> (m_end - m_begin).count();
        std::cout << m_message << ": " << time_ms << "[ms]\n";
    }

    std::chrono::steady_clock::time_point m_begin = {};
    std::chrono::steady_clock::time_point m_end = {};
    std::string m_message;
};

// In order to fully load GPU, the image is stacked a few times.
struct ImageMultiplier
{
    // the recommended multiplier so we don't try to allocate more memory than GPU has
    static size_t ComputeMultiplier(const cv::Mat& image)
    {
        return IMAGE_MULTIPLIER / (image.elemSize1() * image.channels());
    }

    static cv::Mat Multiply(const cv::Mat& image)
    {
        const auto multiplier = ComputeMultiplier(image);
        const auto imageSize = image.dataend - image.datastart;

        auto multipliedImage = cv::Mat(cv::saturate_cast<int>(image.rows * multiplier), image.cols, image.type());
        for (int i = 1; i < multiplier; ++i)
        {
            std::memcpy(multipliedImage.data + i * imageSize, image.data, imageSize);
        }

        return multipliedImage;
    }

    // for the first two tasks, we stack 32 images vertically; ~3Gb image is allocated, ~1M pixels
    // for the third task, we stack 2 images vertically
    static constexpr auto IMAGE_MULTIPLIER = size_t{ 32 };
};


void SumPixelsBenchmark(const cv::Mat& image);
void ReducePixelsBenchmark(const cv::Mat& image);
void FilterBenchmark(const cv::Mat& image);
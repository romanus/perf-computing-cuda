#include <iostream>
#include <chrono>

#include <opencv2/imgcodecs.hpp>

constexpr auto imgPath = "../resources/scenery.jpg";

// we stack 32 images vertically; ~3Gb image is allocated, ~1M pixels
constexpr auto IMAGE_MULTIPLIER = size_t{ 32 };

struct MeasureTime
{
    MeasureTime(const std::string& message)
    {
        m_message = message;
        m_begin = std::chrono::steady_clock::now();
    }

    ~MeasureTime()
    {
        const auto end = std::chrono::steady_clock::now();
        const auto time_ns = std::chrono::duration_cast<std::chrono::microseconds> (end - m_begin).count();
        std::cout << m_message << ": " << time_ns << "[us]\n";
    }

    std::chrono::steady_clock::time_point m_begin;
    std::string m_message;
};

void SumPixelsBenchmark(const cv::Mat& image);
void ReducePixelsBenchmark(const cv::Mat& image);
void FilterBenchmark(const cv::Mat& image);
#include <chrono>

const auto imgPath = "../resources/scenery.jpg";

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
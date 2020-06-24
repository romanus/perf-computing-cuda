#include "main.h"

int main()
{
    const auto image = cv::imread(imgPath, cv::IMREAD_COLOR);
    assert(image.type() == CV_8UC3);

    SumPixelsBenchmark(image);
    ReducePixelsBenchmark(image);

    auto imageFloat = cv::Mat{};
    image.convertTo(imageFloat, CV_32F);
    assert(imageFloat.type() == CV_32FC3);
    FilterBenchmark(imageFloat);

    return 0;
}
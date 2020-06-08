#include "main.h"

int main()
{
    const auto image = cv::imread(imgPath, cv::IMREAD_COLOR);
    assert(image.type() == CV_8UC3);

    SumPixelsBenchmark(image);
    ReducePixelsBenchmark(image);
    FilterBenchmark(image);

    return 0;
}
#pragma once

#include <opencv2/opencv.hpp>

namespace byte_track
{
class LBPFeatureExtractor
{
    public:
    enum class Type
    {
        NORMAL = 0,
        UNIFORM = 1,
    };

    LBPFeatureExtractor() = delete;
    ~LBPFeatureExtractor() = delete;

    static void exec(const cv::Mat &src, cv::Mat &dst, const Type &type = Type::NORMAL);

    private:
    static void calcLBP(const cv::Mat &src, cv::Mat &dst);
    static void calcUniformLBP(const cv::Mat &src, cv::Mat &dst);
};
}

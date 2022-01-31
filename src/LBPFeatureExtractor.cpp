#include <ByteTrack/LBPFeatureExtractor.h>

void byte_track::LBPFeatureExtractor::exec(const cv::Mat &src, cv::Mat &dst, const Type &type)
{
    if (src.type() != CV_8UC1)
    {
        throw std::runtime_error("The src should be 1 channel image.");
    }
    if (dst.type() != CV_8UC1 || src.size() != dst.size())
    {
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
    }

    switch (type)
    {
    case Type::NORMAL:
        calcLBP(src, dst);
        break;
    case Type::UNIFORM:
        calcUniformLBP(src, dst);
        break;
    default:
        std::runtime_error("Invalid type is specified in byte_track::LBPFeatureExtractor::exec()");
        break;
    }
}

void byte_track::LBPFeatureExtractor::calcLBP(const cv::Mat &src, cv::Mat &dst)
{
    for (int ri = 0; ri < src.rows; ri++)
    {
        const auto dst_ptr = dst.ptr<unsigned char>(ri);
        const auto src_t_ptr = src.ptr<unsigned char>(std::max(0, ri - 1));
        const auto src_m_ptr = src.ptr<unsigned char>(ri);
        const auto src_b_ptr = src.ptr<unsigned char>(std::min(src.rows - 1, ri + 1));
        for (int ci = 0; ci < src.cols; ci++)
        {
            const auto &centor = src_m_ptr[ci];
            const auto l_idx = std::max(0, ci - 1);
            const auto m_idx = ci;
            const auto r_idx = std::min(src.cols - 1, ci + 1);
            dst_ptr[ci] = 0;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_t_ptr[m_idx]) << 7;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_t_ptr[l_idx]) << 6;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_m_ptr[l_idx]) << 5;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_b_ptr[l_idx]) << 4;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_b_ptr[m_idx]) << 3;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_b_ptr[r_idx]) << 2;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_m_ptr[r_idx]) << 1;
            dst_ptr[ci] |= static_cast<unsigned char>(centor <= src_t_ptr[r_idx]) << 0;
        }
    }
}

void byte_track::LBPFeatureExtractor::calcUniformLBP(const cv::Mat &src, cv::Mat &dst)
{
    // TODO: impl
}

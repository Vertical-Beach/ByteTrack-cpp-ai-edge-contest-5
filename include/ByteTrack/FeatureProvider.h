#pragma once

#include <ByteTrack/Rect.h>
#include <ByteTrack/LBPFeatureExtractor.h>

#include <opencv2/opencv.hpp>

namespace byte_track
{

struct FeatureProviderCfg
{
    float scale {1.0};
    LBPFeatureExtractor::Type lbp_type {LBPFeatureExtractor::Type::NORMAL};
    int n_lbp_feature_hist_bins {10};
    int n_hue_hist_bins {10};
    int n_saturation_hist_bins {10};
    int lbp_feature_hist_range_lower {0};
    int lbp_feature_hist_range_upper {255};
    int saturation_hist_range_lower {0};
    int saturation_hist_range_upper {192};
};

class FeatureProvider
{
public:
    using Cfg = FeatureProviderCfg;

    explicit FeatureProvider(const cv::Mat &image,
                             const Cfg &cfg = Cfg());
    ~FeatureProvider();

    void setImage(const cv::Mat &image);

    const cv::Mat& getScaledImage() const;

    size_t getImageWidth() const;
    size_t getImageHeight() const;

    cv::Rect2i rect2ScaledRect2i(const byte_track::Rect<float> &rect) const;

    std::vector<float> getLbpFeature(const byte_track::Rect<float> &rect) const;
    std::vector<float> getLbpFeature(const cv::Rect2i &scaled_rect) const;
    std::pair<std::vector<float>, std::vector<float>> getColorFeature(const byte_track::Rect<float> &rect) const;
    std::pair<std::vector<float>, std::vector<float>> getColorFeature(const cv::Rect2i &scaled_rect) const;

private:
    void preproc();

    cv::Mat calcHist(const cv::Mat &mat, const int& range_lower, const int &range_upper, const int &bins) const;

    const Cfg cfg_;

    cv::Size original_size_;
    cv::Mat image_;
    cv::Mat hue_;
    cv::Mat saturation_;
    cv::Mat lbp_mat_;
};
}

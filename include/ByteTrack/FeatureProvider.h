#pragma once

#include <ByteTrack/Rect.h>
#include <ByteTrack/LBPFeatureExtractor.h>

#include <opencv2/opencv.hpp>

namespace byte_track
{

struct FeatureProviderCfg
{
    LBPFeatureExtractor::Type lbp_type {LBPFeatureExtractor::Type::NORMAL};
    int n_lbp_feature_hist_bins {10};
};

class FeatureProvider
{
public:
    using Cfg = FeatureProviderCfg;

    explicit FeatureProvider(const cv::Mat &image,
                             const Cfg &cfg = Cfg(),
                             const bool &clone = false);
    ~FeatureProvider();

    void setImage(const cv::Mat &image, const bool &clone = false);

    std::vector<float> getLbpFeature(const byte_track::Rect<float> &rect) const;

private:
    void preproc();

    const Cfg cfg_;

    cv::Mat image_;
    cv::Mat gray_;
    cv::Mat lbp_mat_;
};
}

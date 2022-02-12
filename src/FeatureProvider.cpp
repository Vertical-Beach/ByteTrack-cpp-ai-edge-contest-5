#include <ByteTrack/FeatureProvider.h>

byte_track::FeatureProvider::FeatureProvider(const cv::Mat &image,
                                             const Cfg &cfg) :
    cfg_(cfg)
{
    setImage(image);
}

byte_track::FeatureProvider::~FeatureProvider()
{
}

void byte_track::FeatureProvider::setImage(const cv::Mat &image)
{
    original_size_ = image.size();
    cv::resize(image, image_, cv::Size(image.cols * cfg_.scale, image.rows * cfg_.scale), 0, 0, cv::INTER_LINEAR);
    preproc();
}

const cv::Mat& byte_track::FeatureProvider::getScaledImage() const
{
    return image_;
}

size_t byte_track::FeatureProvider::getImageWidth() const
{
    return original_size_.width;
}

size_t byte_track::FeatureProvider::getImageHeight() const
{
    return original_size_.height;
}

cv::Rect2i byte_track::FeatureProvider::rect2ScaledRect2i(const byte_track::Rect<float> &rect) const
{
    cv::Rect2i rect2i;
    rect2i.x = std::clamp(static_cast<int>(std::round(rect.x() * cfg_.scale)), 0, image_.size().width - 1);
    rect2i.y = std::clamp(static_cast<int>(std::round(rect.y() * cfg_.scale)), 0, image_.size().height - 1);
    rect2i.width = std::clamp(static_cast<int>(std::round(rect.width() * cfg_.scale)), 0, image_.size().width - rect2i.x);
    rect2i.height = std::clamp(static_cast<int>(std::round(rect.height() * cfg_.scale)), 0, image_.size().height - rect2i.y);
    return rect2i;
}

std::vector<float> byte_track::FeatureProvider::getLbpFeature(const byte_track::Rect<float> &rect) const
{
    return getLbpFeature(rect2ScaledRect2i(rect));
}

std::vector<float> byte_track::FeatureProvider::getLbpFeature(const cv::Rect2i &scaled_rect) const
{
    cv::Mat hist = calcHist(lbp_mat_(scaled_rect), cfg_.lbp_feature_hist_range_lower,
                            cfg_.lbp_feature_hist_range_upper, cfg_.n_lbp_feature_hist_bins);

    std::vector<float> hist_v(hist.rows);
    for (int hi = 0; hi < hist.rows; hi++)
    {
        hist_v[hi] = hist.ptr<float>(hi)[0];
    }
    return hist_v;
}

std::pair<std::vector<float>, std::vector<float>> byte_track::FeatureProvider::getColorFeature(const byte_track::Rect<float> &rect) const
{
    return getColorFeature(rect2ScaledRect2i(rect));
}

std::pair<std::vector<float>, std::vector<float>> byte_track::FeatureProvider::getColorFeature(const cv::Rect2i &scaled_rect) const
{
    cv::Mat hue_hist = calcHist(hue_(scaled_rect), 0, 180, cfg_.n_hue_hist_bins);
    cv::Mat saturation_hist = calcHist(saturation_(scaled_rect), cfg_.saturation_hist_range_lower,
                                       cfg_.saturation_hist_range_upper, cfg_.n_saturation_hist_bins);

    std::pair<std::vector<float>, std::vector<float>> hist_v;
    hist_v.first.resize(hue_hist.rows);
    for (int hi = 0; hi < hue_hist.rows; hi++)
    {
        hist_v.first[hi] = hue_hist.ptr<float>(hi)[0];
    }

    hist_v.second.resize(saturation_hist.rows);
    for (int hi = 0; hi < saturation_hist.rows; hi++)
    {
        hist_v.second[hi] = saturation_hist.ptr<float>(hi)[0];
    }

    return hist_v;
}

void byte_track::FeatureProvider::preproc()
{
    cv::Mat gray, hsv;
    std::vector<cv::Mat> rected_hsv_mat_v;

    cv::cvtColor(image_, hsv, CV_RGB2HSV);
    cv::split(hsv, rected_hsv_mat_v);
    hue_ = rected_hsv_mat_v[0].clone();
    saturation_ = rected_hsv_mat_v[1].clone();

    cv::cvtColor(image_, gray, CV_RGB2GRAY);
    LBPFeatureExtractor::exec(gray, lbp_mat_, cfg_.lbp_type);
}

cv::Mat byte_track::FeatureProvider::calcHist(const cv::Mat &mat, const int &range_lower, const int &range_upper, const int &bins) const
{
    const float range[] = {static_cast<float>(range_lower), static_cast<float>(range_upper)};
    const float *r = range;
    cv::Mat hist;
    cv::calcHist(&mat, 1, 0, cv::Mat(), hist, 1, &bins, &r);
    return hist;
}

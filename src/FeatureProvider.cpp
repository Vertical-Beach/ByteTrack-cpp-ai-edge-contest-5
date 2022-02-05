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
    cv::resize(image, image_, cv::Size(image.cols * cfg_.scale, image.rows * cfg_.scale), 0, 0, cv::INTER_LINEAR);
    preproc();
}

std::vector<float> byte_track::FeatureProvider::getLbpFeature(const byte_track::Rect<float> &rect) const
{
    cv::Rect2i rect2i;
    rect2i.x = std::clamp(static_cast<int>(std::round(rect.x() * cfg_.scale)), 0, image_.size().width - 1);
    rect2i.y = std::clamp(static_cast<int>(std::round(rect.y() * cfg_.scale)), 0, image_.size().height - 1);
    rect2i.width = std::clamp(static_cast<int>(std::round(rect.width() * cfg_.scale)), 0, image_.size().width - rect2i.x);
    rect2i.height = std::clamp(static_cast<int>(std::round(rect.height() * cfg_.scale)), 0, image_.size().height - rect2i.y);

    const cv::Mat rected_lbp_mat = lbp_mat_(rect2i);
    const float range[] = {0, static_cast<float>(cfg_.n_lbp_feature_hist_bins)};
    const float *r = range;

    cv::Mat hist;
    cv::calcHist(&rected_lbp_mat, 1, 0, cv::Mat(), hist, 1, &cfg_.n_lbp_feature_hist_bins, &r);
    cv::normalize(hist, hist, 0.0, 1.0, cv::NORM_MINMAX, -1, cv::Mat());

    std::vector<float> hist_v(hist.rows);
    for (int hi = 0; hi < hist.rows; hi++)
    {
        hist_v[hi] = hist.ptr<float>(hi)[0];
    }
    return hist_v;
}

void byte_track::FeatureProvider::preproc()
{
    cv::cvtColor(image_, gray_, CV_RGB2GRAY);
    LBPFeatureExtractor::exec(gray_, lbp_mat_, cfg_.lbp_type);
}

#include <ByteTrack/STrack.h>

byte_track::STrack::STrack(const Rect<float>& rect,
                           const float& score,
                           const FeatureProviderPtr &fp_ptr,
                           const float &rect_h_padding_ratio,
                           const float &rect_v_padding_ratio,
                           const size_t &block_h_size,
                           const size_t &block_v_size) :
    rect_h_padding_ratio_(rect_h_padding_ratio),
    rect_v_padding_ratio_(rect_v_padding_ratio),
    block_h_size_(block_h_size),
    block_v_size_(block_v_size),
    kalman_filter_(),
    mean_(),
    covariance_(),
    rect_(rect),
    state_(STrackState::New),
    fp_ptr_(fp_ptr),
    is_activated_(false),
    score_(score),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0)
{
    const auto &fp_cfg = fp_ptr->getCfg();
    const auto feature_block_size = block_v_size_ * block_h_size_;
    lbp_feature_ = std::vector<float>(fp_cfg.n_lbp_feature_hist_bins * feature_block_size, 0);
    hue_feature_ = std::vector<float>(fp_cfg.n_hue_hist_bins * feature_block_size, 0);
    saturation_feature_ = std::vector<float>(fp_cfg.n_saturation_hist_bins * feature_block_size, 0);
    updateFeature();
}

byte_track::STrack::~STrack()
{
}

const byte_track::KalmanFilter::StateMean& byte_track::STrack::getKFStateMean() const
{
    return mean_;
}

const byte_track::Rect<float>& byte_track::STrack::getRect() const
{
    return rect_;
}

const byte_track::STrackState& byte_track::STrack::getSTrackState() const
{
    return state_;
}

const std::vector<float>& byte_track::STrack::getLBPFeature() const
{
    return lbp_feature_;
}

const std::vector<float>& byte_track::STrack::getHueFeature() const
{
    return hue_feature_;
}

const std::vector<float>& byte_track::STrack::getSaturationFeature() const
{
    return saturation_feature_;
}

const byte_track::STrack::FeatureProviderPtr& byte_track::STrack::getFeatureProviderPtr() const
{
    return fp_ptr_;
}

const bool& byte_track::STrack::isActivated() const
{
    return is_activated_;
}

const bool& byte_track::STrack::isOutOfFrame() const
{
    return out_of_frame_;
}

const float& byte_track::STrack::getScore() const
{
    return score_;
}

const size_t& byte_track::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& byte_track::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& byte_track::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& byte_track::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

void byte_track::STrack::activate(const size_t& frame_id, const size_t& track_id)
{
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());

    constexpr bool update_feature = false;
    updateRect(update_feature);

    state_ = STrackState::Tracked;
    if (frame_id == 1)
    {
        is_activated_ = true;
    }
    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void byte_track::STrack::reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    state_ = STrackState::Tracked;
    is_activated_ = true;
    if (0 <= new_track_id)
    {
        track_id_ = new_track_id;
    }
    frame_id_ = frame_id;
    tracklet_len_ = 0;

    score_ = new_track.getScore();
    fp_ptr_ = new_track.getFeatureProviderPtr();

    constexpr bool update_feature = true;
    updateRect(update_feature);
}

void byte_track::STrack::predict()
{
    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;
    }
    kalman_filter_.predict(mean_, covariance_);

    const bool update_feature = (state_ == STrackState::Tracked);
    updateRect(update_feature);
}

void byte_track::STrack::update(const STrack &new_track, const size_t &frame_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    state_ = STrackState::Tracked;
    is_activated_ = true;
    frame_id_ = frame_id;
    tracklet_len_++;

    score_ = new_track.getScore();
    fp_ptr_ = new_track.getFeatureProviderPtr();

    constexpr bool update_feature = true;
    updateRect(update_feature);
}

void byte_track::STrack::markAsLost()
{
    state_ = STrackState::Lost;
}

void byte_track::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void byte_track::STrack::updateRect(const bool &update_feature)
{
    rect_.width() = mean_[2] * mean_[3];
    rect_.height() = mean_[3];
    rect_.x() = mean_[0] - rect_.width() / 2;
    rect_.y() = mean_[1] - rect_.height() / 2;

    const auto &im_width = fp_ptr_->getImageWidth();
    const auto &im_height = fp_ptr_->getImageHeight();
    if (rect_.x() < 0 || im_width <= rect_.x() + rect_.width() || rect_.y() < 0 || im_height <= rect_.y() + rect_.height())
    {
        Rect<float> new_rect;
        new_rect.x() = std::max(0.0f, rect_.x());
        new_rect.y() = std::max(0.0f, rect_.y());
        new_rect.width() = (rect_.x() < 0) ? rect_.width() + rect_.x() : std::min(im_width - new_rect.x() - 1, rect_.width());
        new_rect.height() = (rect_.y() < 0) ? rect_.height() + rect_.y() : std::min(im_height - new_rect.y() - 1, rect_.height());
        mean_[0] = new_rect.x() + new_rect.width() / 2;
        mean_[1] = new_rect.y() + new_rect.height() / 2;
        mean_[2] = new_rect.width() / new_rect.height();
        mean_[3] = new_rect.height();
        mean_[4] = mean_[4] * (new_rect.width() / rect_.width());
        mean_[5] = mean_[5] * (new_rect.height() / rect_.height());
        mean_[6] = 0.0f;
        mean_[7] = 0.0f;
        rect_ = new_rect;
        out_of_frame_ = true;
    }
    if (update_feature)
    {
        updateFeature();
    }
}

void byte_track::STrack::updateFeature()
{
    const auto valid_rect = Rect<float>(
        rect_.x() + rect_.width() * rect_h_padding_ratio_,
        rect_.y() + rect_.height() * rect_v_padding_ratio_,
        rect_.width() * (1 - rect_h_padding_ratio_ - rect_h_padding_ratio_),
        rect_.height() * (1 - rect_v_padding_ratio_ - rect_v_padding_ratio_)
    );
    const auto block_width = valid_rect.width() / block_h_size_;
    const auto block_height = valid_rect.height() / block_v_size_;

    std::vector<Rect<float>> blocks;
    for (size_t ri = 0; ri < block_v_size_; ri++)
    {
        for (size_t ci = 0; ci < block_h_size_; ci++)
        {
            blocks.emplace_back(
                rect_.x() + ci * block_width,
                rect_.y() + ri * block_height,
                block_width,
                block_height
            );
        }
    }

    auto lbp_feature_itr = lbp_feature_.begin();
    auto hue_feature_itr = hue_feature_.begin();
    auto saturation_feature_itr = saturation_feature_.begin();
    for (const auto &block : blocks)
    {
        const auto update_feature = [](auto &features_itr, const auto &features) -> void
        {
            for (const auto &feature : features)
            {
                *features_itr = (feature + *features_itr) / 2;
                features_itr++;
            }
        };

        const auto block_lbp_feature = fp_ptr_->getLbpFeature(block);
        update_feature(lbp_feature_itr, block_lbp_feature);

        const auto [block_hue_feature, block_saturation_feature] = fp_ptr_->getColorFeature(block);
        update_feature(hue_feature_itr, block_hue_feature);
        update_feature(saturation_feature_itr, block_saturation_feature);
    }
}

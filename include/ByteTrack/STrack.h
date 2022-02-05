#pragma once

#include <memory>

#include <ByteTrack/Rect.h>
#include <ByteTrack/KalmanFilter.h>
#include <ByteTrack/FeatureProvider.h>

namespace byte_track
{
enum class STrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3,
};

class STrack
{
public:
    using FeatureProviderPtr = std::shared_ptr<FeatureProvider>;

    STrack(const Rect<float>& rect, const float& score, const FeatureProviderPtr &fp_ptr);
    ~STrack();

    const Rect<float>& getRect() const;
    const STrackState& getSTrackState() const;

    const std::vector<float>& getLBPFeature() const;
    const std::vector<float>& getHueFeature() const;
    const std::vector<float>& getSaturationFeature() const;

    const FeatureProviderPtr& getFeatureProviderPtr() const;

    void setFeatureProviderPtr(const FeatureProviderPtr& fp_ptr);

    const bool& isActivated() const;
    const float& getScore() const;
    const size_t& getTrackId() const;
    const size_t& getFrameId() const;
    const size_t& getStartFrameId() const;
    const size_t& getTrackletLength() const;

    void activate(const size_t& frame_id, const size_t& track_id);
    void reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id = -1);

    void predict();
    void update(const STrack &new_track, const size_t &frame_id);

    void markAsLost();
    void markAsRemoved();

private:
    KalmanFilter kalman_filter_;
    KalmanFilter::StateMean mean_;
    KalmanFilter::StateCov covariance_;

    Rect<float> rect_;
    STrackState state_;

    FeatureProviderPtr fp_ptr_;
    std::vector<float> lbp_feature_;
    std::vector<float> hue_feature_;
    std::vector<float> saturation_feature_;

    bool is_activated_;
    float score_;
    size_t track_id_;
    size_t frame_id_;
    size_t start_frame_id_;
    size_t tracklet_len_;

    void updateRect();
};
}
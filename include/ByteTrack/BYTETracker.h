#pragma once

#include <map>
#include <memory>

#include <ByteTrack/FeatureProvider.h>
#include <ByteTrack/STrack.h>
#include <ByteTrack/lapjv.h>
#include <ByteTrack/Object.h>

namespace byte_track
{
class BYTETracker
{
public:
    using STrackPtr = std::shared_ptr<STrack>;

    BYTETracker(const int& frame_rate = 30,
                const int& track_buffer = 30,
                const float& track_thresh = 0.5,
                const float& high_thresh = 0.6,
                const float& match_thresh = 0.8);
    ~BYTETracker();

    std::vector<STrackPtr> update(const std::vector<Object> &objects, const FeatureProvider &fp);

private:
    std::vector<STrackPtr> jointStracks(const std::vector<STrackPtr> &a_tlist,
                                        const std::vector<STrackPtr> &b_tlist) const;

    std::vector<STrackPtr> subStracks(const std::vector<STrackPtr> &a_tlist,
                                      const std::vector<STrackPtr> &b_tlist) const;

    void removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                const std::vector<STrackPtr> &b_stracks,
                                std::vector<STrackPtr> &a_res,
                                std::vector<STrackPtr> &b_res) const;

    std::vector<std::vector<float>> calcFirstCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                        const std::vector<STrack> &a_tracks_prev,
                                                        const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcSecondCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                         const std::vector<STrackPtr> &b_tracks) const;

    void linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                          const int &cost_matrix_size,
                          const int &cost_matrix_size_size,
                          const float &thresh,
                          std::vector<std::vector<int>> &matches,
                          std::vector<int> &b_unmatched,
                          std::vector<int> &a_unmatched) const;

    std::vector<std::vector<float>> calcLBPCostMatrix(const std::vector<STrack> &a_tracks,
                                                      const std::vector<STrackPtr> &b_tracks) const;

    float calcCosSimilarity(const std::vector<float> &a, const std::vector<float> &b) const;

    std::vector<std::vector<float>> calcIoUCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                      const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>> &a_rect,
                                             const std::vector<Rect<float>> &b_rect) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = LONG_MAX,
                     bool return_cost = true) const;

private:
    const float track_thresh_;
    const float high_thresh_;
    const float match_thresh_;
    const size_t max_time_lost_;

    size_t frame_id_;
    size_t track_id_count_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;
};
}
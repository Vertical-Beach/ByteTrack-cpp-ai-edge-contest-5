#pragma once

#include <map>
#include <memory>

#include <ByteTrack/FeatureProvider.h>
#include <ByteTrack/STrack.h>
#include <ByteTrack/lapjv.h>
#include <ByteTrack/Object.h>

namespace byte_track
{

struct BYTETrackerCfg
{
    size_t len_lost_time {5};
    float track_thr {0.5f};
    float high_thr {0.6f};

    float first_match_thr {0.8f};
    float second_match_thr {0.5f};
    float unconfirmed_match_thr {0.7f};

    float max_area_retio {2.0f};
    float min_appearance_cost {0.6f};
    float min_appearance_cost_for_lost_track {0.4f};
    float min_appearance_cost_for_stray_rect {0.1f};

    size_t max_len_dist_cost {3};
    size_t max_len_appearance_cost {3};
    float start_cost_dist_cost {0.9f};
    float start_cost_appearance_cost {0.95f};
    float step_dist_cost {0.1f};
    float step_appearance_cost {0.05f};

    float appearance_lbp_weight {0.2f};
    float appearance_hue_weight {0.5f};
    float appearance_saturation_weight {0.3f};

    float dist_cost_max_pix {300.0f};

    bool remove_duplicate_stracks {false};
};

class BYTETracker
{
public:
    using Cfg = BYTETrackerCfg;
    using STrackPtr = std::shared_ptr<STrack>;

    #ifdef RISCV
    explicit BYTETracker(volatile int* riscv_dmem_base = NULL, const Cfg &cfg = Cfg());
    #else
    explicit BYTETracker(const Cfg &cfg = Cfg());
    #endif
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

    std::vector<std::vector<std::pair<float, size_t>>> calcAppearanceCostMatrix(const std::vector<STrack> &a_tracks,
                                                                                const std::vector<STrackPtr> &b_tracks) const;

    float calcCosSimilarity(const std::vector<float> &a, const std::vector<float> &b) const;

    std::vector<std::vector<float>> calcIoUCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                      const std::vector<STrackPtr> &b_tracks) const;

    std::vector<std::vector<float>> calcIous(const std::vector<Rect<float>> &a_rect,
                                             const std::vector<Rect<float>> &b_rect) const;

    std::vector<std::vector<std::pair<float, size_t>>> calcDistCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                                          const std::vector<STrackPtr> &b_tracks) const;

    double execLapjv(const std::vector<std::vector<float> > &cost,
                     std::vector<int> &rowsol,
                     std::vector<int> &colsol,
                     bool extend_cost = false,
                     float cost_limit = LONG_MAX,
                     bool return_cost = true) const;

private:
    const Cfg cfg_;

    size_t frame_id_;
    size_t track_id_count_;

    std::vector<STrackPtr> tracked_stracks_;
    std::vector<STrackPtr> lost_stracks_;
    std::vector<STrackPtr> removed_stracks_;
    #ifdef RISCV
    volatile int* riscv_dmem_base;
    #endif
};
}
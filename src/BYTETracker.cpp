#include <ByteTrack/BYTETracker.h>

#ifdef RISCV
byte_track::BYTETracker::BYTETracker(volatile int* riscv_dmem_base, const Cfg &cfg) :
    cfg_(cfg),
    frame_id_(0),
    track_id_count_(0),
    riscv_dmem_base(riscv_dmem_base)
{
}
#else
byte_track::BYTETracker::BYTETracker(const Cfg &cfg) :
    cfg_(cfg),
    frame_id_(0),
    track_id_count_(0)
{
}
#endif

byte_track::BYTETracker::~BYTETracker()
{
}

std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::update(const std::vector<Object>& objects, const FeatureProvider &fp)
{
    ////////////////// Step 1: Get detections //////////////////
    frame_id_++;

    // Create new STracks using the result of object detection
    std::vector<STrackPtr> det_stracks;
    std::vector<STrackPtr> det_low_stracks;

    const auto fp_ptr = std::make_shared<FeatureProvider>(fp);
    for (const auto &object : objects)
    {
        const auto strack = std::make_shared<STrack>(
            object.rect,
            object.prob,
            fp_ptr,
            cfg_.appearance_rect_h_padding_ratio,
            cfg_.appearance_rect_v_padding_ratio,
            cfg_.appearance_block_h_size,
            cfg_.appearance_block_v_size
        );

        if (cfg_.track_high_thr < object.prob)
        {
            det_stracks.push_back(strack);
        }
        else if (cfg_.track_low_thr < object.prob)
        {
            det_low_stracks.push_back(strack);
        }
    }

    // Create lists of existing STrack
    std::vector<STrackPtr> active_stracks;
    std::vector<STrackPtr> non_active_stracks;
    std::vector<STrackPtr> strack_pool;

    for (const auto& tracked_strack : tracked_stracks_)
    {
        if (!tracked_strack->isActivated())
        {
            non_active_stracks.push_back(tracked_strack);
        }
        else
        {
            active_stracks.push_back(tracked_strack);
        }
    }

    strack_pool = jointStracks(active_stracks, lost_stracks_);

    // Predict current pose by KF
    std::vector<STrack> strack_pool_prev;
    for (auto &strack : strack_pool)
    {
        strack_pool_prev.push_back(*strack);
        strack->predict();
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    std::vector<STrackPtr> current_tracked_stracks;
    std::vector<STrackPtr> remain_tracked_stracks;
    std::vector<STrackPtr> remain_det_stracks;
    std::vector<STrackPtr> refind_stracks;

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_detection_idx, unmatch_track_idx;

        const auto dists = calcFirstCostMatrix(strack_pool, strack_pool_prev, det_stracks);
        linearAssignment(dists, strack_pool.size(), det_stracks.size(), cfg_.first_match_thr,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto &match_idx : matches_idx)
        {
            const auto track = strack_pool[match_idx[0]];
            const auto det = det_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                track->update(*det, frame_id_);
                current_tracked_stracks.push_back(track);
            }
            else
            {
                track->reActivate(*det, frame_id_);
                refind_stracks.push_back(track);
            }
        }

        for (const auto &unmatch_idx : unmatch_detection_idx)
        {
            remain_det_stracks.push_back(det_stracks[unmatch_idx]);
        }

        for (const auto &unmatch_idx : unmatch_track_idx)
        {
            if (strack_pool[unmatch_idx]->getSTrackState() == STrackState::Tracked)
            {
                remain_tracked_stracks.push_back(strack_pool[unmatch_idx]);
            }
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    std::vector<STrackPtr> current_lost_stracks;

    {
        std::vector<std::vector<int>> matches_idx;
        std::vector<int> unmatch_track_idx, unmatch_detection_idx;

        const auto dists = calcSecondCostMatrix(remain_tracked_stracks, det_low_stracks);
        linearAssignment(dists, remain_tracked_stracks.size(), det_low_stracks.size(), cfg_.second_match_thr,
                         matches_idx, unmatch_track_idx, unmatch_detection_idx);

        for (const auto &match_idx : matches_idx)
        {
            const auto track = remain_tracked_stracks[match_idx[0]];
            const auto det = det_low_stracks[match_idx[1]];
            if (track->getSTrackState() == STrackState::Tracked)
            {
                track->update(*det, frame_id_);
                current_tracked_stracks.push_back(track);
            }
            else
            {
                track->reActivate(*det, frame_id_);
                refind_stracks.push_back(track);
            }
        }

        for (const auto &unmatch_track : unmatch_track_idx)
        {
            const auto track = remain_tracked_stracks[unmatch_track];
            if (track->getSTrackState() != STrackState::Lost)
            {
                track->markAsLost();
                current_lost_stracks.push_back(track);
            }
        }
    }

    ////////////////// Step 4: Init new stracks //////////////////
    std::vector<STrackPtr> current_removed_stracks;

    {
        std::vector<int> unmatch_detection_idx;
        std::vector<int> unmatch_unconfirmed_idx;
        std::vector<std::vector<int>> matches_idx;

        // Deal with unconfirmed tracks, usually tracks with only one beginning frame
        const auto dists = calcIoUCostMatrix(non_active_stracks, remain_det_stracks);
        linearAssignment(dists, non_active_stracks.size(), remain_det_stracks.size(), cfg_.unconfirmed_match_thr,
                         matches_idx, unmatch_unconfirmed_idx, unmatch_detection_idx);

        for (const auto &match_idx : matches_idx)
        {
            non_active_stracks[match_idx[0]]->update(*remain_det_stracks[match_idx[1]], frame_id_);
            current_tracked_stracks.push_back(non_active_stracks[match_idx[0]]);
        }

        for (const auto &unmatch_idx : unmatch_unconfirmed_idx)
        {
            const auto track = non_active_stracks[unmatch_idx];
            track->markAsRemoved();
            current_removed_stracks.push_back(track);
        }

        // Add new stracks
        for (const auto &unmatch_idx : unmatch_detection_idx)
        {
            const auto track = remain_det_stracks[unmatch_idx];
            if (track->getScore() < cfg_.high_thr)
            {
                continue;
            }
            track_id_count_++;
            track->activate(frame_id_, track_id_count_);
            current_tracked_stracks.push_back(track);
        }
    }

    ////////////////// Step 5: Update state //////////////////
    for (const auto &lost_strack : lost_stracks_)
    {
        if (frame_id_ - lost_strack->getFrameId() > cfg_.len_lost_time)
        {
            lost_strack->markAsRemoved();
            current_removed_stracks.push_back(lost_strack);
        }
    }

    tracked_stracks_ = jointStracks(current_tracked_stracks, refind_stracks);
    lost_stracks_ = subStracks(jointStracks(subStracks(lost_stracks_, tracked_stracks_), current_lost_stracks), removed_stracks_);
    removed_stracks_ = jointStracks(removed_stracks_, current_removed_stracks);

    if (cfg_.remove_duplicate_stracks)
    {
        std::vector<STrackPtr> tracked_stracks_out, lost_stracks_out;
        removeDuplicateStracks(tracked_stracks_, lost_stracks_, tracked_stracks_out, lost_stracks_out);
        tracked_stracks_ = tracked_stracks_out;
        lost_stracks_ = lost_stracks_out;
    }

    std::vector<STrackPtr> output_stracks;
    for (auto &track : tracked_stracks_)
    {
        if (track->isActivated())
        {
            output_stracks.push_back(track);
        }
    }

    return output_stracks;
}
std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::jointStracks(const std::vector<STrackPtr> &a_tlist,
                                                                                      const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, int> exists;
    std::vector<STrackPtr> res;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        exists.emplace(a_tlist[i]->getTrackId(), 1);
        res.push_back(a_tlist[i]);
    }
    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (!exists[tid] || exists.count(tid) == 0)
        {
            exists[tid] = 1;
            res.push_back(b_tlist[i]);
        }
    }
    return res;
}

std::vector<byte_track::BYTETracker::STrackPtr> byte_track::BYTETracker::subStracks(const std::vector<STrackPtr> &a_tlist,
                                                                                    const std::vector<STrackPtr> &b_tlist) const
{
    std::map<int, STrackPtr> stracks;
    for (size_t i = 0; i < a_tlist.size(); i++)
    {
        stracks.emplace(a_tlist[i]->getTrackId(), a_tlist[i]);
    }

    for (size_t i = 0; i < b_tlist.size(); i++)
    {
        const int &tid = b_tlist[i]->getTrackId();
        if (stracks.count(tid) != 0)
        {
            stracks.erase(tid);
        }
    }

    std::vector<STrackPtr> res;
    std::map<int, STrackPtr>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it)
    {
        res.push_back(it->second);
    }

    return res;
}

void byte_track::BYTETracker::removeDuplicateStracks(const std::vector<STrackPtr> &a_stracks,
                                                     const std::vector<STrackPtr> &b_stracks,
                                                     std::vector<STrackPtr> &a_res,
                                                     std::vector<STrackPtr> &b_res) const
{
    const auto ious = calcIoUCostMatrix(a_stracks, b_stracks);

    std::vector<std::pair<size_t, size_t>> overlapping_combinations;
    for (size_t i = 0; i < ious.size(); i++)
    {
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            if (ious[i][j] < 0.15)
            {
                overlapping_combinations.emplace_back(i, j);
            }
        }
    }

    std::vector<bool> a_overlapping(a_stracks.size(), false), b_overlapping(b_stracks.size(), false);
    for (const auto &[a_idx, b_idx] : overlapping_combinations)
    {
        const int timep = a_stracks[a_idx]->getFrameId() - a_stracks[a_idx]->getStartFrameId();
        const int timeq = b_stracks[b_idx]->getFrameId() - b_stracks[b_idx]->getStartFrameId();
        if (timep > timeq)
        {
            b_overlapping[b_idx] = true;
        }
        else
        {
            a_overlapping[a_idx] = true;
        }
    }

    for (size_t ai = 0; ai < a_stracks.size(); ai++)
    {
        if (!a_overlapping[ai])
        {
            a_res.push_back(a_stracks[ai]);
        }
    }

    for (size_t bi = 0; bi < b_stracks.size(); bi++)
    {
        if (!b_overlapping[bi])
        {
            b_res.push_back(b_stracks[bi]);
        }
    }
}

std::vector<std::vector<float>> byte_track::BYTETracker::calcFirstCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                                             const std::vector<STrack> &a_tracks_prev,
                                                                             const std::vector<STrackPtr> &b_tracks) const
{
    if (a_tracks.size() != a_tracks_prev.size())
    {
        throw std::runtime_error("The size of a_tracks and a_tracks_prev are different in BYTETracker::calcFirstCostMatrix()");
    }

    if (a_tracks.size() == 0 || b_tracks.size() == 0)
    {
        return std::vector<std::vector<float>>();
    }

    const auto iou_cost = calcIoUCostMatrix(a_tracks, b_tracks);
    const auto dist_cost = calcDistCostMatrix(a_tracks, b_tracks);
    const auto appearance_cost = calcAppearanceCostMatrix(a_tracks_prev, b_tracks);

    std::vector<std::vector<float>> cost_matrix(a_tracks.size(), std::vector<float>(b_tracks.size()));
    const auto height = a_tracks[0]->getFeatureProviderPtr()->getImageHeight();
    const auto width = a_tracks[0]->getFeatureProviderPtr()->getImageHeight();
    for (size_t ai = 0; ai < a_tracks.size(); ai++)
    {
        /*
        const auto &a_rect = a_tracks[ai]->getRect();
        const auto &a_prev_rect = a_tracks_prev[ai].getRect();
        */
        const auto &a_rect = a_tracks[ai]->getRect();
        const auto a_rect_cx = a_rect.x() + a_rect.width() / 2;
        const auto a_rect_cy = a_rect.y() + a_rect.height() / 2;
        const auto stray_a_rect = (a_rect_cx < 0 || a_rect_cy < 0 || width <= a_rect_cx || height <= a_rect_cy);
        for (size_t bi = 0; bi < b_tracks.size(); bi++)
        {
            /*
            const auto &b_rect = b_tracks[bi]->getRect();
            const auto draw_rect = [](cv::Mat &image, const cv::Rect2i &rect, const cv::Scalar &color, const std::string &label)
            {
                cv::rectangle(image, rect.tl(), rect.br(), color, 5);
                cv::putText(image, label, cv::Point(rect.tl().x, rect.tl().y - 10), cv::FONT_HERSHEY_PLAIN, 1, color, 1, cv::LINE_AA);
            };
            const auto &fp_ptr = b_tracks[bi]->getFeatureProviderPtr();
            cv::Mat scaled_image = fp_ptr->getScaledImage().clone();
            draw_rect(scaled_image, fp_ptr->rect2ScaledRect2i(a_rect), cv::Scalar(255, 0, 0), "Prediction");
            draw_rect(scaled_image, fp_ptr->rect2ScaledRect2i(a_prev_rect), cv::Scalar(0, 255, 0), "Base");
            draw_rect(scaled_image, fp_ptr->rect2ScaledRect2i(b_rect), cv::Scalar(0, 0, 255), "Detection");
            std::cout << "ai: " << ai << ", bi: " << bi << std::endl;
            cv::imshow("scaled_image", scaled_image);
            cv::waitKey(0);
            */
            const auto &b_rect = b_tracks[bi]->getRect();
            const auto area_retio = a_rect.area() / b_rect.area();

            if (cfg_.max_area_retio < area_retio || cfg_.min_appearance_cost < appearance_cost[ai][bi].first)
            {
                cost_matrix[ai][bi] = 1;
            }
            else if (a_tracks[ai]->getSTrackState() == STrackState::Lost && cfg_.min_appearance_cost_for_lost_track < appearance_cost[ai][bi].first)
            {
                cost_matrix[ai][bi] = 1;
            }
            else if (stray_a_rect && cfg_.min_appearance_cost_for_stray_rect < appearance_cost[ai][bi].first)
            {
                cost_matrix[ai][bi] = 1;
            }
            else
            {
                cost_matrix[ai][bi] = iou_cost[ai][bi];
                if (!stray_a_rect && dist_cost[ai][bi].first != std::numeric_limits<float>::max())
                {
                    const auto &dist_sort_idx = dist_cost[ai][bi].second;
                    if (dist_sort_idx < cfg_.max_len_dist_cost)
                    {
                        cost_matrix[ai][bi] *= cfg_.start_cost_dist_cost - cfg_.step_dist_cost * (cfg_.max_len_dist_cost - dist_sort_idx - 1);
                    }

                    const auto &a_sort_idx = appearance_cost[ai][bi].second;
                    if (a_sort_idx < cfg_.max_len_appearance_cost)
                    {
                        cost_matrix[ai][bi] *= cfg_.start_cost_appearance_cost - cfg_.step_appearance_cost * (cfg_.max_len_appearance_cost - a_sort_idx - 1);
                    }
                }
            }
        }
    }
    return cost_matrix;
}

std::vector<std::vector<float>> byte_track::BYTETracker::calcSecondCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                                              const std::vector<STrackPtr> &b_tracks) const
{
    if (a_tracks.size() == 0 || b_tracks.size() == 0)
    {
        return std::vector<std::vector<float>>();
    }

    const auto cost_matrix = calcIoUCostMatrix(a_tracks, b_tracks);

    return cost_matrix;
}

void byte_track::BYTETracker::linearAssignment(const std::vector<std::vector<float>> &cost_matrix,
                                               const int &cost_matrix_size,
                                               const int &cost_matrix_size_size,
                                               const float &thresh,
                                               std::vector<std::vector<int>> &matches,
                                               std::vector<int> &a_unmatched,
                                               std::vector<int> &b_unmatched) const
{
    if (cost_matrix.size() == 0)
    {
        for (int i = 0; i < cost_matrix_size; i++)
        {
            a_unmatched.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++)
        {
            b_unmatched.push_back(i);
        }
        return;
    }

    std::vector<int> rowsol; std::vector<int> colsol;
    execLapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (size_t i = 0; i < rowsol.size(); i++)
    {
        if (rowsol[i] >= 0)
        {
            std::vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        }
        else
        {
            a_unmatched.push_back(i);
        }
    }

    for (size_t i = 0; i < colsol.size(); i++)
    {
        if (colsol[i] < 0)
        {
            b_unmatched.push_back(i);
        }
    }
}

float byte_track::BYTETracker::calcCosSimilarity(const std::vector<float> &a, const std::vector<float> &b) const
{
    if (a.size() != b.size())
    {
        throw std::runtime_error("The size of vectors are different in byte_track::BYTETracker::calcCosSimilarity(): a.size()" +
                                 std::to_string(a.size()) + ", b.size(): " + std::to_string(b.size()));
    }

    float ab = 0;
    float a_sq = 0;
    float b_sq = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        ab += a[i] * b[i];
        a_sq += a[i] * a[i];
        b_sq += b[i] * b[i];
    }
    return ab / (std::sqrt(a_sq) * std::sqrt(b_sq));
}

std::vector<std::vector<std::pair<float, size_t>>> byte_track::BYTETracker::calcAppearanceCostMatrix(const std::vector<STrack> &a_tracks,
                                                                                                     const std::vector<STrackPtr> &b_tracks) const
{
    std::vector<std::vector<std::pair<float, size_t>>> cost_matrix;
    for (const auto &a_track : a_tracks)
    {
        std::vector<std::pair<float, size_t>> similarity_with_idx;
        const auto &a_lbp_feature = a_track.getLBPFeature();
        const auto &a_hue_feature = a_track.getHueFeature();
        const auto &a_saturation_feature = a_track.getSaturationFeature();

        /*
        const auto &a_rect_prev = a_track.getRect();
        const auto &fp_ptr = a_track.getFeatureProviderPtr();
        cv::Mat scaled_image = fp_ptr->getScaledImage().clone();
        */

        for (size_t bi = 0; bi < b_tracks.size(); bi++)
        {
            const auto &b_track = b_tracks[bi];
            const auto &b_lbp_feature = b_track->getLBPFeature();
            const auto &b_hue_feature = b_track->getHueFeature();
            const auto &b_saturation_feature = b_track->getSaturationFeature();

            const auto lbp_simirality = calcCosSimilarity(a_lbp_feature, b_lbp_feature);
            const auto hue_simirality = calcCosSimilarity(a_hue_feature, b_hue_feature);
            const auto saturation_simirality = calcCosSimilarity(a_saturation_feature, b_saturation_feature);

            /*
            std::cout << "a_lbp_feature: ";
            for (const auto &v : a_lbp_feature)
            {
                std::cout << v << " ";
            }
            std::cout << std::endl;

            std::cout << "b_lbp_feature: ";
            for (const auto &v : b_lbp_feature)
            {
                std::cout << v << " ";
            }
            std::cout << std::endl;

            std::cout << "simirality: " << lbp_simirality << std::endl;

            std::cout << "a_hue_feature: ";
            for (const auto &v : a_hue_feature)
            {
                std::cout << v << " ";
            }
            std::cout << std::endl;

            std::cout << "b_hue_feature: ";
            for (const auto &v : b_hue_feature)
            {
                std::cout << v << " ";
            }
            std::cout << std::endl;

            std::cout << "simirality_hue: " << hue_simirality << std::endl;

            std::cout << "a_saturation_feature: ";
            for (const auto &v : a_saturation_feature)
            {
                std::cout << v << " ";
            }
            std::cout << std::endl;

            std::cout << "b_saturation_feature: ";
            for (const auto &v : b_saturation_feature)
            {
                std::cout << v << " ";
            }
            std::cout << std::endl;

            std::cout << "simirality_saturation: " << saturation_simirality << std::endl;

            const auto draw_rect = [](cv::Mat &image, const cv::Rect2i &rect, const cv::Scalar &color, const std::string &label)
            {
                cv::rectangle(image, rect.tl(), rect.br(), color, 5);
                cv::putText(image, label, cv::Point(rect.tl().x, rect.tl().y - 10), cv::FONT_HERSHEY_PLAIN, 1, color, 1, cv::LINE_AA);
            };
            const auto &b_rect = b_track->getRect();
            const auto a_scaled_rect_prev = fp_ptr->rect2ScaledRect2i(a_rect_prev);
            const auto b_scaled_rect = fp_ptr->rect2ScaledRect2i(b_rect);

            cv::Mat draw_scaled_image = scaled_image.clone();
            draw_rect(draw_scaled_image, a_scaled_rect_prev, cv::Scalar(0, 255, 0), "Base");
            draw_rect(draw_scaled_image, b_scaled_rect, cv::Scalar(0, 0, 255), "Detection");
            cv::namedWindow("scaled_image", cv::WINDOW_NORMAL);
            cv::imshow("scaled_image", draw_scaled_image);
            cv::waitKey(0);
            */
            const auto w_mean = cfg_.appearance_lbp_weight * lbp_simirality +
                                cfg_.appearance_hue_weight * hue_simirality +
                                cfg_.appearance_saturation_weight * saturation_simirality;
            similarity_with_idx.emplace_back(1 - w_mean, bi);
        }

        std::sort(similarity_with_idx.begin(), similarity_with_idx.end());

        std::vector<std::pair<float, size_t>> similarity_with_sort_idx(b_tracks.size());
        for (size_t bi = 0; bi < b_tracks.size(); bi++)
        {
            similarity_with_sort_idx[similarity_with_idx[bi].second] = std::make_pair(similarity_with_idx[bi].first, bi);
        }
        cost_matrix.push_back(std::move(similarity_with_sort_idx));
    }
    return cost_matrix;
}

std::vector<std::vector<float>> byte_track::BYTETracker::calcIous(const std::vector<Rect<float>> &a_rect,
                                                                  const std::vector<Rect<float>> &b_rect) const
{
    std::vector<std::vector<float>> ious;
    if (a_rect.size() * b_rect.size() == 0)
    {
        return ious;
    }

    ious.resize(a_rect.size());
    for (size_t i = 0; i < ious.size(); i++)
    {
        ious[i].resize(b_rect.size());
    }

    for (size_t bi = 0; bi < b_rect.size(); bi++)
    {
        for (size_t ai = 0; ai < a_rect.size(); ai++)
        {
            ious[ai][bi] = b_rect[bi].calcIoU(a_rect[ai]);
        }
    }
    return ious;
}

std::vector<std::vector<float> > byte_track::BYTETracker::calcIoUCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                                            const std::vector<STrackPtr> &b_tracks) const
{
    std::vector<byte_track::Rect<float>> a_rects, b_rects;
    for (size_t i = 0; i < a_tracks.size(); i++)
    {
        a_rects.push_back(a_tracks[i]->getRect());
    }

    for (size_t i = 0; i < b_tracks.size(); i++)
    {
        b_rects.push_back(b_tracks[i]->getRect());
    }

    const auto ious = calcIous(a_rects, b_rects);

    std::vector<std::vector<float>> cost_matrix;
    for (size_t i = 0; i < ious.size(); i++)
    {
        std::vector<float> iou;
        for (size_t j = 0; j < ious[i].size(); j++)
        {
            iou.push_back(1 - ious[i][j]);
        }
        cost_matrix.push_back(iou);
    }

    return cost_matrix;
}

std::vector<std::vector<std::pair<float, size_t>>> byte_track::BYTETracker::calcDistCostMatrix(const std::vector<STrackPtr> &a_tracks,
                                                                                               const std::vector<STrackPtr> &b_tracks) const
{
    std::vector<std::vector<std::pair<float, size_t>>> cost_matrix;
    for (const auto &a_track : a_tracks)
    {
        std::vector<std::pair<float, size_t>> distances_with_idx;
        const auto kf_mean = a_track->getKFStateMean();
        const auto a_xc = kf_mean[0];
        const auto a_yc = kf_mean[1];
        const auto a_xv = kf_mean[4];
        const auto a_yv = kf_mean[5];
        const auto a_v_norm = std::sqrt(a_xv * a_xv + a_yv * a_yv);
        const auto a_rect_short_side_len = std::max(a_track->getRect().width(), a_track->getRect().height());
        const auto dist_near_pix = std::min(cfg_.dist_cost_max_pix, a_rect_short_side_len / 2);
        for (size_t bi = 0; bi < b_tracks.size(); bi++)
        {
            const auto &b_rect = b_tracks[bi]->getRect();
            const auto b_xc = b_rect.x() + b_rect.width() / 2;
            const auto b_yc = b_rect.y() + b_rect.height() / 2;
            const auto b_xv = b_xc - a_xc;
            const auto b_yv = b_yc - a_yc;
            const auto b_v_norm = std::sqrt(b_xv * b_xv + b_yv * b_yv);
            const auto dist = std::sqrt((a_xc - b_xc) * (a_xc - b_xc) + (a_yc - b_yc) * (a_yc - b_yc));
            if (dist == 0)
            {
                distances_with_idx.emplace_back(dist, bi);
            }
            else
            {
                const auto cos = (a_xv * b_xv + a_yv * b_yv) / (a_v_norm * b_v_norm);
                if ((dist < dist_near_pix) ||
                    ((a_xv != 0 && a_yv != 0) && 0 < cos && dist < cfg_.dist_cost_max_pix) ||
                    ((a_xv == 0 && a_yv == 0) && dist < cfg_.dist_cost_max_pix))
                {
                    distances_with_idx.emplace_back(dist, bi);
                }
                else
                {
                    distances_with_idx.emplace_back(std::numeric_limits<float>::max(), bi);
                }
            }
        }

        std::sort(distances_with_idx.begin(), distances_with_idx.end());

        std::vector<std::pair<float, size_t>> distances_with_sort_idx(b_tracks.size());
        for (size_t bi = 0; bi < b_tracks.size(); bi++)
        {
            distances_with_sort_idx[distances_with_idx[bi].second] = std::make_pair(distances_with_idx[bi].first, bi);
        }
        cost_matrix.push_back(std::move(distances_with_sort_idx));
    }
    return cost_matrix;
}

double byte_track::BYTETracker::execLapjv(const std::vector<std::vector<float>> &cost,
                                          std::vector<int> &rowsol,
                                          std::vector<int> &colsol,
                                          bool extend_cost,
                                          float cost_limit,
                                          bool return_cost) const
{
    std::vector<std::vector<float> > cost_c;
    cost_c.assign(cost.begin(), cost.end());

    std::vector<std::vector<float> > cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols)
    {
        n = n_rows;
    }
    else
    {
        if (!extend_cost)
        {
            throw std::runtime_error("The `extend_cost` variable should set True");
        }
    }

    if (extend_cost || cost_limit < LONG_MAX)
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (size_t i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX)
        {
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (size_t i = 0; i < cost_c.size(); i++)
            {
                for (size_t j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (size_t i = 0; i < cost_c_extended.size(); i++)
            {
                for (size_t j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (size_t i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (size_t j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[sizeof(double *) * n];
    for (int i = 0; i < n; i++)
        cost_ptr[i] = new double[sizeof(double) * n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }


    int ret;
    int* x_c = new int[sizeof(int) * n];
    int *y_c = new int[sizeof(int) * n];
    #ifdef RISCV
    ret = lapjv_internal(n, cost_ptr, x_c, y_c, riscv_dmem_base);
    #else
    ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    #endif
    if (ret != 0)
    {
        throw std::runtime_error("The result of lapjv_internal() is invalid.");
    }

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++)
        {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++)
        {
            colsol[i] = y_c[i];
        }

        if (return_cost)
        {
            for (size_t i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1)
                {
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    }
    else if (return_cost)
    {
        for (size_t i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++)
    {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}

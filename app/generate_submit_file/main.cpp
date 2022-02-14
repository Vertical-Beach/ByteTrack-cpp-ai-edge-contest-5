#include <ByteTrack/BYTETracker.h>

#include <opencv2/opencv.hpp>

#include "json11/json11.hpp"

#include <filesystem>
#include <regex>

namespace
{
    void write_video(const std::vector<cv::Mat> &images, const std::filesystem::path &path, const size_t &fps)
    {
        if (images.size() == 0)
        {
            return;
        }
        cv::VideoWriter writer;
        writer.open(path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, images[0].size());
        for (const auto &image : images)
        {
            writer << image;
        }
    }

    // key: frame id, value: objects
    std::map<size_t, std::vector<byte_track::Object>> read_json(std::filesystem::path jsondir, std::string basename, int cat_index, int im_width, int im_height)
    {
        auto jsonpath = jsondir;
        jsonpath.append(basename + "_detection_result_" + std::to_string(cat_index) + ".json");
        std::cout << jsonpath << std::endl;
        if(!std::filesystem::exists(jsonpath))
        {
            throw std::runtime_error("Could not open the json file");
        }

        std::ifstream ifs(jsonpath.string());
        std::string jsonstr((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        std::string err;
        auto jobj = json11::Json::parse(jsonstr, err);
        ifs.close();

        std::map<size_t, std::vector<byte_track::Object>> inputs_ref;
        for(auto result: jobj["results"].array_items())
        {
            const auto frame_id = std::stoi(result["frame_id"].string_value());
            const auto prob = std::stof(result["prob"].string_value());
            const auto x = std::clamp(std::stof(result["x"].string_value()), 0.F, im_width - 1.F);
            const auto y = std::clamp(std::stof(result["y"].string_value()), 0.F, im_height - 1.F);
            const auto width = std::clamp(std::stof(result["width"].string_value()), 0.F, im_width - x);
            const auto height = std::clamp(std::stof(result["height"].string_value()), 0.F, im_height - y);
            inputs_ref.try_emplace(frame_id, std::vector<byte_track::Object>());
            inputs_ref[frame_id].emplace_back(byte_track::Rect(x, y, width, height), 0, prob);
        }
        return inputs_ref;
    }

    cv::Rect2i get_rounded_rect2i(const byte_track::Rect<float> &rect, const int &im_width, const int &im_height)
    {
        const auto x = std::clamp(static_cast<int>(std::round(rect.x())), 0, im_width - 1);
        const auto y = std::clamp(static_cast<int>(std::round(rect.y())), 0, im_height - 1);
        const auto width = std::clamp(static_cast<int>(std::round(rect.width())), 0, im_width - x);
        const auto height = std::clamp(static_cast<int>(std::round(rect.height())), 0, im_height - y);
        return cv::Rect2i(x, y, width, height);
    }

    void draw_rect(cv::Mat &image, const cv::Rect2i &rect, const size_t &track_id, const std::string &label)
    {
        const auto color = cv::Scalar(37 * (track_id + 3) % 255, 17 * (track_id + 3) % 255, 29 * (track_id + 3) % 255);
        cv::rectangle(image, rect.tl(), rect.br(), color, 5);
        cv::putText(image, label, cv::Point(rect.tl().x, rect.tl().y - 10), cv::FONT_HERSHEY_PLAIN, 1, color, 2, cv::LINE_AA);
    }
}

#ifdef RISCV
#include <sys/mman.h>
#include <fcntl.h>
#include "riscv_imm.h"
#endif
#ifdef DPU
#include "yolo_runner.h"
#endif

int main(int argc, char *argv[])
{
    try
    {
        const std::string usage = "Usage1: $ ./generate_submit_file <video path> json <detection results path>\n"\
                "Usage2: $ ./generate_submit_file <video path> dpu <modelconfig .prototxt> <modelfile .xmodel>";
        if(argc < 3) throw std::runtime_error(usage);

        std::filesystem::path video_path(argv[1]);
        std::string runmode = std::string(argv[2]);
        #ifndef DPU
        if(runmode == "dpu") throw std::runtime_error("build app with -DDPU=ON");
        #endif
        if(runmode == "json")
        {
            if(argc != 4) throw std::runtime_error(usage);
        }
        else if(runmode == "dpu")
        {
            if(argc != 5) throw std::runtime_error(usage);
        }
        else throw std::runtime_error("unknown run_mode, specify dpu or json");

        std::cout << video_path.filename() << std::endl;
        const std::string video_basename = video_path.stem();

        #ifdef DPU
        std::shared_ptr<YoloRunner> yolorunner;
        if(runmode == "dpu")
        {
            char* configfile = argv[3];
            char* modelfile = argv[4];
            yolorunner = std::shared_ptr<YoloRunner>(new YoloRunner(configfile, modelfile));
        }
        #endif

        byte_track::BYTETrackerCfg car_cfg;
        car_cfg.len_lost_time = 7;
        car_cfg.track_high_thr = 0.75f;
        car_cfg.track_low_thr = 0.6f;
        car_cfg.high_thr = 0.7f;
        car_cfg.first_match_thr = 0.8f;
        car_cfg.second_match_thr = 0.6f;
        car_cfg.unconfirmed_match_thr = 0.7f;
        car_cfg.max_area_retio = 2.0f;
        car_cfg.min_appearance_cost = 0.5f;
        car_cfg.min_appearance_cost_for_lost_track = 0.25f;
        car_cfg.min_appearance_cost_for_stray_rect = 0.2f;
        car_cfg.max_len_dist_cost = 5;
        car_cfg.max_len_appearance_cost = 1;
        car_cfg.start_cost_dist_cost = 0.9f;
        car_cfg.start_cost_appearance_cost = 0.3f;
        car_cfg.step_dist_cost = 0.05f;
        car_cfg.step_appearance_cost = 0.0f;
        car_cfg.center_area_horizontal_offset_ratio = 0.25f;
        car_cfg.appearance_rect_h_padding_ratio = 0.0f;
        car_cfg.appearance_rect_v_padding_ratio = 0.0f;
        car_cfg.appearance_block_h_size = 1;
        car_cfg.appearance_block_v_size = 3;
        car_cfg.appearance_lbp_weight = 0.5f;
        car_cfg.appearance_hue_weight = 0.25f;
        car_cfg.appearance_saturation_weight = 0.25f;
        car_cfg.dist_cost_max_pix = 100.0f;

        byte_track::BYTETrackerCfg pedestrian_cfg;
        pedestrian_cfg.len_lost_time = 5;
        pedestrian_cfg.track_high_thr = 0.75f;
        pedestrian_cfg.track_low_thr = 0.4f;
        pedestrian_cfg.high_thr = 0.7f;
        pedestrian_cfg.first_match_thr = 0.9f;
        pedestrian_cfg.second_match_thr = 0.6f;
        pedestrian_cfg.unconfirmed_match_thr = 0.7f;
        pedestrian_cfg.max_area_retio = 1.5f;
        pedestrian_cfg.min_appearance_cost = 0.5f;
        pedestrian_cfg.min_appearance_cost_for_lost_track = 0.2f;
        pedestrian_cfg.min_appearance_cost_for_stray_rect = 0.1f;
        pedestrian_cfg.max_len_dist_cost = 5;
        pedestrian_cfg.max_len_appearance_cost = 2;
        pedestrian_cfg.start_cost_dist_cost = 0.9f;
        pedestrian_cfg.start_cost_appearance_cost = 0.8f;
        pedestrian_cfg.step_dist_cost = 0.1f;
        pedestrian_cfg.step_appearance_cost = 0.2f;
        pedestrian_cfg.center_area_horizontal_offset_ratio = 0.5f;
        pedestrian_cfg.appearance_rect_h_padding_ratio = 0.2f;
        pedestrian_cfg.appearance_rect_v_padding_ratio = 0.1f;
        pedestrian_cfg.appearance_block_h_size = 1;
        pedestrian_cfg.appearance_block_v_size = 2;
        pedestrian_cfg.appearance_lbp_weight = 0.8f;
        pedestrian_cfg.appearance_hue_weight = 0.15f;
        pedestrian_cfg.appearance_saturation_weight = 0.05f;
        pedestrian_cfg.dist_cost_max_pix = 100.0f;

        byte_track::FeatureProviderCfg fp_cfg;
        fp_cfg.scale = 0.2f;
        fp_cfg.n_lbp_feature_hist_bins = 10;
        fp_cfg.n_hue_hist_bins = 10;
        fp_cfg.n_saturation_hist_bins = 5;

        // Get detection results
        #ifdef RISCV
        int uio0_fd = open("/dev/uio0", O_RDWR | O_SYNC);
        volatile int* riscv_dmem_base = (int*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio0_fd, 0);
        int uio1_fd = open("/dev/uio1", O_RDWR | O_SYNC);
        volatile unsigned int* riscv_imem_base = (unsigned int*) mmap(NULL, 0x10000, PROT_READ|PROT_WRITE, MAP_SHARED, uio1_fd, 0);
        if(uio0_fd < 0 || uio1_fd < 0){
            std::cerr << "Device Open Failed" << std::endl;
            return -1;
        }
        //write instruction
        riscv_imm(riscv_imem_base);
        auto car_tracker = byte_track::BYTETracker(riscv_dmem_base, car_cfg);
        auto pedestrian_tracker = byte_track::BYTETracker(riscv_dmem_base, pedestrian_cfg);
        #else
        // Execute tracking
        auto car_tracker = byte_track::BYTETracker(car_cfg);
        auto pedestrian_tracker = byte_track::BYTETracker(pedestrian_cfg);
        #endif


        // std::vector<cv::Mat> draw_images;
        std::vector<std::vector<byte_track::STrack>> outputs_car;
        std::vector<std::vector<byte_track::STrack>> outputs_pedestrian;
        cv::VideoCapture video;
        video.open(video_path.string());
        if (video.isOpened() == false)
        {
            throw std::runtime_error("Could not open the video file: " + video_path.string());
        }

        const auto copy_results = [](const auto tracker_outputs, auto &outputs) -> void
        {
            outputs.emplace_back();
            for (const auto &tracker_output : tracker_outputs)
            {
                // Copy from std::vector<std::shared_ptr<byte_track::STrack>> to std::vector<byte_track::STrack>
                outputs.back().push_back(*tracker_output.get());
            }
        };

        cv::Mat image;
        video >> image;
        if (image.empty())
        {
            throw std::runtime_error("The input video is empty.");
        }

        const size_t width = image.cols;
        const size_t height = image.rows;

        int frame_cnt = 0;
        if (runmode == "json")
        {
            std::filesystem::path jsondir(argv[3]);
            auto json_inputs_car = read_json(jsondir, video_basename, 0, width, height);
            auto json_inputs_pedestrian = read_json(jsondir, video_basename, 1, width, height);

            while (true)
            {
                if (image.empty())
                    break;
                frame_cnt++;
                // draw_images.push_back(image.clone());
                byte_track::FeatureProvider fp(image, fp_cfg);
                copy_results(car_tracker.update(json_inputs_car[frame_cnt - 1], fp), outputs_car);
                copy_results(pedestrian_tracker.update(json_inputs_pedestrian[frame_cnt - 1], fp), outputs_pedestrian);
                video >> image;
            }
        }
        else if (runmode == "dpu")
        {
            #ifdef DPU
            while (true)
            {
                if (image.empty())
                    break;
                frame_cnt++;
                // draw_images.push_back(image.clone());
                byte_track::FeatureProvider fp(image, fp_cfg);
                auto detection_results = yolorunner->Run(image);
                copy_results(car_tracker.update(detection_results[0], fp), outputs_car);
                copy_results(pedestrian_tracker.update(detection_results[1], fp), outputs_pedestrian);
                video >> image;
            }
            #endif
        }

        video.release();

        // Results: vector of vector{track_id, rect}, and the idx means frame_id
        using Results = std::vector<std::vector<std::pair<size_t, cv::Rect2i>>>;

        // Validate tracks
        const auto validate_outputs = [&](const std::vector<std::vector<byte_track::STrack>> &outputs) -> Results
        {
            // track_id -> vector of {frame_id, rect}
            std::map<size_t, std::vector<std::pair<size_t, cv::Rect2i>>> map;

            // Encode from std::vector<byte_track::STrack> to std::map<size_t, std::vector<std::pair<size_t, byte_track::Rect<float>>>>
            for (size_t fi = 0; fi < outputs.size(); fi++)
            {
                for (size_t oi = 0; oi < outputs[fi].size(); oi++)
                {
                    const auto &strack = outputs[fi][oi];
                    const auto rect2i = get_rounded_rect2i(strack.getRect(), width, height);
                    if (1024 <= rect2i.area())
                    {
                        map.try_emplace(strack.getTrackId(), std::vector<std::pair<size_t, cv::Rect2i>>());
                        map[strack.getTrackId()].emplace_back(fi, rect2i);
                    }
                }
            }

            // Validate
            Results result(frame_cnt);
            for (const auto &[track_id, stracks] : map)
            {
                if (stracks.size() < 3)
                {
                    continue;
                }
                for (const auto &[frame_id, rect] : stracks)
                {
                    result[frame_id].emplace_back(track_id, rect);
                }
            }
            return result;
        };

        auto results_car = validate_outputs(outputs_car);
        auto results_pedestrian = validate_outputs(outputs_pedestrian);
        // Generate submit file
        std::vector<json11::Json> frame_objects;
        for (int fi = 0; fi < frame_cnt; fi++)
        {
            //key: category
            //value: lists of tracked objects
            std::map<std::string, std::vector<json11::Json>> objs_for_category_map;
            const auto gen_objs_pt_and_draw_rect = [&](Results &results,
                                                        const std::string &name) -> void
            {
                std::vector<json11::Json> objs_jobj;
                for (const auto &[track_id, rect] : results[fi])
                {
                    auto jobj = json11::Json::object();
                    jobj["id"] = (int)track_id;
                    jobj["box2d"] = json11::Json::array({rect.tl().x, rect.tl().y, rect.br().x, rect.br().y});
                    objs_jobj.push_back(jobj);
                    // draw_rect(draw_images[fi], rect, track_id, name.substr(0, 1));
                }
                if (objs_jobj.size() != 0)
                {
                    objs_for_category_map[name] = objs_jobj;
                }
            };

            gen_objs_pt_and_draw_rect(results_car, "Car");
            gen_objs_pt_and_draw_rect(results_pedestrian, "Pedestrian");
            frame_objects.push_back(json11::Json(objs_for_category_map));
        }
        // frame_objects_map[video_path.filename().string()] = json11::Json(frame_objects);
        std::ofstream file;
        file.open((std::string)"prediction_" + (std::string)video_path.stem() + (std::string)".json");
        file << json11::Json(frame_objects).dump();
        file.close();

        // write_video(draw_images, (std::string)"./" + (std::string)video_path.stem() + (std::string)".mp4", 5);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
    }
}

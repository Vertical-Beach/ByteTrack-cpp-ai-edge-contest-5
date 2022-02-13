#include <ByteTrack/BYTETracker.h>

#include <opencv2/opencv.hpp>

#include "json11/json11.hpp"

#include <chrono>
#include <filesystem>
#include <regex>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef RISCV
#include <sys/mman.h>
#include <fcntl.h>
#include "riscv_imm.h"
#endif
#ifdef DPU
#include "yolo_runner.h"
#endif

namespace
{
    template<typename T>
    class ObjWithMtx {
    public:
        T obj;

        ObjWithMtx() = default;
        ObjWithMtx(const T& _obj) : obj(_obj) {}

        void operator =(const ObjWithMtx& obj) = delete;
        ObjWithMtx(const ObjWithMtx& obj) = delete;

        void lock() { mtx_.lock(); }
        bool try_lock() { return mtx_.try_lock(); }
        void unlock() { mtx_.unlock(); }

    private:
        std::mutex mtx_;
    };

    template<typename T, size_t D>
    class MultiThreadFIFO {
    public:
        explicit MultiThreadFIFO(const uint32_t& sleep_t_us = 100) :
            sleep_t_us_(sleep_t_us) {
            init();
        }

        void operator =(const MultiThreadFIFO& obj) = delete;
        MultiThreadFIFO(const MultiThreadFIFO& obj) = delete;

        void init() {
            std::unique_lock<std::mutex> lock_w_func(w_func_guard_, std::try_to_lock);
            std::unique_lock<std::mutex> lock_r_func(r_func_guard_, std::try_to_lock);
            if (!lock_w_func.owns_lock() || !lock_r_func.owns_lock()) {
                throw std::runtime_error("[ERROR] Initialization of the FIFO failed.");
            }
            for (auto& state : fifo_state_) {
                std::lock_guard<ObjWithMtx<ElementState>> lock_state(state);
                state.obj = ElementState::INVALID;
            }
            r_idx_ = 0;
            w_idx_ = 0;
        }

        void write(ObjWithMtx<bool>& no_abnormality, const bool& is_last, std::function<void(T&)> write_func) {
            std::unique_lock<std::mutex> lock_w_func(w_func_guard_, std::try_to_lock);
            if (!lock_w_func.owns_lock()) {
                throw std::runtime_error("[ERROR] The write function can't be called at the same time from multiple threads.");
            }
        while (true) {
                {
                    std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[w_idx_]);
                    if (fifo_state_[w_idx_].obj == ElementState::INVALID) {
                        break;
                    } else {
                        std::lock_guard<ObjWithMtx<bool>> lock(no_abnormality);
                        if (no_abnormality.obj == false) {
                            throw std::runtime_error("[ERROR] Terminate write process.");
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_t_us_));
            }
            {
                std::lock_guard<ObjWithMtx<T>> lock_fifo(fifo_[w_idx_]);
                write_func(fifo_[w_idx_].obj);
            }
            {
                std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[w_idx_]);
                fifo_state_[w_idx_].obj = is_last ? ElementState::VALID_LAST : ElementState::VALID;
            }
            incrementIdx(w_idx_);
        }

        void read(ObjWithMtx<bool>& no_abnormality, std::function<void(const T&)> read_func) {
            std::unique_lock<std::mutex> lock_r_func(r_func_guard_, std::try_to_lock);
            if (!lock_r_func.owns_lock()) {
                throw std::runtime_error("[ERROR] The read function can't be called at the same time from multiple threads.");
            }
            while (true) {
                {
                    std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[r_idx_]);
                    if (fifo_state_[r_idx_].obj == ElementState::VALID ||
                        fifo_state_[r_idx_].obj == ElementState::VALID_LAST) {
                        break;
                    } else {
                        std::lock_guard<ObjWithMtx<bool>> lock(no_abnormality);
                        if (no_abnormality.obj == false) {
                            throw std::runtime_error("[ERROR] Terminate read process.");
                        }
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_t_us_));
            }
            {
                std::lock_guard<ObjWithMtx<T>> lock_fifo(fifo_[r_idx_]);
                read_func(fifo_[r_idx_].obj);
            }
            {
                std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[r_idx_]);
                if (fifo_state_[r_idx_].obj == ElementState::VALID) {
                    fifo_state_[r_idx_].obj = ElementState::INVALID;
                    incrementIdx(r_idx_);
                } else {
                    fifo_state_[r_idx_].obj = ElementState::INVALID_LAST;
                }
            }
        }

        bool neverReadNextElement() {
            std::unique_lock<std::mutex> lock_r_func(r_func_guard_, std::try_to_lock);
            if (!lock_r_func.owns_lock()) {
                throw std::runtime_error("[ERROR] The read function can't be called at the same time from multiple threads.");
            }
            std::lock_guard<ObjWithMtx<ElementState>> lock_state(fifo_state_[r_idx_]);
            return (fifo_state_[r_idx_].obj == ElementState::INVALID_LAST);
        }

    private:
        enum class ElementState { VALID, VALID_LAST, INVALID, INVALID_LAST };

        void incrementIdx(size_t& idx) const {
            idx = (idx < D - 1) ? idx + 1 : 0;
        }

        const uint32_t sleep_t_us_;

        std::array<ObjWithMtx<T>, D> fifo_;
        std::array<ObjWithMtx<ElementState>, D> fifo_state_;
        std::mutex r_func_guard_, w_func_guard_;
        size_t r_idx_{0}, w_idx_{0};
    };
    ObjWithMtx<bool> no_abnormality(true);
    using InferenceFIFOElementType = std::pair<cv::Mat, std::vector<std::vector<byte_track::Object>>>;
    constexpr auto InferenceFIFO_DEPTH = 30U;
    constexpr auto SLEEP_T_US = 100U;
    std::vector<cv::Mat> video_images;
    MultiThreadFIFO<InferenceFIFOElementType, InferenceFIFO_DEPTH> inference_fifo(SLEEP_T_US);
    std::mutex cout_guard;
    std::map<size_t, std::vector<byte_track::Object>> json_inputs_car;
    std::map<size_t, std::vector<byte_track::Object>> json_inputs_pedestrian;

    #ifdef DPU
    std::shared_ptr<YoloRunner> yolorunner;
    #endif
    std::string runmode;
    float inference_time_sum = 0.0f;
    void do_inference(){
        for(size_t frame_idx = 0; frame_idx < video_images.size(); frame_idx++)
        {
            cv::Mat img = video_images[frame_idx];
            bool end_flag = (frame_idx == (video_images.size() - 1));
            std::chrono::system_clock::time_point t_start, t_end;
            t_start = std::chrono::system_clock::now();
            std::vector<std::vector<byte_track::Object>> detection_results;
            if (runmode == "json")
            {
                detection_results = {json_inputs_car[frame_idx], json_inputs_pedestrian[frame_idx]};
            } else if (runmode == "dpu")
            {
                #ifdef DPU
                detection_results = yolorunner->Run(img);
                #else
                #endif
            }
            t_end = std::chrono::system_clock::now();
            inference_time_sum += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
            inference_fifo.write(no_abnormality, end_flag, [&](InferenceFIFOElementType& dst) -> void {
                dst.first = img;
                dst.second = detection_results;
            });
        }
    }

    std::vector<std::vector<byte_track::STrack>> outputs_car;
    std::vector<std::vector<byte_track::STrack>> outputs_pedestrian;
    std::shared_ptr<byte_track::BYTETracker> car_tracker;
    std::shared_ptr<byte_track::BYTETracker> pedestrian_tracker;

    float tracking_time_sum = 0.0f;
    void do_tracking(){
        bool end_flag = false;
        int frame_idx = 0;

        const auto copy_results = [](const auto tracker_outputs, auto &outputs) -> void
        {
            outputs.emplace_back();
            for (const auto &tracker_output : tracker_outputs)
            {
                // Copy from std::vector<std::shared_ptr<byte_track::STrack>> to std::vector<byte_track::STrack>
                outputs.back().push_back(*tracker_output.get());
            }
        };

        while (!end_flag) {
            cv::Mat img;
            std::vector<std::vector<byte_track::Object>> detection_results;
            inference_fifo.read(no_abnormality, [&](const InferenceFIFOElementType& src) -> void {
                img = src.first;
                detection_results = src.second;
            });
            // auto detection_results = yolorunner->Run(img);
            frame_idx++;
            end_flag = inference_fifo.neverReadNextElement();
            std::chrono::system_clock::time_point t_start, t_end;
            t_start = std::chrono::system_clock::now();
            byte_track::FeatureProvider fp(img);
            copy_results(car_tracker->update(detection_results[0], fp), outputs_car);
            copy_results(pedestrian_tracker->update(detection_results[1], fp), outputs_pedestrian);
            t_end = std::chrono::system_clock::now();
            tracking_time_sum += (float)std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
        }
    }

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



int main(int argc, char *argv[])
{
    try
    {
        std::exception_ptr ep;
        const auto gen_func = [&](const auto do_worker) -> auto {
            return [&]() -> void {
                try {
                    do_worker();
                }
                catch(...) {
                    std::lock_guard<ObjWithMtx<bool>> lock(no_abnormality);
                    if (no_abnormality.obj) {
                        no_abnormality.obj = false;
                        ep = std::current_exception();
                    }
                }
            };
        };

        const std::string usage = "Usage1: $ ./generate_submit_file <video path> json <detection results path>\n"\
                "Usage2: $ ./generate_submit_file <video path> dpu <modelconfig .prototxt> <modelfile .xmodel>";
        if(argc < 3) throw std::runtime_error(usage);

        std::filesystem::path video_path(argv[1]);
        runmode = std::string(argv[2]);
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

        cv::VideoCapture video;
        video.open(video_path.string());
        if (video.isOpened() == false)
        {
            throw std::runtime_error("Could not open the video file: " + video_path.string());
        }

        int frame_cnt = 0;
        cv::Mat image;
        video >> image;
        int width = image.cols;
        int height = image.rows;
        std::cout << "Loading images from video..." << std::endl;
        while(true)
        {
            if (image.empty()) break;
            video_images.push_back(image.clone());
            video >> image;
            frame_cnt++;
        }
        video.release();

        if (runmode == "json")
        {
            std::filesystem::path jsondir(argv[3]);
            json_inputs_car = read_json(jsondir, video_basename, 0, width, height);
            json_inputs_pedestrian = read_json(jsondir, video_basename, 1, width, height);
        }

        byte_track::BYTETrackerCfg car_cfg;
        car_cfg.len_lost_time = 5;
        car_cfg.track_thr = 0.75f;
        car_cfg.high_thr = 0.8f;
        car_cfg.first_match_thr = 0.8f;
        car_cfg.second_match_thr = 0.5f;
        car_cfg.unconfirmed_match_thr = 0.7f;
        car_cfg.max_area_retio = 2.0f;
        car_cfg.min_appearance_cost = 0.4f;
        car_cfg.min_appearance_cost_for_lost_track = 0.2f;
        car_cfg.min_appearance_cost_for_stray_rect = 0.1f;
        car_cfg.max_len_dist_cost = 5;
        car_cfg.max_len_appearance_cost = 1;
        car_cfg.start_cost_dist_cost = 0.9f;
        car_cfg.start_cost_appearance_cost = 0.5f;
        car_cfg.step_dist_cost = 0.1f;
        car_cfg.step_appearance_cost = 0.0f;
        car_cfg.appearance_rect_h_padding_ratio = 0.0f;
        car_cfg.appearance_rect_v_padding_ratio = 0.0f;
        car_cfg.appearance_block_h_size = 1;
        car_cfg.appearance_block_v_size = 3;
        car_cfg.appearance_lbp_weight = 0.3f;
        car_cfg.appearance_hue_weight = 0.5f;
        car_cfg.appearance_saturation_weight = 0.2f;
        car_cfg.dist_cost_max_pix = 200.0f;

        byte_track::BYTETrackerCfg pedestrian_cfg;
        pedestrian_cfg.len_lost_time = 5;
        pedestrian_cfg.track_thr = 0.75f;
        pedestrian_cfg.high_thr = 0.8f;
        pedestrian_cfg.first_match_thr = 0.9f;
        pedestrian_cfg.second_match_thr = 0.5f;
        pedestrian_cfg.unconfirmed_match_thr = 0.7f;
        pedestrian_cfg.max_area_retio = 1.5f;
        pedestrian_cfg.min_appearance_cost = 0.3f;
        pedestrian_cfg.min_appearance_cost_for_lost_track = 0.15f;
        pedestrian_cfg.min_appearance_cost_for_stray_rect = 0.08f;
        pedestrian_cfg.max_len_dist_cost = 5;
        pedestrian_cfg.max_len_appearance_cost = 1;
        pedestrian_cfg.start_cost_dist_cost = 0.9f;
        pedestrian_cfg.start_cost_appearance_cost = 0.5f;
        pedestrian_cfg.step_dist_cost = 0.1f;
        pedestrian_cfg.step_appearance_cost = 0.0f;
        pedestrian_cfg.appearance_rect_h_padding_ratio = 0.2f;
        pedestrian_cfg.appearance_rect_v_padding_ratio = 0.1f;
        pedestrian_cfg.appearance_block_h_size = 1;
        pedestrian_cfg.appearance_block_v_size = 3;
        pedestrian_cfg.appearance_lbp_weight = 0.5f;
        pedestrian_cfg.appearance_hue_weight = 0.4f;
        pedestrian_cfg.appearance_saturation_weight = 0.1f;
        pedestrian_cfg.dist_cost_max_pix = 200.0f;

        byte_track::FeatureProviderCfg fp_cfg;
        fp_cfg.scale = 0.4f;
        fp_cfg.n_lbp_feature_hist_bins = 10;
        fp_cfg.n_hue_hist_bins = 10;
        fp_cfg.n_saturation_hist_bins = 10;

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
        car_tracker = std::shared_ptr<byte_track::BYTETracker>(new byte_track::BYTETracker(riscv_dmem_base=riscv_dmem_base));
        pedestrian_tracker = std::shared_ptr<byte_track::BYTETracker>(new byte_track::BYTETracker(riscv_dmem_base=riscv_dmem_base));
        #else
        car_tracker = std::shared_ptr<byte_track::BYTETracker>(new byte_track::BYTETracker());
        pedestrian_tracker = std::shared_ptr<byte_track::BYTETracker>(new byte_track::BYTETracker());
        #endif


        std::cout << "Start detection and tracking" << std::endl;
        inference_fifo.init();
        std::cout << video_path.filename() << std::endl;

        #ifdef DPU
        if(runmode == "dpu")
        {
            char* configfile = argv[3];
            char* modelfile = argv[4];
            yolorunner = std::shared_ptr<YoloRunner>(new YoloRunner(configfile, modelfile));
        }
        #endif

        std::chrono::system_clock::time_point t_start, t_end;
        t_start = std::chrono::system_clock::now();
        auto inference   = std::thread(gen_func([&]() -> void { do_inference(); }));
        auto tracking = std::thread(gen_func([&]() -> void { do_tracking(); }));
        inference.join();
        tracking.join();
        t_end = std::chrono::system_clock::now();

        float all_time = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
        std::map<std::string, float> time_summary = {
            {"inference", inference_time_sum/(float)frame_cnt},
            {"tracking", tracking_time_sum/(float)frame_cnt},
            {"all", all_time}
        };
        std::ofstream timefile;
        timefile.open((std::string)"time_summary_" + (std::string)video_path.stem() + (std::string)".json");
        timefile << json11::Json(time_summary).dump();
        timefile.close();

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

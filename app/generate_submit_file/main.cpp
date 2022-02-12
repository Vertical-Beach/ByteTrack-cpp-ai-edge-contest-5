#include <ByteTrack/BYTETracker.h>

#include <opencv2/opencv.hpp>

#include "json11/json11.hpp"

#include <filesystem>
#include <regex>

namespace
{
    // return [fps, width, height, num_frames]
    std::tuple<int, int, int, int> get_video_info(const std::filesystem::path &path)
    {
        // cv::VideoCapture video;
        // video.open(path.string());
        // if (video.isOpened() == false)
        // {
        //     throw std::runtime_error("Could not open the video file: " + path.string());
        // }
        // const auto fps = video.get(cv::CAP_PROP_FPS);

        // cv::Mat image;
        // int num_frames = 0;
        // int width = -1;;
        // int height = -1;
        // while(true)
        // {
        //     video >> image;
        //     if (image.empty()) break;
        //     num_frames++;
        //     width = image.cols;
        //     height = image.rows;
        // }
        // video.release();
        int fps = 5;
        int width = 1936;
        int height = 1216;
        int num_frames = 150;
        return std::make_tuple((int)fps, width, height, num_frames);
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

    std::vector<std::vector<byte_track::Object>> read_json(std::filesystem::path jsondir, std::string basename, int cat_index, int im_width, int im_height, int num_frames)
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

        std::vector<std::vector<byte_track::Object>> inputs_ref;
        inputs_ref.resize(num_frames);
        for(auto result: jobj["results"].array_items())
        {
            const auto frame_id = std::stoi(result["frame_id"].string_value());
            const auto prob = std::stof(result["prob"].string_value());
            const auto x = std::clamp(std::stof(result["x"].string_value()), 0.F, im_width - 1.F);
            const auto y = std::clamp(std::stof(result["y"].string_value()), 0.F, im_height - 1.F);
            const auto width = std::clamp(std::stof(result["width"].string_value()), 0.F, im_width - x);
            const auto height = std::clamp(std::stof(result["height"].string_value()), 0.F, im_height - y);
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

        // Get video info
        const auto [fps, width, height, num_frames] = get_video_info(video_path);

        std::vector<std::vector<byte_track::Object>> json_inputs_car, json_inputs_pedestrian;
        if (runmode == "json")
        {
            std::filesystem::path jsondir(argv[3]);
            json_inputs_car = read_json(jsondir, video_basename, 0, width, height, num_frames);
            json_inputs_pedestrian = read_json(jsondir, video_basename, 1, width, height, num_frames);
        }
        #ifdef DPU
        std::shared_ptr<YoloRunner> yolorunner;
        if(runmode == "dpu")
        {
            char* configfile = argv[3];
            char* modelfile = argv[4];
            yolorunner = std::shared_ptr<YoloRunner>(new YoloRunner(configfile, modelfile));
        }
        #endif

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
        byte_track::BYTETracker car_tracker(fps, fps, riscv_dmem_base);
        byte_track::BYTETracker pedestrian_tracker(fps, fps, riscv_dmem_base);
        #else
        // Execute tracking
        byte_track::BYTETracker car_tracker(fps, fps);
        byte_track::BYTETracker pedestrian_tracker(fps, fps);
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

        int fi = 0;
        while(true){
            cv::Mat image;
            video >> image;
            if (image.empty()) break;
            // draw_images.push_back(images[fi].clone());

            const auto copy = [](const auto tracker_outputs, auto &outputs) -> void
            {
                outputs.emplace_back();
                for (const auto &tracker_output : tracker_outputs)
                {
                    // Copy from std::vector<std::shared_ptr<byte_track::STrack>> to std::vector<byte_track::STrack>
                    outputs.back().push_back(*tracker_output.get());
                }
            };

            if (runmode == "json")
            {
                copy(car_tracker.update(json_inputs_car[fi]), outputs_car);
                copy(pedestrian_tracker.update(json_inputs_pedestrian[fi]), outputs_pedestrian);
            }
            else if (runmode == "DPU")
            {
                #ifdef DPU
                auto detection_results = yolorunner->Run(image);
                copy(car_tracker.update(detection_results[0]), outputs_car);
                copy(pedestrian_tracker.update(detection_results[1]), outputs_pedestrian);
                #endif
            }
            fi++;
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
            Results result(num_frames);
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
        for (int fi = 0; fi < num_frames; fi++)
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
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
    }
}

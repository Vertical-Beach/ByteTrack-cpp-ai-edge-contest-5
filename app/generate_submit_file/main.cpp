#include <ByteTrack/BYTETracker.h>

#include <opencv2/opencv.hpp>

#include "json11/json11.hpp"

#include <filesystem>
#include <regex>

namespace
{
    // return [images, fps]
    std::pair<std::vector<cv::Mat>, int> read_video(const std::filesystem::path &path)
    {
        cv::VideoCapture video;
        video.open(path.string());
        if (video.isOpened() == false)
        {
            throw std::runtime_error("Could not open the video file: " + path.string());
        }
        const auto fps = video.get(cv::CAP_PROP_FPS);

        std::vector<cv::Mat> images;
        while (true)
        {
            cv::Mat image;
            video >> image;
            if (image.empty())
            {
                break;
            }
            images.push_back(image);
        }
        return std::make_pair(images, fps);
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

    std::vector<std::vector<byte_track::Object>> get_inputs(const json11::Json jobj,
                                                            const size_t &total_frame,
                                                            const int &im_width,
                                                            const int &im_height)
    {
        std::vector<std::vector<byte_track::Object>> inputs_ref;
        inputs_ref.resize(total_frame);
        for(auto result: jobj["results"].array_items())
        {
            const auto frame_id = stoi(result["frame_id"].string_value());
            const auto prob = stof(result["prob"].string_value());
            const auto x = std::clamp(stof(result["x"].string_value()), 0.F, im_width - 1.F);
            const auto y = std::clamp(stof(result["y"].string_value()), 0.F, im_height - 1.F);
            const auto width = std::clamp(stof(result["width"].string_value()), 0.F, im_width - x);
            const auto height = std::clamp(stof(result["height"].string_value()), 0.F, im_height - y);
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

int main(int argc, char *argv[])
{
    try
    {
        if (argc < 3)
        {
            throw std::runtime_error("Usage: $ ./generate_submit_file <detection results path> <videos path>");
        }

        std::vector<std::filesystem::path> detection_results_path;
        for (const auto &file : std::filesystem::directory_iterator(argv[1]))
        {
            if (file.path().extension() == ".json")
            {
                detection_results_path.push_back(file);
            }
        }

        std::vector<std::filesystem::path> videos_path;
        for (const auto &file : std::filesystem::directory_iterator(argv[2]))
        {
            if (file.path().extension() == ".mp4")
            {
                videos_path.push_back(file);
            }
        }

        std::sort(detection_results_path.begin(), detection_results_path.end());
        std::sort(videos_path.begin(), videos_path.end());

        //key: video_name
        //value: tracked object dict list
        std::map<std::string, json11::Json> frame_objects_map;
        auto detection_results_filename_itr = detection_results_path.begin();
        for (const auto &video_path : videos_path)
        {
            std::cout << video_path.filename() << std::endl;
            const std::string video_basename = video_path.stem();
            const auto read_json = [&](decltype(detection_results_path)::iterator &itr, json11::Json &jobj) -> bool
            {
                const auto &path = *itr;
                const std::string basename = path.stem();
                const auto prefix_is_same = (video_basename.size() <= basename.size() && std::equal(video_basename.begin(), video_basename.end(), basename.begin()));
                if (prefix_is_same)
                {
                    // boost::property_tree::read_json(path.string(), pt);
                    std::ifstream ifs(path.string());
                    std::string jsonstr((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                    std::string err;
                    jobj = json11::Json::parse(jsonstr, err);
                    ifs.close();
                    itr++;
                }
                return prefix_is_same;
            };

            json11::Json jobj_car, jobj_pedestrian;
            if (!read_json(detection_results_filename_itr, jobj_car))
            {
                continue;
            }
            if (!read_json(detection_results_filename_itr, jobj_pedestrian))
            {
                break;
            }

            // Get video info
            const auto [images, fps] = read_video(video_path);
            const auto width = images[0].size().width;
            const auto height = images[0].size().height;

            // Get detection results
            const auto inputs_car = get_inputs(jobj_car, images.size(), width, height);
            const auto inputs_pedestrian = get_inputs(jobj_pedestrian, images.size(), width, height);

            #ifdef RISCV
            int uio0_fd = open("/dev/uio0", O_RDWR | O_SYNC);
            int* riscv_dmem_base = (int*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio0_fd, 0);
            int uio1_fd = open("/dev/uio1", O_RDWR | O_SYNC);
            unsigned int* riscv_imem_base = (unsigned int*) mmap(NULL, 0x10000, PROT_READ|PROT_WRITE, MAP_SHARED, uio1_fd, 0);
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
            for (size_t fi = 0; fi < images.size(); fi++)
            {
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
                copy(car_tracker.update(inputs_car[fi]), outputs_car);
                copy(pedestrian_tracker.update(inputs_pedestrian[fi]), outputs_pedestrian);
            }

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
                Results result(images.size());
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
            for (size_t fi = 0; fi < images.size(); fi++)
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
            frame_objects_map[video_path.filename().string()] = json11::Json(frame_objects);

            // Write video with tracking result
            // write_video(draw_images, video_path.filename(), fps);

            if (detection_results_filename_itr == detection_results_path.end())
            {
                break;
            }
        }

        // Write submit file
        std::ofstream file;
        file.open("predictions.json");
        file << json11::Json(frame_objects_map).dump();
        file.close();

    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
    }
}

#include <ByteTrack/BYTETracker.h>

#include <opencv2/opencv.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <boost/optional.hpp>

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

    template <typename T>
    T get_data(const boost::property_tree::ptree &pt, const std::string &key)
    {
        T ret;
        if (boost::optional<T> data = pt.get_optional<T>(key))
        {
            ret = data.get();
        }
        else
        {
            throw std::runtime_error("Could not read the data from ptree: [key: " + key + "]");
        }
        return ret;
    }

    std::vector<std::vector<byte_track::Object>> get_inputs(const boost::property_tree::ptree &pt, const size_t &total_frame)
    {
        std::vector<std::vector<byte_track::Object>> inputs_ref;
        inputs_ref.resize(total_frame);
        BOOST_FOREACH (const boost::property_tree::ptree::value_type &child, pt.get_child("results"))
        {
            const boost::property_tree::ptree &result = child.second;
            const auto frame_id = get_data<int>(result, "frame_id");
            const auto prob = get_data<float>(result, "prob");
            const auto x = get_data<float>(result, "x");
            const auto y = get_data<float>(result, "y");
            const auto width = get_data<float>(result, "width");
            const auto height = get_data<float>(result, "height");
            inputs_ref[frame_id].emplace_back(byte_track::Rect(x, y, width, height), 0, prob);
        }
        return inputs_ref;
    }

    cv::Rect2i get_rounded_rect2i(const byte_track::Rect<float> &rect)
    {
        return cv::Rect2i(
            std::round(rect.x()),
            std::round(rect.y()),
            std::round(rect.width()),
            std::round(rect.height())
        );
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

        boost::property_tree::ptree submit_pt;
        auto detection_results_filename_itr = detection_results_path.begin();
        for (const auto &video_path : videos_path)
        {
            std::cout << video_path.filename() << std::endl;
            const std::string video_basename = video_path.stem();
            const auto read_json = [&](decltype(detection_results_path)::iterator &itr, boost::property_tree::ptree &pt) -> bool
            {
                const auto &path = *itr;
                const std::string basename = path.stem();
                const auto prefix_is_same = (video_basename.size() <= basename.size() && std::equal(video_basename.begin(), video_basename.end(), basename.begin()));
                if (prefix_is_same)
                {
                    boost::property_tree::read_json(path.string(), pt);
                    itr++;
                }
                return prefix_is_same;
            };

            boost::property_tree::ptree pt_results_car, pt_results_pedestrian;
            if (!read_json(detection_results_filename_itr, pt_results_car))
            {
                continue;
            }
            if (!read_json(detection_results_filename_itr, pt_results_pedestrian))
            {
                break;
            }

            // Get video info
            const auto [images, fps] = read_video(video_path);
            const auto width = images[0].size().width;
            const auto height = images[0].size().height;

            // Get detection results
            const auto inputs_car = get_inputs(pt_results_car, images.size());
            const auto inputs_pedestrian = get_inputs(pt_results_pedestrian, images.size());

            // Execute tracking
            byte_track::BYTETracker car_tracker(fps, fps);
            byte_track::BYTETracker pedestrian_tracker(fps, fps);

            std::vector<cv::Mat> draw_images;
            std::vector<std::vector<byte_track::STrack>> outputs_car;
            std::vector<std::vector<byte_track::STrack>> outputs_pedestrian;
            for (size_t fi = 0; fi < images.size(); fi++)
            {
                draw_images.push_back(images[fi].clone());
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
                        const auto rect2i = get_rounded_rect2i(strack.getRect());
                        if (1024 <= rect2i.area() &&
                            0 <= rect2i.tl().x && 0 <= rect2i.tl().y &&
                            rect2i.br().x < width && rect2i.br().y < height)
                        {
                            map.try_emplace(strack.getTrackId(), std::vector<std::pair<size_t, cv::Rect2i>>());
                            map[strack.getTrackId()].emplace_back(fi, rect2i);
                        }
                    }
                }

                // Validate
                Results result(draw_images.size());
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
            boost::property_tree::ptree frames_pt;
            for (size_t fi = 0; fi < images.size(); fi++)
            {
                boost::property_tree::ptree frame_pt;
                const auto gen_objs_pt_and_draw_rect = [&](Results &results,
                                                           const std::string &name) -> void
                {
                    const auto get_obj_pt = [](boost::property_tree::ptree &pt,
                                               const cv::Rect2i &rect,
                                               const size_t &track_id) -> void
                    {
                        boost::property_tree::ptree box2d_pt, tl_x_pt, tl_y_pt, br_x_pt, br_y_pt;
                        tl_x_pt.put_value(rect.tl().x);
                        tl_y_pt.put_value(rect.tl().y);
                        br_x_pt.put_value(rect.br().x);
                        br_y_pt.put_value(rect.br().y);
                        box2d_pt.push_back({"", tl_x_pt});
                        box2d_pt.push_back({"", tl_y_pt});
                        box2d_pt.push_back({"", br_x_pt});
                        box2d_pt.push_back({"", br_y_pt});
                        pt.put("id", track_id);
                        pt.add_child("box2d", box2d_pt);
                    };

                    boost::property_tree::ptree objs_pt;
                    for (const auto &[track_id, rect] : results[fi])
                    {
                        boost::property_tree::ptree obj_pt;
                        get_obj_pt(obj_pt, rect, track_id);
                        objs_pt.push_back({"", obj_pt});
                        draw_rect(draw_images[fi], rect, track_id, name.substr(0, 1));
                    }
                    if (objs_pt.size() != 0)
                    {
                        frame_pt.add_child(name, objs_pt);
                    }
                };

                gen_objs_pt_and_draw_rect(results_car, "Car");
                gen_objs_pt_and_draw_rect(results_pedestrian, "Pedestrian");
                frames_pt.push_back({"", frame_pt});
            }

            submit_pt.add_child(boost::property_tree::path(video_path.filename().string(), '\0'), frames_pt);

            // Write video with tracking result
            write_video(draw_images, video_path.filename(), fps);

            if (detection_results_filename_itr == detection_results_path.end())
            {
                break;
            }
        }

        // Write submit file
        std::ostringstream oss;
        boost::property_tree::write_json(oss, submit_pt);

        std::regex reg1("\\\"([0-9]+\\.{0,1}[0-9]*)\\\"");
        std::regex reg2("\\\"\\\"");
        std::ofstream file;
        file.open("predictions.json");
        file << std::regex_replace(std::regex_replace(oss.str(), reg1, "$1"), reg2, "{}");
        file.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
    }
}

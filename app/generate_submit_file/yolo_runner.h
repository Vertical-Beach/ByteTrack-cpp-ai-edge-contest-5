#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <ByteTrack/BYTETracker.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <fstream>
#include <map>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;

// The parameters of yolov3_voc, each value could be set as actual needs.
// Such format could be refer to the prototxts in /etc/dpu_model_param.d.conf/.

const string readFile(const char *filename){
  ifstream ifs(filename);
  return string(istreambuf_iterator<char>(ifs),
                istreambuf_iterator<char>());
}

class YoloRunner{
  private:
    unique_ptr<vitis::ai::DpuTask> task;
    vitis::ai::proto::DpuModelParam modelconfig;
    cv::Size model_input_size;
    vector<vitis::ai::library::InputTensor> input_tensor;

  public: YoloRunner(const char* modelconfig_path, const char* modelfile_path){
    const string config_str = readFile(modelconfig_path);
    auto ok = google::protobuf::TextFormat::ParseFromString(config_str, &(this->modelconfig));
    if (!ok) {
      cerr << "Set parameters failed!" << endl;
      abort();
    }
    this->task = vitis::ai::DpuTask::create(modelfile_path);
    this->input_tensor = task->getInputTensor(0u);
    int width = this->input_tensor[0].width;
    int height = this->input_tensor[0].height;
    this->model_input_size = cv::Size(width, height);
    this->task->setMeanScaleBGR({0.0f, 0.0f, 0.0f},
                        {0.00390625f, 0.00390625f, 0.00390625f});
  }
  private: cv::Mat Preprocess(cv::Mat img){
    cv::Mat resized_img;
    cv::resize(img, resized_img, this->model_input_size);
    return resized_img;
  }
  public: vector<vector<byte_track::Object>> Run(cv::Mat img){
    cv::Mat resized_img = this->Preprocess(img);
    vector<int> input_cols = {img.cols};
    vector<int> input_rows = {img.rows};
    vector<cv::Mat> inputs = {resized_img};
    task->setImageRGB(inputs);
    task->run(0);

    auto output_tensor = task->getOutputTensor(0u);
    auto results = vitis::ai::yolov3_post_process(
        input_tensor, output_tensor, this->modelconfig, input_cols, input_rows);
    auto result = results[0]; //batch_size is 1
    vector<vector<byte_track::Object>> objs(2); //[car, pedestrian]
    for(auto& yolobbox: result.bboxes){
      const auto x = std::clamp(yolobbox.x * (float)img.cols, 0.F, img.cols - 1.F);
      const auto y = std::clamp(yolobbox.y * (float)img.rows, 0.F, img.rows - 1.F);
      const auto width = std::clamp(yolobbox.width * (float)img.cols, 0.F, img.cols - x);
      const auto height = std::clamp(yolobbox.height * (float)img.rows, 0.F, img.rows - y);
      auto rect = byte_track::Rect(x, y, width, height);
      objs[yolobbox.label].push_back(byte_track::Object(rect, yolobbox.label, yolobbox.score));
    }
    return objs;
  }

};

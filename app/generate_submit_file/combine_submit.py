import json
import sys
import glob
import numpy as np

resmap = {}
args = sys.argv
if len(args) != 2:
    print("Usage: $ python3 combine_submit.py <video name prefix>")
else:
    video_name_prefix = args[1]
    prediction_files = glob.glob("prediction_" + video_name_prefix + "*.json")
    for file in prediction_files:
        video_idx = file[-7:]
        video_idx = video_idx[:2]
        videoname = f"{video_name_prefix}_{video_idx}.mp4"
        print(videoname)
        videores = json.loads(open(f"./prediction_{video_name_prefix}_{video_idx}.json").read())
        resmap[videoname] = videores
    open("predictions.json", "w").write(json.dumps(resmap))

    detection_time_videos = []
    detection_time_frames = []
    tracking_time_videos = []
    tracking_time_frames = []
    all_time_videos = []
    all_time_frames = []
    time_summary_files = glob.glob("time_summary_" + video_name_prefix + "*.json")
    for file in time_summary_files:
        video_idx = file[-7:]
        video_idx = video_idx[:2]
        timeres = json.loads(open(f"./time_summary_{video_name_prefix}_{video_idx}.json").read())
        frame_cnt = int(timeres["frame_cnt"])
        detection_time_videos.append(timeres["detection_sum"])
        detection_time_frames.append(timeres["detection_sum"]/frame_cnt)
        tracking_time_videos.append(timeres["tracking_sum"])
        tracking_time_frames.append(timeres["tracking_sum"]/frame_cnt)
        all_time_videos.append(timeres["all_sum"])
        all_time_frames.append(timeres["all_sum"]/frame_cnt)
    print("time summary:")
    print("detection : {:.2f}ms/frame {:.2f}ms/video".format(
        np.mean(np.array(detection_time_frames)), np.mean(np.array(detection_time_videos))))
    print("tracking : {:.2f}ms/frame {:.2f}ms/video".format(
        np.mean(np.array(tracking_time_frames)), np.mean(np.array(tracking_time_videos))))
    print("detection + tracking : {:.2f}ms/frame {:.2f}ms/video".format(
        np.mean(np.array(all_time_frames)), np.mean(np.array(all_time_videos))))
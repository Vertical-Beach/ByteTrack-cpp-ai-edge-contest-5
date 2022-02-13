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

    inference_times = []
    tracking_times = []
    all_time_sum = 0
    time_summary_files = glob.glob("time_summary_" + video_name_prefix + "*.json")
    for file in time_summary_files:
        video_idx = file[-7:]
        video_idx = video_idx[:2]
        timeres = json.loads(open(f"./time_summary_{video_name_prefix}_{video_idx}.json").read())
        inference_times.append(timeres["inference"])
        tracking_times.append(timeres["tracking"])
        all_time_sum += timeres["all"]
    print("time summary:")
    print("inference avg.", np.mean(np.array(inference_times)), "[ms]")
    print("tracking avg.", np.mean(np.array(tracking_times)), "[ms]")
    print("all time sum.", all_time_sum, "[ms]")

import json
import sys
import glob

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

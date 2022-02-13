import json
import os

resmap = {}
for i in range(74):
    videoname = f"test_{str(i).zfill(2)}.mp4"
    print(videoname)
    videores = json.loads(open(f"./build/prediction_test_{str(i).zfill(2)}.json").read())
    assert(len(videores) == 150)
    resmap[videoname] = videores

open("predictions.json", "w").write(json.dumps(resmap))

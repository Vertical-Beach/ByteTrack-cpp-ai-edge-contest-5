1. Environment:
  - python>=3.6
  - library
    - scipy==1.1.0
    - numpy==1.16.3

2. Usage:
  - In the command line, execute
    ```
    $ python evaluate.py --prediction-file path/to/prediction --answer-file path/to/answer --threshold threshold
    ```
    and you can get the MOTA@threshold score of the prediction for the given ground truth(answer).

3. Notes:
  - The format of the answer file and the prediction file:
    - File Name: ***.json (*** = whatever name you like(e.g. predictions))
    - Description:
      - video_file_0 []:
          - category_1 []:
              - id: int
              - box2d: [x1, y1, x2, y2]
          - category_2 []:
              - id: int
              - box2d: [x1, y1, x2, y2]
          ...
      - video_file_1 []:
          - category_1 []:
              - id: int
              - box2d: [x1, y1, x2, y2]
          - category_2 []:
              - id: int
              - box2d: [x1, y1, x2, y2]
          ...
      ...
    - (x1, y1, x2, y2) corresponds to (left, top, right, bottom).
    - In the answer file(not the prediction file), for each video file, there have to exist at least one object.
    - The objects to be evaluated (objects in the answer file) is limited to those that satisfy all of the following.
      - There are two categories: "Car" and "Pedestrian"
      - Objects with 3 or more frames in each video (Frames do not have to be continuous)
      - Objects with a rectangle size of 1024 pix^2 or more
    - Even if an object appears in three or more frames in the answer file, if the number of object of size 1024 pix^2
      or more is less than three, the object including the size of 1024 pix^2 or more will not be evaluated.
    - For each video file, the annotations in each frame should be placed in the list in chronological order.
    - Please also refer to "ans.json" or "predictions.json".

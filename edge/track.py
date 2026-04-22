import math

THRESHOLD = 0.6

def _intersection_area(a, b):
    top = max(a[1], b[1])
    left = max(a[0], b[0])
    bottom = min(a[3], b[3])
    right = min(a[2], b[2])
    w = max(0, right - left)
    h = max(0, bottom - top)
    return w * h

def _area(a):
    w = max(0, a[2] - a[0])
    h = max(0, a[3] - a[1])
    return w * h

def iou(a, b):
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    if union == 0:
        return 0.0
    return inter / union

def preprocess_data(last_frame, current_frame):
    res = []
    temp = []

    if len(last_frame) > 0:
        for _, target in enumerate(current_frame):
            max_iou = 0
            temp_pair = []
            remove_index = 0
            remove_target = []
            a = [target['shape'][0], target['shape'][1], target['shape'][2], target['shape'][3]]
            for l_index, l_target in enumerate(last_frame):
                b = [l_target['shape'][0], l_target['shape'][1], l_target['shape'][2], l_target['shape'][3]]
                _iou = iou(a, b)
                if _iou > THRESHOLD:
                    if _iou > max_iou:
                        temp_pair = [l_target['name'], target['name']]
                        max_iou = _iou
                        remove_index = l_index
                        remove_target = l_target
            if temp_pair:
                res.append(temp_pair)
                last_frame.pop(remove_index)
                temp.append(remove_target)
    last_frame.extend(temp)
    return last_frame, current_frame, res

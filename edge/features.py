import math

def target_size(bbox, frame_w, frame_h):
    w = max(0.0, float(bbox[2]) - float(bbox[0]))
    h = max(0.0, float(bbox[3]) - float(bbox[1]))
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return 0.0
    return (w * h) / frame_area

def target_dynamics(curr_bbox, prev_bbox, diag):
    if diag <= 0:
        return 0.0
    cx1 = (float(curr_bbox[0]) + float(curr_bbox[2])) / 2.0
    cy1 = (float(curr_bbox[1]) + float(curr_bbox[3])) / 2.0
    cx0 = (float(prev_bbox[0]) + float(prev_bbox[2])) / 2.0
    cy0 = (float(prev_bbox[1]) + float(prev_bbox[3])) / 2.0
    dist = math.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)
    return dist / diag

def frame_uncertainty(mid_confidences):
    if not mid_confidences:
        return 0.0
    return 1.0 - sum(mid_confidences) / len(mid_confidences)

def compute_frame_features(all_targets, tracked_pairs, frame_w, frame_h,
                           low_thr, high_thr):
    diag = math.sqrt(frame_w ** 2 + frame_h ** 2) if (frame_w > 0 and frame_h > 0) else 1.0

    sizes = []
    for t in all_targets:
        bbox = t.get('shape') or t.get('bbox')
        if bbox and len(bbox) >= 4:
            sizes.append(target_size(bbox, frame_w, frame_h))
    s = sum(sizes) / len(sizes) if sizes else 0.0

    dynamics = []
    for prev_t, curr_t in tracked_pairs:
        prev_bbox = prev_t.get('shape') or prev_t.get('bbox')
        curr_bbox = curr_t.get('shape') or curr_t.get('bbox')
        if prev_bbox and curr_bbox and len(prev_bbox) >= 4 and len(curr_bbox) >= 4:
            dynamics.append(target_dynamics(curr_bbox, prev_bbox, diag))
    d = sum(dynamics) / len(dynamics) if dynamics else 0.0

    mid_confs = []
    for t in all_targets:
        conf = float(t.get('confidence', t.get('score', t.get('conf', 0.0))))
        if low_thr <= conf < high_thr:
            mid_confs.append(conf)
    u = frame_uncertainty(mid_confs)

    return s, d, u

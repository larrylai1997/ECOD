import os
import cv2 as cv
try:
    from . import config
except ImportError:

    import importlib
    config = importlib.import_module('config')

_yolo = None
if config.USE_YOLO:
    import sys as _sys
    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    try:
        from utils import yolo_det as _yolo
    except Exception:
        _yolo = None

def _normalize_boxes(image_path, detections):
    if not detections or not image_path:
        return detections

    img = cv.imread(image_path)
    if img is None:
        return detections

    h, w = img.shape[:2]
    normalized = []
    for det in detections:
        if len(det) < 4:
            continue
        x1, y1, x2, y2 = det[:4]
        try:
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
        except Exception:
            continue

        x1 = max(0, min(int(round(x1)), w - 1))
        x2 = max(0, min(int(round(x2)), w))
        y1 = max(0, min(int(round(y1)), h - 1))
        y2 = max(0, min(int(round(y2)), h))
        if x2 <= x1 or y2 <= y1:
            continue

        norm = [x1, y1, x2, y2]
        if len(det) > 4:
            norm.extend(det[4:])
        normalized.append(norm)

    return normalized

def detect(image_path):
    if _yolo is not None:
        return _normalize_boxes(image_path, _yolo.detect(image_path))

    det_txt_dir = os.path.join(os.path.dirname(__file__), 'det_txt')
    detections = []
    if not os.path.isdir(det_txt_dir) or not image_path:
        return detections

    img_name = os.path.basename(image_path)
    base_id = os.path.splitext(img_name)[0]

    rel_id = None
    try:
        rel_candidate = os.path.relpath(image_path, getattr(config, 'DATASET_DIR', ''))
        if not rel_candidate.startswith(os.pardir):
            rel_id = os.path.splitext(rel_candidate)[0].replace('\\', '/')
    except Exception:
        rel_id = None

    def _match_case_insensitive(target_lower):
        for root, _, files in os.walk(det_txt_dir):
            for fname in files:
                if not fname.lower().endswith('.txt'):
                    continue
                path = os.path.join(root, fname)
                rel_stem = os.path.splitext(os.path.relpath(path, det_txt_dir))[0].replace('\\', '/').lower()
                stem = os.path.splitext(fname)[0].lower()
                if rel_stem == target_lower or stem == target_lower:
                    return path
        return None

    candidate = None

    if rel_id:
        candidate = os.path.join(det_txt_dir, rel_id + '.txt')
        if not os.path.isfile(candidate):
            candidate = _match_case_insensitive(rel_id.lower())

    if (not candidate) and base_id:
        direct = os.path.join(det_txt_dir, base_id + '.txt')
        if os.path.isfile(direct):
            candidate = direct

    if not candidate and base_id:
        candidate = _match_case_insensitive(base_id.lower())
    if not candidate:
        return detections

    try:
        with open(candidate, 'r', encoding='utf-8') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                if len(parts) < 6:
                    continue
                try:
                    x1 = int(round(float(parts[0])))
                    y1 = int(round(float(parts[1])))
                    x2 = int(round(float(parts[2])))
                    y2 = int(round(float(parts[3])))
                    conf = float(parts[4])
                    label = str(int(float(parts[5])))
                except Exception:
                    continue
                detections.append([x1, y1, x2, y2, conf, label])
    except Exception:
        return []
    return _normalize_boxes(image_path, detections)

from flask import Flask, request, make_response
import os
import json
import cv2 as cv
import requests
import logging
import sys
from collections import OrderedDict
import tempfile

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_LOW = os.path.join(APP_ROOT, "low_img.txt")
FUSED_TXT_DIR = os.path.join(APP_ROOT, "fused_txt")
DETECTION_SERVICE_URL = os.environ.get("DETECTION_SERVICE_URL", "http://127.0.0.1:5002/detect")
DETECTION_TIMEOUT = float(os.environ.get("DETECTION_TIMEOUT", "10.0"))
SERVER_CONF_THRESHOLD = float(os.environ.get("SERVER_CONF_THRESHOLD", "0.3"))
FUSED_NMS_IOU_THRESHOLD = float(os.environ.get("FUSED_NMS_IOU_THRESHOLD", "0.5"))
FUSED_NMS_CLASS_AWARE = os.environ.get("FUSED_NMS_CLASS_AWARE", "0").strip().lower() not in ("0", "false", "no")

app = Flask(__name__)

LOG = logging.getLogger("ecod.server")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    LOG.addHandler(console)
    file_handler = logging.FileHandler(os.path.join(APP_ROOT, "server.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)

def _ensure_fused_dir():
    if not os.path.isdir(FUSED_TXT_DIR):
        os.makedirs(FUSED_TXT_DIR, exist_ok=True)

def _normalize_frame_id(frame_id):
    if not frame_id:
        return ""
    stem = os.path.splitext(frame_id)[0]
    stem = stem.replace("\\", "/")
    return stem.lstrip("./")

def _compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom

def _write_fused(frame_id, detections):
    _ensure_fused_dir()
    out_path = os.path.join(FUSED_TXT_DIR, f"{frame_id}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for det in detections or []:
            bbox = det.get('bbox') or []
            if len(bbox) < 4:
                continue
            try:
                line = (
                    f"{float(bbox[0]):.2f},"
                    f"{float(bbox[1]):.2f},"
                    f"{float(bbox[2]):.2f},"
                    f"{float(bbox[3]):.2f},"
                    f"{float(det.get('score', 0.0)):.2f},"
                    f"{int(det.get('cls', det.get('label', 0)))}"
                )
            except (TypeError, ValueError):
                continue
            f.write(line + "\n")

def _filter_remote(detections):
    filtered = []
    for det in detections or []:
        try:
            score = float(det.get('score', det.get('conf', 0.0)))
        except (TypeError, ValueError):
            continue
        if score < SERVER_CONF_THRESHOLD:
            continue
        filtered.append(det)
    return filtered

def _normalize_local_from_meta(meta):
    high = []

    def _push(dst, det):
        bbox = det.get('bbox') or det.get('shape')
        if not bbox or len(bbox) < 4:
            return
        try:
            coords = [float(bbox[i]) for i in range(4)]
            score = float(det.get('score', det.get('confidence', det.get('conf', 0.0))))
            cls = int(det.get('cls', det.get('label', det.get('result', 0))))
        except (TypeError, ValueError):
            return
        dst.append({'bbox': coords, 'score': score, 'cls': cls})

    for d in meta.get('local_high') or []:
        _push(high, d)
    return {'high': high}

def _nms(detections, iou_thr, class_aware=True):
    if not detections:
        return []
    try:
        iou_thr = float(iou_thr)
    except (TypeError, ValueError):
        iou_thr = 0.5

    def _score(det):
        try:
            return float(det.get('score', 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _valid_bbox(det):
        bbox = det.get('bbox')
        if not bbox or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    norm = []
    for det in detections:
        bbox = _valid_bbox(det)
        if bbox is None:
            continue
        copied = dict(det)
        copied['bbox'] = bbox
        norm.append(copied)

    if not norm:
        return []

    groups = {}
    if class_aware:
        for det in norm:
            groups.setdefault(det.get('cls'), []).append(det)
    else:
        groups[None] = norm

    kept_all = []
    for _cls, dets in groups.items():
        dets_sorted = sorted(dets, key=_score, reverse=True)
        kept = []
        for det in dets_sorted:
            bbox = det.get('bbox')
            should_keep = True
            for k in kept:
                if _compute_iou(bbox, k.get('bbox')) >= iou_thr:
                    should_keep = False
                    break
            if should_keep:
                kept.append(det)
        kept_all.extend(kept)

    kept_all.sort(key=_score, reverse=True)
    return kept_all

def _compose_fused(remote_filtered, local_contrib):

    fused = []
    fused.extend(remote_filtered or [])
    fused.extend((local_contrib or {}).get('high') or [])
    return _nms(fused, iou_thr=FUSED_NMS_IOU_THRESHOLD, class_aware=FUSED_NMS_CLASS_AWARE)

def _normalize_remote_detections(detections):
    normalized = []
    for det in detections or []:
        bbox = det.get('bbox') or det.get('shape')
        if not bbox or len(bbox) < 4:
            continue
        try:
            coords = [float(bbox[i]) for i in range(4)]
            score = float(det.get('score', det.get('conf', 0.0)))
            cls = int(det.get('cls', det.get('label', 0)))
        except (TypeError, ValueError):
            continue
        normalized.append({
            'bbox': coords,
            'score': score,
            'cls': cls,
            'source': 'server'
        })
    return normalized

class DetectionServiceError(Exception):
    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code

def _request_detections_from_service(frame_id, image_bytes):
    if not image_bytes:
        raise DetectionServiceError("image_not_found", status_code=404)

    try:
        resp = requests.post(
            DETECTION_SERVICE_URL,
            files={'image': ('frame.png', image_bytes, 'image/png')},
            data={'frame_id': frame_id},
            timeout=DETECTION_TIMEOUT
        )
    except requests.RequestException as exc:
        raise DetectionServiceError(f"service_unreachable: {exc}", status_code=502)

    if resp.status_code == 404:
        raise DetectionServiceError("service_reported_not_ready", status_code=404)
    if not resp.ok:
        raise DetectionServiceError(f"service_error:{resp.status_code}", status_code=resp.status_code)

    try:
        payload = resp.json()
    except ValueError:
        raise DetectionServiceError("service_invalid_json", status_code=502)

    detections = _normalize_remote_detections(payload.get('detections', []))
    frame_label = payload.get('frame_id') or os.path.splitext(os.path.basename(frame_id))[0]
    return {
        'frame_id': frame_label,
        'detections': detections
    }

@app.before_first_request
def init():
    for fpath in [LOG_LOW]:
        if os.path.isfile(fpath):
            os.remove(fpath)
        open(fpath, 'w').close()
    _ensure_fused_dir()

def _combine_roi_background(roi, bg):
    if roi is None or bg is None:
        return None
    if roi.shape[:2] != bg.shape[:2]:
        roi = cv.resize(roi, (bg.shape[1], bg.shape[0]), interpolation=cv.INTER_LINEAR)

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    bg_part = cv.bitwise_and(bg, bg, mask=mask_inv)
    roi_part = cv.bitwise_and(roi, roi, mask=mask)
    return cv.add(bg_part, roi_part)

def _encode_png_bytes(image):
    if image is None:
        return None
    ok, buf = cv.imencode(".png", image)
    if not ok:
        return None
    return buf.tobytes()

def _read_video_frame(cap, index=None):
    if cap is None or not cap.isOpened():
        return None
    if index is not None:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

@app.route("/")
@app.route("/index")
def index():
    LOG.info("hello world")
    return "ECOD server running"

@app.route("/low_composite", methods=["POST"])
def receive_low_composite():
    batch_metadata_json = request.form.get("metadata")
    if not batch_metadata_json:
        return make_response("invalid_request", 400)

    try:
        metadata_list = json.loads(batch_metadata_json)

        for idx, meta in enumerate(metadata_list):
            frame_id = meta.get('frame_id') or f"{idx:010d}"
            norm_id = _normalize_frame_id(frame_id)

            image_file = request.files.get(f'image_{idx}')
            if image_file is None:
                LOG.warning("[Composite] missing image_%d", idx)
                continue

            file_bytes = image_file.read()
            image_bytes = file_bytes

            remote = []
            if image_bytes:
                try:
                    resp = _request_detections_from_service(frame_id, image_bytes)
                    remote = resp.get('detections', [])
                except DetectionServiceError as exc:
                    LOG.warning("[Composite] detection error for %s: %s", frame_id, exc)
                except Exception as exc:
                    LOG.warning("[Composite] detection failed for %s: %s", frame_id, exc)

            local = _normalize_local_from_meta(meta)
            remote_filtered = _filter_remote(remote)
            fused = _compose_fused(remote_filtered, local)
            _write_fused(norm_id, fused)
            LOG.info("[Composite] fused %s: %d objects", norm_id, len(fused))

    except Exception as e:
        LOG.error("Error processing composite batch: %s", e)
        return make_response("0.0", 500)

    return make_response("0.0")

@app.route("/low", methods=["POST"])
def receive_low():
    video_file = request.files.get("video")
    background_file = request.files.get("background")
    batch_metadata_json = request.form.get("metadata")

    if not video_file or not batch_metadata_json:
        return make_response("invalid_request", 400)

    try:
        metadata_list = json.loads(batch_metadata_json)

        with tempfile.TemporaryDirectory(prefix="ecod_upload_") as tmp_dir:
            roi_video_path = os.path.join(tmp_dir, "roi.mp4")
            video_file.save(roi_video_path)

            bg_video_path = None
            if background_file:
                bg_video_path = os.path.join(tmp_dir, "background.mp4")
                background_file.save(bg_video_path)

            roi_cap = cv.VideoCapture(roi_video_path)
            bg_cap = cv.VideoCapture(bg_video_path) if bg_video_path else None
            bg_cache = OrderedDict()
            max_bg_cache = 8

            for idx, meta in enumerate(metadata_list):
                frame_id = meta.get('frame_id') or f"{idx:010d}"
                norm_id = _normalize_frame_id(frame_id)

                roi_frame = _read_video_frame(roi_cap, index=idx)
                if roi_frame is None:

                    LOG.warning("[Fusion] ROI video frame missing at idx=%d; skipping remote detection", idx)

                combined = None
                bg_idx = meta.get('background_index')
                if roi_frame is not None and bg_cap is not None and bg_idx is not None and isinstance(bg_idx, int) and bg_idx >= 0:
                    if bg_idx in bg_cache:
                        bg_frame = bg_cache[bg_idx]
                        bg_cache.move_to_end(bg_idx)
                    else:
                        bg_frame = _read_video_frame(bg_cap, index=bg_idx)
                        if bg_frame is not None:
                            bg_cache[bg_idx] = bg_frame
                            if len(bg_cache) > max_bg_cache:
                                bg_cache.popitem(last=False)
                    if bg_frame is not None:
                        combined = _combine_roi_background(roi_frame, bg_frame)

                image_bytes = _encode_png_bytes(combined if combined is not None else roi_frame)

                remote = []
                if image_bytes:
                    try:
                        resp = _request_detections_from_service(frame_id, image_bytes)
                        remote = resp.get('detections', [])
                    except DetectionServiceError as exc:
                        LOG.warning("[Fusion] detection service error for %s: %s", frame_id, exc)
                    except Exception as exc:
                        LOG.warning("[Fusion] detection service failed for %s: %s", frame_id, exc)

                local = _normalize_local_from_meta(meta)
                remote_filtered = _filter_remote(remote)
                fused = _compose_fused(remote_filtered, local)
                _write_fused(norm_id, fused)
                LOG.info("[Fusion] stored fused detections for %s (%d objects)", norm_id, len(fused))

            try:
                roi_cap.release()
            except Exception:
                pass
            if bg_cap is not None:
                try:
                    bg_cap.release()
                except Exception:
                    pass

    except Exception as e:
        LOG.error("Error processing video batch: %s", e)
        return make_response("0.0", 500)

    return make_response("0.0")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)

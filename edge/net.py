import os
import time
import math
import json
import requests
import shutil
import subprocess
import shutil as _shutil
import cv2 as cv

try:
    from . import config
except ImportError:
    import importlib

    config = importlib.import_module('config')

def _ensure_dirs():
    for d in [config.CACHE_DIR, config.TEMP_DIR]:
        if os.path.isdir(d):
            continue
        os.makedirs(d, exist_ok=True)

def _normalize_frame_id(frame_id):
    norm = (frame_id or "").replace("\\", "/")
    fs_safe = norm.replace("/", "__")
    return norm, fs_safe

def _expand_rect(rect, width, height, pad_ratio=0.0, pad_pix=0):
    try:
        x0, y0, x1, y1 = map(float, rect)
    except Exception:
        return None
    pad_w = (x1 - x0) * pad_ratio + pad_pix
    pad_h = (y1 - y0) * pad_ratio + pad_pix
    x0 = max(0.0, x0 - pad_w)
    y0 = max(0.0, y0 - pad_h)
    x1 = min(float(width), x1 + pad_w)
    y1 = min(float(height), y1 + pad_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

def _infer_frame_file_from_name(name):
    suffix = name.split('_', 1)[1] if '_' in name else name
    return os.path.join(config.DATASET_DIR, suffix)

def _compose_sparse_image_multi(frame_path, rects, save_name):
    base = cv.imread(frame_path)
    if base is None:
        return None

    canvas = cv.cvtColor(cv.cvtColor(base, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    canvas[:, :, :] = 0
    h, w = base.shape[:2]
    pad_ratio = getattr(config, 'ROI_PAD_RATIO', 0.0)
    pad_pix = getattr(config, 'ROI_PAD_PIX', 0)

    for r in rects:
        padded = _expand_rect(r, w, h, pad_ratio=pad_ratio, pad_pix=pad_pix)
        if not padded:
            continue
        x0, y0, x1, y1 = padded
        canvas[y0:y1, x0:x1] = base[y0:y1, x0:x1]

    base_name, _ = os.path.splitext(save_name)
    out_path = os.path.join(config.TEMP_DIR, base_name + ".png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cv.imwrite(out_path, canvas, [cv.IMWRITE_PNG_COMPRESSION, 0])
    return out_path

def prepare_sparse_frame(targets, frame_id=None):
    frame_suffix = None
    frame_path = None

    if targets:
        first_name = targets[0]['name']
        frame_suffix = first_name.split('_', 1)[1] if '_' in first_name else first_name
        frame_path = _infer_frame_file_from_name(first_name)
    elif frame_id:
        frame_suffix = frame_id
        frame_path = os.path.join(config.DATASET_DIR, frame_suffix)
    else:
        return None, None

    norm_id, fs_safe = _normalize_frame_id(frame_suffix)
    rects = [t['shape'] for t in targets]

    save_name = f"sparse_{fs_safe}"
    temp_path = _compose_sparse_image_multi(frame_path, rects, save_name)

    if temp_path is None:
        return None, None

    metadata = {
        'frame_id': norm_id,
        'targets': [{
            'name': t['name'],
            'shape': t['shape'],
            'conf': t['confidence'],
            'label': t['result']
        } for t in targets]
    }

    return temp_path, metadata

def prepare_composite_frame(targets, frame_path, frame_id):
    _ensure_dirs()
    img = cv.imread(frame_path)
    if img is None:
        return None, 0
    h, w = img.shape[:2]
    pad_ratio = getattr(config, 'ROI_PAD_RATIO', 0.0)
    pad_pix = getattr(config, 'ROI_PAD_PIX', 0)
    bg_quality = getattr(config, 'COMPOSITE_BG_QUALITY', 10)
    composite_quality = getattr(config, 'COMPOSITE_QUALITY', 30)

    _, bg_buf = cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, bg_quality])
    composite = cv.imdecode(bg_buf, cv.IMREAD_COLOR)

    rects = [t['shape'] for t in targets]
    for r in rects:
        padded = _expand_rect(r, w, h, pad_ratio=pad_ratio, pad_pix=pad_pix)
        if not padded:
            continue
        x0, y0, x1, y1 = padded
        composite[y0:y1, x0:x1] = img[y0:y1, x0:x1]

    norm_id, fs_safe = _normalize_frame_id(frame_id)
    out_path = os.path.join(config.TEMP_DIR, f"composite_{fs_safe}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cv.imwrite(out_path, composite, [cv.IMWRITE_PNG_COMPRESSION, 3])
    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    return out_path, size

def prepare_background_frame(frame_path, frame_id):
    _ensure_dirs()
    img = cv.imread(frame_path)
    if img is None:
        return None
    base_name = os.path.splitext(os.path.basename(frame_id))[0]
    out_path = os.path.join(config.TEMP_DIR, f"background_{base_name}.png")
    cv.imwrite(out_path, img, [cv.IMWRITE_PNG_COMPRESSION, 3])
    return out_path

def _encode_video_stream(frame_paths, batch_dir, prefix, qp):
    if not frame_paths:
        return None, 0

    seq_dir = os.path.join(batch_dir, f"{prefix}_seq")
    os.makedirs(seq_dir, exist_ok=True)

    for idx, src_path in enumerate(frame_paths):
        dst_path = os.path.join(seq_dir, f"{idx:010d}.png")
        shutil.copy(src_path, dst_path)

    out_path = os.path.join(batch_dir, f"{prefix}.mp4")
    ffmpeg_bin = getattr(config, 'FFMPEG_BIN', 'ffmpeg')
    if _shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"ffmpeg not found. Set config.FFMPEG_BIN to full path or add ffmpeg to PATH. Current: '{ffmpeg_bin}'"
        )

    input_pattern = os.path.join(seq_dir, "%010d.png")
    tried_errors = []
    success = False

    def _run_cmd(cmd):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc.returncode, proc.stderr.decode(errors='ignore')

    cmd_x264 = [
        ffmpeg_bin, "-y",
        "-loglevel", "error",
        "-start_number", "0",
        "-i", input_pattern,
        "-vcodec", "libx264",
        "-g", "15",
        "-keyint_min", "15",
        "-qp", str(qp),
        "-pix_fmt", "yuv420p",
        "-frames:v", str(len(frame_paths)),
        out_path
    ]
    code, err = _run_cmd(cmd_x264)
    if code == 0:
        success = True
    else:
        tried_errors.append(("libx264", err))
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass
        cmd_mpeg4 = [
            ffmpeg_bin, "-y",
            "-loglevel", "error",
            "-start_number", "0",
            "-i", input_pattern,
            "-vcodec", "mpeg4",
            "-g", "15",
            "-pix_fmt", "yuv420p",
            "-frames:v", str(len(frame_paths)),
            out_path
        ]
        code2, err2 = _run_cmd(cmd_mpeg4)
        if code2 == 0:
            success = True
        else:
            tried_errors.append(("mpeg4", err2))
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass

    if not success:
        msgs = " | ".join([f"{c}: {e.strip()}" for c, e in tried_errors if e])
        raise RuntimeError(f"ffmpeg encoding failed. Tried: {msgs}")

    size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    return out_path, size

def send_video_batch(frame_paths, metadata_list, bw_bytes, background_paths=None):
    if not frame_paths:
        return bw_bytes, 0.0

    batch_id = int(time.time() * 1000)
    batch_dir = os.path.join(config.TEMP_DIR, f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)
    nC = 0.0
    try:
        _enc_start = time.time()
        roi_video_path, roi_size = _encode_video_stream(frame_paths, batch_dir, "roi", config.ENCODING_QP)
        bw_bytes += roi_size

        background_video_path = None
        if background_paths:
            bg_qp = getattr(config, 'BACKGROUND_ENCODING_QP', config.ENCODING_QP + 10)
            background_video_path, bg_size = _encode_video_stream(background_paths, batch_dir, "background", bg_qp)
            bw_bytes += bg_size
        _enc_elapsed = (time.time() - _enc_start) * 1000
        print(f"[Timing-net] encode={_enc_elapsed:.1f}ms (roi+bg)")

        if roi_video_path:
            files = {'video': open(roi_video_path, 'rb')}
            if background_video_path:
                files['background'] = open(background_video_path, 'rb')
            data = {'metadata': json.dumps(metadata_list)}
            try:
                _upload_start = time.time()
                response = requests.Session().post(
                    f"http://{config.SERVER_HOST}/low",
                    files=files,
                    data=data
                )
                _upload_elapsed = (time.time() - _upload_start) * 1000
                print(f"[Timing-net] upload+server={_upload_elapsed:.1f}ms")
                try:
                    nC = float(response.content)
                except Exception:
                    nC = 0.0
            except Exception as e:
                print(f"Upload failed: {e}")
                nC = 0.0
            finally:
                for f in files.values():
                    try:
                        f.close()
                    except Exception:
                        pass
    except Exception as e:
        print(f"Encoding/Sending failed: {e}")
        nC = 0.0
    finally:
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)
        if background_paths:
            for p in background_paths:
                if os.path.exists(p):
                    os.remove(p)

    return bw_bytes, nC

def send_composite_batch(composite_paths, metadata_list, bw_bytes):
    if not composite_paths:
        return bw_bytes, 0.0

    nC = 0.0
    try:
        files = {}
        for idx, path in enumerate(composite_paths):
            bw_bytes += os.path.getsize(path)
            files[f'image_{idx}'] = open(path, 'rb')
        data = {'metadata': json.dumps(metadata_list)}
        try:
            response = requests.Session().post(
                f"http://{config.SERVER_HOST}/low_composite",
                files=files, data=data)
            try:
                nC = float(response.content)
            except Exception:
                nC = 0.0
        except Exception as e:
            print(f"Upload failed: {e}")
            nC = 0.0
        finally:
            for f in files.values():
                try:
                    f.close()
                except Exception:
                    pass
    finally:
        for p in composite_paths:
            if os.path.exists(p):
                os.remove(p)
    return bw_bytes, nC

_last_frame_key = None
_last_frame_img = None

def _get_frame_image(suffix):
    global _last_frame_key, _last_frame_img
    if _last_frame_key == suffix and _last_frame_img is not None:
        return _last_frame_img
    raw_path = os.path.join(config.DATASET_DIR, suffix)
    _last_frame_img = cv.imread(raw_path)
    _last_frame_key = suffix
    return _last_frame_img

def get_frame_dims(img_path):
    suffix = os.path.relpath(img_path, config.DATASET_DIR)
    img = _get_frame_image(suffix)
    if img is not None:
        h, w = img.shape[:2]
        return w, h
    return 0, 0

def cache_append(img):
    _ensure_dirs()
    pos = img['shape']
    suffix = img['name'].split('_', 1)[1] if '_' in img['name'] else img['name']
    raw = _get_frame_image(suffix)
    if raw is None:
        return
    seg_path = os.path.join(config.CACHE_DIR, img['name'])
    h, w = raw.shape[:2]
    padded = _expand_rect(
        pos,
        w,
        h,
        pad_ratio=getattr(config, 'ROI_PAD_RATIO', 0.0),
        pad_pix=getattr(config, 'ROI_PAD_PIX', 0)
    )
    if not padded:
        return
    x1, y1, x2, y2 = padded

    crop = raw[y1:y2, x1:x2]
    if crop.size == 0:
        return
    cv.imwrite(seg_path, crop, [cv.IMWRITE_PNG_COMPRESSION, 0])

def cache_pop(name):
    try:
        os.remove(os.path.join(config.CACHE_DIR, name))
    except FileNotFoundError:
        pass

def find_target_by_name(targets, name):
    for index, t in enumerate(targets):
        if t['name'] == name:
            return targets.pop(index)
    return -1

def cleanup_cache():
    try:
        shutil.rmtree(config.CACHE_DIR)
    except FileNotFoundError:
        return
    except OSError as e:
        print(f"Failed to clean cache: {e}")

import os
import shutil
import time
import math
import csv
import sys

try:
    from . import config
    from . import detector
    from . import net
    from . import track
    from . import features
    from .threshold import ThresholdManager
except ImportError:

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    import detector
    import net
    import track
    import features
    from threshold import ThresholdManager

def _normalize_frame_id(frame_id):
    if not frame_id:
        return ""
    stem = os.path.splitext(frame_id)[0]
    stem = stem.replace("\\", "/")
    return stem.lstrip("./")

def run():
    try:

        for d in [config.CACHE_DIR, config.TEMP_DIR]:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        if os.path.isfile(config.RESULT_BW_CSV):
            os.remove(config.RESULT_BW_CSV)
        with open(config.RESULT_BW_CSV, 'w+') as _:
            pass
        if os.path.isfile(config.RESULT_BW_TXT):
            os.remove(config.RESULT_BW_TXT)
        with open(config.RESULT_BW_TXT, 'w+') as _:
            pass
        features_csv = getattr(config, 'FEATURES_CSV', None)
        if features_csv:
            with open(features_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['frame_id', 's', 'd', 'u'])

        if not os.path.isdir(config.DATASET_DIR):
            os.makedirs(config.DATASET_DIR, exist_ok=True)

        def _collect_frames(root):
            collected = []
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    abs_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(abs_path, root)
                    collected.append(rel_path)
            collected.sort(key=lambda p: p.replace(os.sep, "/"))
            return collected

        frames = _collect_frames(config.DATASET_DIR)
        if not frames:
            print(f"数据目录为空: {config.DATASET_DIR}。请放入帧图像后重试。")
            return

        obs_window = getattr(config, 'OBSERVATION_WINDOW', 1)
        thr_mgr = ThresholdManager(
            use_anfis=getattr(config, 'USE_ANFIS', False),
            weights_path=getattr(config, 'ANFIS_WEIGHTS', None))
        print(f"Found {len(frames)} frames. Starting processing "
              f"(Paper Aligned Mode, obs_window={obs_window}, anfis={thr_mgr.active})...")
        last_frame = []
        persistence = {}
        current_seq = None
        first_frame_in_seq = False

        def _write_seq_ratio(seq_name, uploaded_bytes, original_bytes):
            ratio = (uploaded_bytes / original_bytes) if original_bytes > 0 else 0.0
            with open(config.RESULT_BW_TXT, 'a') as f:
                f.write(f"{seq_name}\t{ratio:.6f}\n")

        def flush_buffers(label="flush"):
            nonlocal bw, video_buffer_paths, video_buffer_meta, background_buffer_paths, seq_uploaded_bytes
            if not video_buffer_paths:
                return
            print(f"Flushing remaining {len(video_buffer_paths)} frames... ({label})")
            flush_start = time.perf_counter()
            bw_before = bw
            bw, _ = net.send_video_batch(video_buffer_paths, video_buffer_meta, bw, background_buffer_paths)
            seq_uploaded_bytes += (bw - bw_before)
            flush_ms = (time.perf_counter() - flush_start) * 1000.0
            video_buffer_paths = []
            video_buffer_meta = []
            background_buffer_paths = []
            print(f"[Timing] flush_send_video_batch={flush_ms:.2f}ms")

        def _collect_local_contrib(targets, high_thr):
            high = []
            for t in targets or []:
                shape = t.get('shape') or t.get('bbox')
                if not shape or len(shape) < 4:
                    continue
                try:
                    bbox = [float(shape[i]) for i in range(4)]
                    score = float(t.get('confidence', t.get('score', t.get('conf', 0.0))))
                    cls_raw = t.get('result', t.get('label', 0))
                    try:
                        cls_id = int(cls_raw)
                    except (TypeError, ValueError):
                        cls_id = int(float(cls_raw))
                except Exception:
                    continue
                if score >= high_thr:
                    high.append({'bbox': bbox, 'score': score, 'cls': cls_id})
            return {'high': high}

        video_buffer_paths = []
        video_buffer_meta = []
        background_buffer_paths = []
        bw = 0
        total_original_bytes = 0
        seq_uploaded_bytes = 0
        seq_original_bytes = 0

        for fname in frames:
            frame_start = time.perf_counter()
            frame_timer_cache = 0.0
            frame_timer_prepare = 0.0
            frame_timer_send = 0.0
            img_path = os.path.join(config.DATASET_DIR, fname)

            seq_key = fname.split(os.sep)[0] if os.sep in fname else ""
            if seq_key != current_seq:
                flush_buffers(label=f"seq_change:{current_seq or 'none'}-> {seq_key or 'root'}")
                if current_seq is not None:
                    _write_seq_ratio(current_seq or "root", seq_uploaded_bytes, seq_original_bytes)
                seq_uploaded_bytes = 0
                seq_original_bytes = 0
                last_frame = []
                persistence = {}
                thr_mgr.reset()
                current_seq = seq_key
                first_frame_in_seq = True
            else:
                first_frame_in_seq = False

            try:
                img_size = os.path.getsize(img_path)
                total_original_bytes += img_size
                seq_original_bytes += img_size
            except OSError:
                pass

            detect_start = time.perf_counter()
            raw_current = detector.detect(img_path)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            print(f"[Debug] frame {fname}: raw detections={len(raw_current)}")

            current_frame = []
            for idx, t in enumerate(raw_current):
                tgt = {
                    'name': f"{idx}_{fname}",
                    'shape': t[:4],
                    'confidence': t[4],
                    'result': t[5]
                }
                current_frame.append(tgt)

            _track_start = time.perf_counter()

            _last_frame_snapshot = {t['name']: t for t in last_frame}
            last_frame, current_frame, tracked = track.preprocess_data(last_frame, current_frame)
            _track_ms = (time.perf_counter() - _track_start) * 1000.0

            put_back = []
            upload_entries = []
            mid_targets_current = []

            def handle_high_conf(target, cache_key=None, container=None):
                if cache_key:
                    net.cache_pop(cache_key)
                dest = container if container is not None else put_back
                dest.append(target)

            cur_lt, cur_ut = thr_mgr.get()

            for pair in tracked:
                last_t = net.find_target_by_name(last_frame, pair[0])
                curr_t = net.find_target_by_name(current_frame, pair[1])

                if last_t == -1 or curr_t == -1:
                    continue

                conf = float(curr_t['confidence'])
                prev_count = persistence.pop(last_t['name'], 0)

                if conf >= cur_ut:
                    handle_high_conf(curr_t, cache_key=last_t['name'])

                elif conf >= cur_lt:
                    net.cache_pop(last_t['name'])
                    current_count = prev_count + 1
                    if current_count >= obs_window:

                        _cache_start = time.perf_counter()
                        net.cache_append(curr_t)
                        frame_timer_cache += (time.perf_counter() - _cache_start) * 1000.0
                        mid_targets_current.append(curr_t)
                        put_back.append(curr_t)
                    else:

                        persistence[curr_t['name']] = current_count
                        put_back.append(curr_t)

                else:

                    net.cache_pop(last_t['name'])

            for lost in last_frame:

                net.cache_pop(lost['name'])
                persistence.pop(lost['name'], None)

            next_frame = list(put_back)
            for nf in current_frame:

                conf = float(nf['confidence'])

                if conf >= cur_ut:

                    handle_high_conf(nf, container=next_frame)

                elif conf >= cur_lt:

                    persistence[nf['name']] = 1
                    next_frame.append(nf)

                else:

                    pass

            _anfis_start = time.perf_counter()
            _frame_w, _frame_h = net.get_frame_dims(img_path)
            _tracked_target_pairs = []
            for pair in tracked:
                prev_t = _last_frame_snapshot.get(pair[0])
                curr_t_name = pair[1]
                curr_t = next((t for t in next_frame if t['name'] == curr_t_name), None)
                if prev_t and curr_t:
                    _tracked_target_pairs.append((prev_t, curr_t))
            feat_s, feat_d, feat_u = features.compute_frame_features(
                next_frame, _tracked_target_pairs,
                _frame_w, _frame_h, cur_lt, cur_ut)
            if features_csv:
                with open(features_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([fname, f"{feat_s:.6f}", f"{feat_d:.6f}", f"{feat_u:.6f}"])

            thr_mgr.update(feat_s, feat_d, feat_u)
            _anfis_ms = (time.perf_counter() - _anfis_start) * 1000.0

            local_contrib = _collect_local_contrib(next_frame, cur_ut)
            local_high = local_contrib.get('high', [])

            upload_entries.append({'frame_id': fname, 'targets': list(mid_targets_current), 'local_high': local_high})

            use_composite = getattr(config, 'USE_COMPOSITE', False)

            if use_composite:

                for entry in upload_entries:
                    fid = entry['frame_id']
                    frame_targets = entry.get('targets') or []
                    entry_high = entry.get('local_high') or []
                    norm_fid = os.path.splitext(fid)[0].replace('\\', '/')

                    if frame_targets:

                        _prep_start = time.perf_counter()
                        frame_img_path = os.path.join(config.DATASET_DIR, fid)
                        comp_path, comp_size = net.prepare_composite_frame(
                            frame_targets, frame_img_path, fid)
                        frame_timer_prepare += (time.perf_counter() - _prep_start) * 1000.0
                        if comp_path:
                            meta = {
                                'frame_id': norm_fid,
                                'local_high': entry_high,
                                'targets': [{'name': t['name'], 'shape': t['shape'],
                                             'conf': t['confidence'], 'label': t['result']}
                                            for t in frame_targets],
                            }
                            video_buffer_paths.append(comp_path)
                            video_buffer_meta.append(meta)
                    elif entry_high:

                        fused_dir = os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), 'server', 'fused_txt')
                        fused_path = os.path.join(fused_dir, norm_fid + '.txt')
                        os.makedirs(os.path.dirname(fused_path), exist_ok=True)
                        with open(fused_path, 'w') as ff:
                            for h in entry_high:
                                bbox = h.get('bbox') or h.get('shape', [])
                                if len(bbox) >= 4:
                                    ff.write(f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},"
                                             f"{h.get('score', h.get('confidence', 0.0)):.2f},"
                                             f"{h.get('cls', h.get('label', 0))}\n")

                if len(video_buffer_paths) >= config.VIDEO_BATCH_SIZE:
                    print(f"Composite batch ({len(video_buffer_paths)} frames)...")
                    _send_start = time.perf_counter()
                    bw_before = bw
                    bw, _ = net.send_composite_batch(
                        video_buffer_paths, video_buffer_meta, bw)
                    seq_uploaded_bytes += (bw - bw_before)
                    frame_timer_send += (time.perf_counter() - _send_start) * 1000.0
                    video_buffer_paths = []
                    video_buffer_meta = []

            else:

                emitted_sparse = False
                if not upload_entries and not local_high:
                    pass

                for entry in upload_entries:
                    fid = entry['frame_id']
                    frame_targets = entry.get('targets') or []
                    entry_high = entry.get('local_high') or []
                    _prep_start = time.perf_counter()
                    if frame_targets:
                        sparse_path, meta = net.prepare_sparse_frame(frame_targets)
                    else:
                        sparse_path, meta = net.prepare_sparse_frame([], frame_id=fid)
                    frame_timer_prepare += (time.perf_counter() - _prep_start) * 1000.0
                    if sparse_path:
                        bg_idx = None
                        if frame_targets:
                            frame_img = os.path.join(config.DATASET_DIR, fid)
                            bg_path = net.prepare_background_frame(frame_img, fid)
                            if bg_path:
                                bg_idx = len(background_buffer_paths)
                                background_buffer_paths.append(bg_path)
                        meta['local_high'] = entry_high
                        meta['background_index'] = bg_idx
                        video_buffer_paths.append(sparse_path)
                        video_buffer_meta.append(meta)
                        emitted_sparse = True

                if len(video_buffer_paths) >= config.VIDEO_BATCH_SIZE:
                    print(f"Buffer full ({len(video_buffer_paths)} frames), encoding video...")
                    _send_start = time.perf_counter()
                    bw_before = bw
                    bw, _ = net.send_video_batch(video_buffer_paths, video_buffer_meta, bw, background_buffer_paths)
                    seq_uploaded_bytes += (bw - bw_before)
                    frame_timer_send += (time.perf_counter() - _send_start) * 1000.0
                    video_buffer_paths = []
                    video_buffer_meta = []
                    background_buffer_paths = []

            last_frame = next_frame

            if total_original_bytes > 0:
                ratio = bw / total_original_bytes
                uploaded_kb = bw / 1024
                original_kb = total_original_bytes / 1024
                print(
                    f"相对带宽占用: {ratio:.6f} ({ratio * 100:.2f}%)  已上传: {uploaded_kb:.2f}KB / 原始: {original_kb:.2f}KB")

            with open(config.RESULT_BW_CSV, 'a', newline='') as f:
                csv.writer(f).writerow([bw])

            frame_elapsed = (time.perf_counter() - frame_start) * 1000.0
            print(f"[Timing] detect={detect_ms:.2f}ms track={_track_ms:.2f}ms "
                  f"anfis={_anfis_ms:.2f}ms cache={frame_timer_cache:.2f}ms "
                  f"prepare={frame_timer_prepare:.2f}ms send={frame_timer_send:.2f}ms "
                  f"total={frame_elapsed:.2f}ms")

        if video_buffer_paths:
            flush_buffers(label="final")
            if total_original_bytes > 0:
                ratio = bw / total_original_bytes
                print(f"最终相对带宽占用: {ratio:.6f} ({ratio * 100:.2f}%)")
        if current_seq is not None:
            _write_seq_ratio(current_seq or "root", seq_uploaded_bytes, seq_original_bytes)

        net.cleanup_cache()
    finally:
        pass

if __name__ == "__main__":
    run()

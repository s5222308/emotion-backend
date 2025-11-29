import sys
from typing import List, Dict
from collections import Counter
import json
import os

def _bbox_xyxy(d: dict, center_coords: bool = False):
    """Return (x1,y1,x2,y2). Assumes coordinates in same units as inputs (e.g. percent 0..100)."""
    if center_coords:
        cx, cy, w, h = d["x"], d["y"], d["width"], d["height"]
        x1 = cx - w/2
        y1 = cy - h/2
    else:
        x1, y1 = d["x"], d["y"]
        w, h = d["width"], d["height"]
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2

def _iou(detA: dict, detB: dict, center_coords: bool = False):
    ax1, ay1, ax2, ay2 = _bbox_xyxy(detA, center_coords)
    bx1, by1, bx2, by2 = _bbox_xyxy(detB, center_coords)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    areaB = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = areaA + areaB - inter
    return 0.0 if union <= 0 else inter / union

def _load_config():
    """Load fusion config to get track label strategy"""
    try:
        config_path = "config/fusion_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def _select_track_label(track: List[Dict], strategy: str = "highest_confidence") -> str:
    """
    Select the best label for a track based on strategy.
    
    Strategies:
    - "highest_confidence": Use label with highest score
    - "most_common": Use most frequently occurring label
    - "most_recent": Use label from last frame
    - "first": Use label from first frame (original behavior)
    """
    if not track:
        return "neutral"
    
    if strategy == "first":
        return track[0].get("label", "neutral")
    
    if strategy == "most_recent":
        return track[-1].get("label", "neutral")
    
    if strategy == "most_common":
        labels = [det.get("label") for det in track if det.get("label")]
        if not labels:
            return "neutral"
        label_counts = Counter(labels)
        return label_counts.most_common(1)[0][0]
    
    if strategy == "highest_confidence":
        # Find detection with highest score
        best_det = max(track, key=lambda d: d.get("score", 0.0))
        return best_det.get("label", "neutral")
    
    # Default fallback
    return track[0].get("label", "neutral")

def merge_detections_into_tracks(
    detections: List[Dict],
    iou_thresh: float = 0.35,
    max_frame_gap: int = 1,
    label_match: bool = True,
    center_coords: bool = False
) -> List[List[Dict]]:
    """
    Greedy track-building:
      - Sort detections by frame
      - For each detection, try to attach to an existing track where:
          * frame gap <= max_frame_gap
          * IoU >= iou_thresh
          * (optional) label matches
      - Otherwise start a new track
    Returns list of tracks; each track is a list of detection dicts (in frame order).
    """
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: int(d["frame"]))
    tracks = []

    for det in dets:
        f = int(det["frame"])
        matched_track = None
        for t in reversed(tracks):
            last = t["frames"][-1]
            last_frame = int(last["frame"])
            if f - last_frame > max_frame_gap:
                continue
            if label_match and det.get("label") != last.get("label"):
                continue
            if _iou(det, last, center_coords=center_coords) >= iou_thresh:
                matched_track = t
                break

        if matched_track is not None:
            matched_track["frames"].append(det)
        else:
            tracks.append({"frames": [det]})

    return [t["frames"] for t in tracks]

def _format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"

def build_labelstudio_regions_from_tracks(
    tracks: List[List[Dict]],
    frames_count: int,
    duration: float,
    include_meta: bool = False,
    clamp_end_frame_to_last: bool = True,
    label_strategy: str = None
) -> List[Dict]:
    """
    For each track emit a compact region with keyframes for all detections.
    
    Args:
        label_strategy: How to select track label. Options:
            - "highest_confidence": Use label with highest score (default)
            - "most_common": Use most frequently occurring label
            - "most_recent": Use label from last frame
            - "first": Use label from first frame
    """
    # Load strategy from config if not provided
    if label_strategy is None:
        config = _load_config()
        label_strategy = config.get("track_label_strategy", "highest_confidence")
    
    regions = []
    frame_duration = (duration / frames_count) if (frames_count and duration) else (1/30)

    for track in tracks:
        start_det = track[0]
        end_det = track[-1]
        start_frame = int(start_det["frame"])
        end_frame = int(end_det["frame"])

        if clamp_end_frame_to_last and frames_count:
            end_frame = min(end_frame, frames_count)

        start_time = start_frame * frame_duration
        end_time = min(duration, (end_frame + 1) * frame_duration) if duration else (end_frame + 1) * frame_duration

        # Mean score for the track
        scores = [float(d.get("score", 0.0)) for d in track if "score" in d]
        region_score = float(sum(scores) / len(scores)) if scores else 0.0

        # Create keyframes for ALL detections in track
        seq = []
        for det in track:
            frame_num = int(det["frame"])
            time = frame_num * frame_duration
            seq.append({
                "enabled": True,
                "frame": frame_num,
                "time": time,
                "x": float(det["x"]),
                "y": float(det["y"]),
                "width": float(det["width"]),
                "height": float(det["height"]),
                "score": float(det.get("score", region_score))
            })
        
        # Add final disabled keyframe at end
        seq.append({
            "enabled": False,
            "frame": end_frame + 1,
            "time": end_time,
            "x": float(end_det["x"]),
            "y": float(end_det["y"]),
            "width": float(end_det["width"]),
            "height": float(end_det["height"]),
            "score": float(end_det.get("score", region_score))
        })

        # Select track label using configured strategy
        track_label = _select_track_label(track, label_strategy)
        print(f"DEBUG: Track has {len(track)} frames, labels: {[f.get('label') for f in track[:5]]}, selected: {track_label}", file=sys.stderr, flush=True)

        value = {
            "labels": [track_label],
            "timestamps": {
                "start": _format_time(start_time),
                "end": _format_time(end_time)
            },
            "sequence": seq
        }

        if include_meta:
            value["framesCount"] = frames_count
            value["duration"] = duration

        region = {
            "from_name": "box",
            "to_name": "video",
            "type": "videorectangle",
            "origin": "manual",
            "score": region_score,
            "value": value
        }
        regions.append(region)

    return regions

def frame_by_frame_regions(results, frames_count=None, duration=None):
    tracks = merge_detections_into_tracks(results, iou_thresh=0.35, max_frame_gap=1, label_match=True)  
    regions = build_labelstudio_regions_from_tracks(tracks, frames_count, duration, include_meta=False)
    return regions
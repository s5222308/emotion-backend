import pytest

# --- Fixtures ---
@pytest.fixture
def sample_detections():
    return [
        {"frame": 1, "time": 0.03, "x": 10, "y": 10, "width": 20, "height": 20, "label": "happy", "score": 0.9},
        {"frame": 2, "time": 0.06, "x": 11, "y": 11, "width": 20, "height": 20, "label": "happy", "score": 0.85},
        {"frame": 5, "time": 0.15, "x": 50, "y": 50, "width": 20, "height": 20, "label": "sad", "score": 0.8},
    ]

# --- Tests ---
def test_bbox_xyxy():
    from group_frames import _bbox_xyxy
    box = {"x": 10, "y": 20, "width": 30, "height": 40}
    x1, y1, x2, y2 = _bbox_xyxy(box)
    assert (x1, y1, x2, y2) == (10, 20, 40, 60)

def test_iou_overlap():
    from group_frames import _iou
    boxA = {"x": 10, "y": 10, "width": 20, "height": 20}
    boxB = {"x": 15, "y": 15, "width": 20, "height": 20}
    iou = _iou(boxA, boxB)
    assert 0.0 < iou < 1.0

def test_merge_detections_into_tracks(sample_detections):
    from group_frames import merge_detections_into_tracks
    tracks = merge_detections_into_tracks(sample_detections, iou_thresh=0.3, max_frame_gap=2)
    assert isinstance(tracks, list)
    assert len(tracks) == 2
    assert all(isinstance(t, list) for t in tracks)

def test_format_time():
    from group_frames import _format_time
    assert _format_time(0.0) == "00:00:00:000"
    assert _format_time(61.234) == "00:01:01:234"
    assert _format_time(3661.999) == "01:01:01:998"

def test_build_labelstudio_regions_from_tracks(sample_detections):
    from group_frames import merge_detections_into_tracks, build_labelstudio_regions_from_tracks
    tracks = merge_detections_into_tracks(sample_detections)
    regions = build_labelstudio_regions_from_tracks(tracks, frames_count=100, duration=3.0)
    assert isinstance(regions, list)
    assert all("value" in r and "sequence" in r["value"] for r in regions)
    assert all("timestamps" in r["value"] for r in regions)

def test_frame_by_frame_regions(sample_detections):
    from group_frames import frame_by_frame_regions
    regions = frame_by_frame_regions(sample_detections, frames_count=100, duration=3.0)
    assert isinstance(regions, list)
    assert all("from_name" in r and "value" in r for r in regions)
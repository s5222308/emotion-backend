import pytest
import tempfile
import os
import json
from unittest.mock import patch

# --- Fixtures ---
@pytest.fixture
def dummy_video_results():
    return [
        {"frame": 1, "time": 0.1, "x": 0, "y": 0, "width": 10, "height": 10, "label": "happy", "score": 0.9},
        {"frame": 2, "time": 0.6, "x": 0, "y": 0, "width": 10, "height": 10, "label": "sad", "score": 0.8},
    ]

@pytest.fixture
def dummy_audio_results():
    return [
        {"start_time": 0.0, "end_time": 0.5, "emotion_label": "happy", "confidence_score": 0.7},
        {"start_time": 0.5, "end_time": 1.0, "emotion_label": "angry", "confidence_score": 0.6},
    ]

# --- Tests ---
def test_confidence_fusion_basic(dummy_video_results, dummy_audio_results):
    from multimodal_fushion import ConfidenceFusionEngine
    fusion = ConfidenceFusionEngine()
    result = fusion.fuse(dummy_video_results, dummy_audio_results)
    assert isinstance(result, list)
    assert all("label" in r and "score" in r for r in result)

def test_temporal_smoothing(dummy_video_results):
    from multimodal_fushion import TemporalSmoother
    smoother = TemporalSmoother(beta=0.5, window_size=0.5, fps=30)
    smoothed = smoother.smooth(dummy_video_results)
    assert isinstance(smoothed, list)
    assert all("label" in r and "score" in r for r in smoothed)

def test_emotion_multimodal_fushion_fuse(dummy_video_results, dummy_audio_results):
    from multimodal_fushion import EmotionFusionEngine
    engine = EmotionFusionEngine(smoothing=True)
    result = engine.fuse(dummy_video_results, dummy_audio_results)
    assert isinstance(result, list)
    assert all("label" in r and "score" in r for r in result)

def test_update_params_clamping():
    from multimodal_fushion import EmotionFusionEngine
    engine = EmotionFusionEngine(smoothing=False)
    engine.update_params(beta=1.5, floor_prob=1e-20, video_bias=-5, audio_bias=20)
    params = engine.get_params()
    assert params["beta"] == 1.0
    assert params["floor_prob"] >= 1e-12
    assert params["video_bias"] == 0.0
    assert params["audio_bias"] == 10.0

@patch("multimodal_fushion.os.makedirs")
@patch("multimodal_fushion.open")
def test_config_save_and_load(mock_open, mock_makedirs):
    from multimodal_fushion import EmotionFusionEngine
    tmp_path = "config/test_fusion_config.json"
    engine = EmotionFusionEngine(config_path=tmp_path)
    engine.params["debug"] = True
    engine._save_config()
    mock_open.assert_called_with(tmp_path, "w")

def test_reset_smoother(dummy_video_results):
    from multimodal_fushion import EmotionFusionEngine
    engine = EmotionFusionEngine(smoothing=True)
    engine.fuse(dummy_video_results, [])
    engine.reset()
    assert engine.smoother._ema_probs is None
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from Functions.Shared_objects.shared_objects import is_processing, global_stop_event

# --- Fixtures ---
@pytest.fixture(autouse=True)
def reset_flags():
    is_processing.clear()
    global_stop_event.clear()

# --- Dummy mocks ---
class DummyModel:
    def __init__(self):
        self.config = MagicMock()
        self.config.id2label = {0: "neutral", 1: "happy"}
    def to(self, device): return self
    def eval(self): pass
    def __call__(self, **inputs):
        class Output:
            logits = torch.tensor([[0.1, 0.9]])
        return Output()

class DummyExtractor:
    def __init__(self):
        self.sampling_rate = 16000
    def __call__(self, audio, sampling_rate, return_tensors):
        return {"input_values": torch.tensor([[0.0]])}

# --- Tests ---
@patch("audio_predictor.get_audio_emotion_model", return_value="dummy-audio-model")
@patch("audio_predictor.AutoModelForAudioClassification.from_pretrained", return_value=DummyModel())
@patch("audio_predictor.AutoFeatureExtractor.from_pretrained", return_value=DummyExtractor())
def test_initialization(mock_extractor, mock_model, mock_get_model):
    from audio_predictor import AudioEmotionRecognizer
    AudioEmotionRecognizer._instance = None
    recognizer = AudioEmotionRecognizer()
    assert recognizer.segment_duration == 0.5
    assert recognizer.id2label == {0: "neutral", 1: "happy"}

@patch("audio_predictor.set_audio_emotion_model", lambda name: None)
@patch("audio_predictor.AutoModelForAudioClassification.from_pretrained", return_value=DummyModel())
@patch("audio_predictor.AutoFeatureExtractor.from_pretrained", return_value=DummyExtractor())
def test_set_model(mock_model, mock_extractor):
    from audio_predictor import AudioEmotionRecognizer
    recognizer = AudioEmotionRecognizer()
    recognizer.set_model("new-model")
    assert recognizer.id2label == {0: "neutral", 1: "happy"}

def test_change_segment_duration_valid():
    from audio_predictor import AudioEmotionRecognizer
    recognizer = AudioEmotionRecognizer()
    recognizer.change_segment_duration(3.0)
    assert recognizer.segment_duration == 3.0

def test_change_segment_duration_invalid():
    from audio_predictor import AudioEmotionRecognizer
    recognizer = AudioEmotionRecognizer()
    recognizer.change_segment_duration(-1.0)
    assert recognizer.segment_duration != -1.0
    recognizer.change_segment_duration(12.0)
    assert recognizer.segment_duration != 12.0

@patch("audio_predictor.subprocess.run")
@patch("audio_predictor.librosa.load", return_value=(np.ones(32000), 16000))
@patch("audio_predictor.get_audio_emotion_model", return_value="dummy-audio-model")
@patch("audio_predictor.AutoModelForAudioClassification.from_pretrained", return_value=DummyModel())
@patch("audio_predictor.AutoFeatureExtractor.from_pretrained", return_value=DummyExtractor())
@patch("audio_predictor.os.remove", lambda path: None)
@patch("audio_predictor.tempfile.NamedTemporaryFile")
def test_process_audio_minimal(mock_tempfile, *mocks):
    mock_tempfile.return_value.name = "dummy.wav"
    mock_tempfile.return_value.close = lambda: None

    from audio_predictor import AudioEmotionRecognizer
    AudioEmotionRecognizer._instance = None
    recognizer = AudioEmotionRecognizer()
    result = recognizer.ProcessAudio("dummy.mp4")
    assert isinstance(result, list)
    assert all("emotion_label" in r and "confidence_score" in r for r in result)
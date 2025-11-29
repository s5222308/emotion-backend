import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from Functions.Shared_objects.shared_objects import is_processing, global_stop_event

# --- Dummy mocks ---
class DummyXY:
    """Mimics a torch tensor-like object with .tolist()."""
    def __init__(self, coords):
        self._coords = coords
    def tolist(self):
        return list(self._coords)

class DummyBox:
    def __init__(self):
        self.xyxy = [DummyXY([0, 0, 10, 10])]
        self.conf = [0.9]
        self.cls = [0]

class DummyYOLO:
    def __init__(self, path):
        self.names = {0: "happy"}
    def to(self, device):
        return self
    def __call__(self, frame, conf=0.5):
        class Result:
            def __init__(self):
                self.boxes = [DummyBox()]
        return [Result()]

class DummyCapture:
    def __init__(self):
        # actual numpy arrays so slicing works
        self.frames = [np.zeros((100, 100, 3), dtype=np.uint8),
                       np.zeros((100, 100, 3), dtype=np.uint8)]
        self.index = 0
    def isOpened(self):
        return self.index < len(self.frames)
    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None
    def get(self, prop):
        # CAP_PROP_FRAME_COUNT = 7, CAP_PROP_FPS = 5
        if prop == 7:
            return len(self.frames)
        elif prop == 5:
            return 1
        return 0
    def release(self):
        pass

# --- Fixtures ---
@pytest.fixture(autouse=True)
def reset_flags():
    is_processing.clear()
    global_stop_event.clear()

# --- Tests ---
@patch("Functions.Helpers.set_models.get_face_model", return_value="dummy-face.pt")
@patch("Functions.Helpers.set_models.get_face_emotion_model", return_value="dummy-emotion.pt")
@patch("Functions.Helpers.set_parameters.get_param", side_effect=lambda k: {
    "frame_step": 1,
    "face_conf": 0.5,
    "emotion_conf": 0.5
}[k])
@patch("predictor.YOLO", new=DummyYOLO)
def test_initialization(mock_get_param, mock_get_face, mock_get_emotion):
    from predictor import EmotionPredictor
    predictor = EmotionPredictor()
    assert predictor.device in ["cpu", "cuda"]
    assert predictor.frame_step == 1
    assert isinstance(predictor.face_conf, float) and 0 <= predictor.face_conf <= 1
    assert isinstance(predictor.emotion_conf, float) and 0 <= predictor.emotion_conf <= 1
    assert predictor.labels == {0: "happy"}

@patch("Functions.Helpers.set_parameters.set_param", lambda k, v: None)
def test_set_frame_step_valid():
    from predictor import EmotionPredictor
    predictor = EmotionPredictor()
    predictor.set_frame_step(3)
    assert predictor.frame_step == 3

def test_set_frame_step_invalid():
    from predictor import EmotionPredictor
    predictor = EmotionPredictor()
    predictor.set_frame_step(0)
    assert predictor.frame_step != 0

def test_set_face_conf_valid():
    from predictor import EmotionPredictor
    predictor = EmotionPredictor()
    predictor.set_face_conf(0.8)
    assert predictor.face_conf == 0.8

def test_set_emotion_conf_valid():
    from predictor import EmotionPredictor
    predictor = EmotionPredictor()
    predictor.set_emotion_conf(0.3)
    assert predictor.emotion_conf == 0.3

@patch("Functions.Helpers.set_models.set_face_model", lambda name: None)
@patch("Functions.Helpers.set_models.set_face_emotion_model", lambda name: None)
def test_set_models_blocked():
    from predictor import EmotionPredictor
    predictor = EmotionPredictor()
    is_processing.set()
    result = predictor.set_models("new-face.pt", "new-emotion.pt")
    assert result is False
    is_processing.clear()

@patch("cv2.VideoCapture", return_value=DummyCapture())
@patch("predictor.YOLO", new=DummyYOLO)
@patch("Functions.Helpers.set_models.get_face_model", return_value="dummy-face.pt")
@patch("Functions.Helpers.set_models.get_face_emotion_model", return_value="dummy-emotion.pt")
@patch("Functions.Helpers.set_parameters.get_param", side_effect=lambda k: {
    "frame_step": 1,
    "face_conf": 0.5,
    "emotion_conf": 0.5
}[k])
def test_run_minimal(*_):
    from predictor import EmotionPredictor
    predictor = EmotionPredictor()
    results, frame_count, duration = predictor.run("dummy.mp4")
    assert isinstance(results, list)
    assert frame_count == 2
    assert duration == 2.0
    assert all("label" in r and "score" in r for r in results)

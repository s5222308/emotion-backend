import os
import json
import logging as lg
from Functions.Helpers.verify_models import verify_correct_models

modeConfig_path = "/config/models_config.json"

def _load_config():
    """Load current config or return defaults if file doesn't exist."""
    if os.path.exists(modeConfig_path):
        try:
            with open(modeConfig_path, "r") as f:
                return json.load(f)
        except Exception as e:
            lg.error(f"Failed to read {modeConfig_path}, resetting config: {e}")
    return {"face": None, "face_emotion": None, "audio_emotion": None}

def _save_config(config):
    """Save config dict to file."""
    try:
        os.makedirs(os.path.dirname(modeConfig_path), exist_ok=True)
        with open(modeConfig_path, "w") as f:
            json.dump(config, f, indent=2)
        lg.info(f"Model config updated: {config}")
    except Exception as e:
        lg.error(f"Failed to save config: {e}")

def set_face_model(face_model):
    config = _load_config()
    # Validate using current values for other models
    face, face_emotion, audio_emotion = verify_correct_models(
        face_model,
        config.get("face_emotion"),
        config.get("audio_emotion")
    )
    config.update({"face": face, "face_emotion": face_emotion, "audio_emotion": audio_emotion})
    _save_config(config)

def set_face_emotion_model(face_emotion_model):
    config = _load_config()
    face, face_emotion, audio_emotion = verify_correct_models(
        config.get("face"),
        face_emotion_model,
        config.get("audio_emotion")
    )
    config.update({"face": face, "face_emotion": face_emotion, "audio_emotion": audio_emotion})
    _save_config(config)

def set_audio_emotion_model(audio_emotion_model):
    config = _load_config()
    face, face_emotion, audio_emotion = verify_correct_models(
        config.get("face"),
        config.get("face_emotion"),
        audio_emotion_model
    )
    config.update({"face": face, "face_emotion": face_emotion, "audio_emotion": audio_emotion})
    _save_config(config)


def get_face_model():
    config = _load_config()
    face, _, _ = verify_correct_models(
        config.get("face"),
        config.get("face_emotion"),
        config.get("audio_emotion")
    )
    return face

def get_face_emotion_model():
    config = _load_config()
    _, face_emotion, _ = verify_correct_models(
        config.get("face"),
        config.get("face_emotion"),
        config.get("audio_emotion")
    )
    return face_emotion

def get_audio_emotion_model():
    config = _load_config()
    _, _, audio_emotion = verify_correct_models(
        config.get("face"),
        config.get("face_emotion"),
        config.get("audio_emotion")
    )
    return audio_emotion

def get_all_models():
    config = _load_config()
    face, face_emotion, audio_emotion = verify_correct_models(
        config.get("face"),
        config.get("face_emotion"),
        config.get("audio_emotion")
    )
    return {
        "face": face,
        "face_emotion": face_emotion,
        "audio_emotion": audio_emotion
    }
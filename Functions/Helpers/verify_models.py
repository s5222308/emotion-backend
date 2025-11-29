import logging as lg
import os


FACE_MODELS_DIR = "/app/models/face_models"
EMOTION_MODELS_DIR = "/app/models/emotion_models"
AUDIO_MODELS_DIR = "/app/models/audio_models"


def _get_models_from_dir(path: str, category: str):
    """Helper to read models from a directory with fault tolerance."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model directory not found for {category}: {path}")

    models = [f for f in os.listdir(path) if not f.startswith(".")]

    if not models:
        raise RuntimeError(f"No models available in {category} directory: {path}")

    return models


def get_face_models():
    return _get_models_from_dir(FACE_MODELS_DIR, "face")


def get_emotion_models():
    return _get_models_from_dir(EMOTION_MODELS_DIR, "emotion")


def get_audio_models():
    return _get_models_from_dir(AUDIO_MODELS_DIR, "audio")


# Load available models at startup
try:
    valid_face_models = get_face_models()
    valid_emotion_models = get_emotion_models()
    valid_audio_models = get_audio_models()
except Exception as e:
    lg.critical(f"Failed to initialize model lists: {e}")
    raise




def verify_correct_models(face, face_emotion, audio_emotion):
    if face not in valid_face_models:
        lg.warning(f"Invalid face model received: '{face}'. Falling back to default: '{valid_face_models[0]}'")
        face = valid_face_models[0]
    else:
        lg.info(f"Valid face model selected: '{face}'")

    if face_emotion not in valid_emotion_models:
        lg.warning(f"Invalid emotion model received: '{face_emotion}'. Falling back to default: '{valid_emotion_models[0]}'")
        face_emotion = valid_emotion_models[0]
    else:
        lg.info(f"Valid emotion model selected: '{face_emotion}'")

    if audio_emotion not in valid_audio_models:
        lg.warning(f"Invalid audio model received: '{audio_emotion}'. Falling back to default: '{valid_audio_models[0]}'")
        audio_emotion = valid_audio_models[0]
    else:
        lg.info(f"Valid audio model selected: '{audio_emotion}'")

    return face, face_emotion, audio_emotion


def list_available_models():
    """
    Returns a dictionary of all available model names for each category.
    """
    return {
        "face": valid_face_models,
        "face_emotion": valid_emotion_models,
        "audio_emotion": valid_audio_models
    }
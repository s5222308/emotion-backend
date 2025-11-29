import logging as lg
from Functions.Helpers.set_models import get_face_model, get_face_emotion_model, get_audio_emotion_model
from Functions.Helpers.verify_models import list_available_models  # hypothetical function
from Functions.Helpers.set_parameters import get_all_params  # import your param getter

def get_models():
    """
    Retrieve current model names and current predictor parameters.
    Includes all available models and verified parameters.
    """
    try:
        current_face = get_face_model()
        current_face_emotion = get_face_emotion_model()
        current_audio_emotion = get_audio_emotion_model()

        lg.info(f"Loaded model config: face='{current_face}', face_emotion='{current_face_emotion}', audio_emotion='{current_audio_emotion}'")

    except Exception as e:
        lg.error(f"Error loading model config: {e}")
        from Functions.Helpers.verify_models import verify_correct_models
        from Functions.Helpers.set_models import set_face_model, set_face_emotion_model, set_audio_emotion_model

        # fallback to defaults
        current_face, current_face_emotion, current_audio_emotion = verify_correct_models(None, None, None)
        set_face_model(current_face)
        set_face_emotion_model(current_face_emotion)
        set_audio_emotion_model(current_audio_emotion)

    # Get all available models
    try:
        available_models = list_available_models()
    except Exception as e:
        lg.warning(f"Could not fetch available models: {e}")
        available_models = {
            "face": [current_face],
            "face_emotion": [current_face_emotion],
            "audio_emotion": [current_audio_emotion]
        }

    # Get all current parameters
    try:
        current_params = get_all_params()
    except Exception as e:
        lg.warning(f"Could not fetch parameters: {e}")
        current_params = {}

    return {
        "current": {
            "face": current_face,
            "face_emotion": current_face_emotion,
            "audio_emotion": current_audio_emotion,
            "parameters": current_params
        },
        "available": available_models
    }

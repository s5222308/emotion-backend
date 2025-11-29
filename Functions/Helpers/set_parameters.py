import os
import json
import logging as lg
import sys
from Functions.Shared_objects.shared_objects import is_processing

PARAMS_CONFIG_PATH = "/config/predictor_params.json"

# Batch mode for efficient bulk updates
_batch_mode = False
_batch_updates = {}

DEFAULT_PARAMS = {
    "segment_duration": 0.5,
    "face_conf": 0.75,
    "emotion_conf": 0.5,
    "frame_step": 10,
    # AU backends
    "use_openface": False,
    "use_libreface": False,
    # Video processing parameters
    "processing_mode": "sparse",
    "temporal_context_window": 0.5,
    "dense_window": False,
    "window_size": 1.0,
    # Audio processing parameters
    "audio_sliding_window": False,
    "audio_window_size": 1.0,
    "audio_window_overlap": 0.5
}

PARAM_BOUNDS = {
    "segment_duration": (0.1, 10.0),
    "face_conf": (0.1, 1.0),
    "emotion_conf": (0.1, 1.0),
    "frame_step": (1, 100),
    "temporal_context_window": (0.1, 2.0),
    "window_size": (0.1, 5.0),
    "audio_window_size": (0.5, 5.0),
    "audio_window_overlap": (0.0, 0.9)
}

def _load_params():
    """Load current parameters or create defaults if file doesn't exist or is invalid."""
    if os.path.exists(PARAMS_CONFIG_PATH):
        try:
            with open(PARAMS_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            lg.error(f"Failed to read {PARAMS_CONFIG_PATH}, resetting config: {e}")

    params = DEFAULT_PARAMS.copy()
    _save_params(params)
    return params

def _save_params(params):
    """Save parameters dict to JSON."""
    try:
        os.makedirs(os.path.dirname(PARAMS_CONFIG_PATH), exist_ok=True)
        with open(PARAMS_CONFIG_PATH, "w") as f:
            json.dump(params, f, indent=2)
        print(f"DEBUG: Parameters saved: {params}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"ERROR: Failed to save parameters config: {e}", file=sys.stderr, flush=True)


def enable_batch_mode():
    """Enable batch mode to defer saves until flush_batch_updates is called"""
    global _batch_mode, _batch_updates
    _batch_mode = True
    _batch_updates = {}


def flush_batch_updates():
    """Save all batched parameter updates and disable batch mode"""
    global _batch_mode, _batch_updates
    _batch_mode = False
    
    if _batch_updates:
        params = _load_params()
        params.update(_batch_updates)
        _save_params(params)
        _batch_updates = {}


def set_param(key, value):
    """
    Generic setter for any parameter.
    Enforces bounds if they are defined.
    Respects is_processing flag.
    Supports batch mode for efficient bulk updates.
    """
    global _batch_updates
    
    if is_processing.is_set():
        lg.warning(f"Attempted to set {key} while processing - skipping.")
        return

    if key in PARAM_BOUNDS:
        min_val, max_val = PARAM_BOUNDS[key]
        if not isinstance(value, (int, float)):
            lg.warning(f"Invalid type for {key}: {type(value)}. Skipping update.")
            return
        if value < min_val or value > max_val:
            lg.warning(f"{key} must be between {min_val} and {max_val}. Skipping update.")
            return

    # In batch mode, accumulate updates instead of saving immediately
    if _batch_mode:
        _batch_updates[key] = value
        return

    # Normal mode: save immediately
    params = _load_params()
    params[key] = value
    _save_params(params)


def get_param(key):
    """
    Generic getter for any parameter.
    Falls back to default if missing or invalid.
    """
    if is_processing.is_set():
        lg.warning(f"Attempted to get {key} while processing - using last known value from JSON.")
    
    params = _load_params()
    value = params.get(key, DEFAULT_PARAMS.get(key))
    
    if key in PARAM_BOUNDS:
        min_val, max_val = PARAM_BOUNDS[key]
        if not isinstance(value, (int, float)) or value < min_val or value > max_val:
            lg.warning(f"Invalid {key} in config: {value}, resetting to default {DEFAULT_PARAMS[key]}")
            value = DEFAULT_PARAMS[key]
            if not is_processing.is_set():
                set_param(key, value)
    
    return value

def get_all_params():
    """Return all parameters from config, merged with defaults."""
    params = _load_params()
    # Merge with defaults to ensure all keys exist
    all_params = DEFAULT_PARAMS.copy()
    all_params.update(params)
    return all_params
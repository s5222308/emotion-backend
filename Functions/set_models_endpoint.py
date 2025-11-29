import sys
from flask import request, jsonify

from Functions.Shared_objects.model_instances import predictor, recognizer
from Functions.Helpers.set_parameters import enable_batch_mode, flush_batch_updates


def set_models_endpoint():
    """
    Flask endpoint to set models and parameters at once via POST request.

    Expected JSON (frontend sends nested `parameters` object):

    {
        "face": "yolov11s-face.pt",
        "face_emotion": "yolo11s-emotion.pt",
        "audio_emotion": "wave2vec-english-speech-recognition-by-r-f",
        "parameters": {
            "frame_step": 10,
            "face_conf": 0.75,
            "emotion_conf": 0.5,
            "segment_duration": 0.5,
            "processing_mode": "dense_window",
            "temporal_context_window": 0.5,
            "dense_window": true,
            "window_size": 1.0,
            "audio_sliding_window": true,
            "audio_window_size": 1.0,
            "audio_window_overlap": 0.0,
            "use_openface": true,
            "use_libreface": false
        }
    }
    """
    try:
        payload = request.get_json(force=True)
        if not payload:
            print("DEBUG: No JSON payload provided", file=sys.stderr, flush=True)
            return jsonify({"error": "No JSON payload provided"}), 400

        print("DEBUG: Raw payload received:", payload, file=sys.stderr, flush=True)

        # ---- Flatten nested "parameters" if present ----
        parameters = payload.pop("parameters", {})
        if parameters:
            print(
                "DEBUG: Found nested 'parameters', merging into top-level:",
                parameters,
                file=sys.stderr,
                flush=True,
            )
            payload.update(parameters)

        print("DEBUG: Flattened payload:", payload, file=sys.stderr, flush=True)

        # ---- Batch mode: accumulate all param changes, then save once ----
        enable_batch_mode()

        updated = {}
        skipped = {}

        # ---------- MODELS ----------
        face = payload.get("face")
        face_emotion = payload.get("face_emotion")
        audio_emotion = payload.get("audio_emotion")

        # Face + face_emotion models
        print(
            "DEBUG: Setting face/face_emotion models:",
            face,
            face_emotion,
            file=sys.stderr,
            flush=True,
        )
        if face or face_emotion:
            ok = predictor.set_models(
                face_model_name=face,
                face_emotion_model_name=face_emotion,
            )
            if ok:
                if face is not None:
                    updated["face"] = face
                if face_emotion is not None:
                    updated["face_emotion"] = face_emotion
                print(
                    "DEBUG: Face models updated successfully",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                skipped["face/face_emotion"] = "processing active"
                print(
                    "DEBUG: Skipped updating face models because processing is active",
                    file=sys.stderr,
                    flush=True,
                )

        # Audio emotion model
        if audio_emotion:
            print(
                "DEBUG: Setting audio emotion model:",
                audio_emotion,
                file=sys.stderr,
                flush=True,
            )
            try:
                recognizer.set_model(audio_emotion)
                updated["audio_emotion"] = audio_emotion
                print(
                    "DEBUG: Audio emotion model updated successfully",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                skipped["audio_emotion"] = str(e)
                print(
                    f"DEBUG: Skipped audio emotion model due to error: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        # ---------- CORE PARAMETERS ----------
        core_params = [
            ("frame_step", predictor.set_frame_step),
            ("face_conf", predictor.set_face_conf),
            ("emotion_conf", predictor.set_emotion_conf),
            # audio segment length (seconds)
            ("segment_duration", recognizer.change_segment_duration),
        ]

        for param_name, setter in core_params:
            if param_name in payload:
                value = payload[param_name]
                print(
                    f"DEBUG: Setting parameter '{param_name}' to {value}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    setter(value)
                    updated[param_name] = value
                    print(
                        f"DEBUG: Parameter '{param_name}' updated successfully",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as e:
                    skipped[param_name] = str(e)
                    print(
                        f"DEBUG: Skipped parameter '{param_name}' due to error: {e}",
                        file=sys.stderr,
                        flush=True,
                    )

        # ---------- VIDEO WINDOW + AU BACKENDS ----------
        au_and_window_params = [
            ("dense_window", predictor.set_dense_window),
            ("window_size", predictor.set_window_size),
            ("use_openface", predictor.set_use_openface),
            ("use_libreface", predictor.set_use_libreface),
            ("processing_mode", predictor.set_processing_mode),
            ("temporal_context_window", predictor.set_temporal_context_window),
            ("audio_sliding_window", recognizer.set_sliding_window),
        ]

        for param_name, setter in au_and_window_params:
            if param_name in payload:
                value = payload[param_name]
                try:
                    print(
                        f"DEBUG: Setting parameter '{param_name}' to {value}",
                        file=sys.stderr,
                        flush=True,
                    )
                    setter(value)
                    updated[param_name] = value
                    print(
                        f"DEBUG: Parameter '{param_name}' updated successfully",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as e:
                    skipped[param_name] = str(e)
                    print(
                        f"WARNING: Failed to set parameter '{param_name}': {e}",
                        file=sys.stderr,
                        flush=True,
                    )

        # ---------- AUDIO WINDOW PARAMS (combined setter) ----------
        if "audio_window_size" in payload or "audio_window_overlap" in payload:
            window_size = payload.get("audio_window_size")
            overlap = payload.get("audio_window_overlap")
            print(
                f"DEBUG: Setting audio window params - size: {window_size}, overlap: {overlap}",
                file=sys.stderr,
                flush=True,
            )
            try:
                recognizer.set_window_params(
                    window_size=window_size,
                    overlap=overlap,
                )
                if window_size is not None:
                    updated["audio_window_size"] = window_size
                if overlap is not None:
                    updated["audio_window_overlap"] = overlap
                print(
                    "DEBUG: Audio window params updated successfully",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                skipped["audio_window_params"] = str(e)
                print(
                    f"DEBUG: Skipped audio window params due to error: {e}",
                    file=sys.stderr,
                    flush=True,
                )

        print(
            "DEBUG: Final update summary - updated:",
            updated,
            "skipped:",
            skipped,
            file=sys.stderr,
            flush=True,
        )

        # Flush all batched parameter updates (single JSON write)
        flush_batch_updates()

        return jsonify(
            {
                "status": "success" if updated else "failed",
                "updated": updated,
                "skipped": skipped,
            }
        )

    except Exception as e:
        # Make sure to flush even on error so batch state isn't stuck
        flush_batch_updates()
        print(
            "DEBUG: Failed to set models/params:",
            e,
            file=sys.stderr,
            flush=True,
        )
        return jsonify({"error": str(e)}), 500

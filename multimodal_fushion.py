import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional


class TripleFusionEngine:
    """
    Triple fusion engine:
    1. Video (YOLO emotion)
    2. Audio (Transformer emotion)
    3. LibreFace FER (facial expression recognition)
    
    OpenFace AUs and gaze are passed through for debugging but not used in fusion.
    """

    def __init__(self, floor_prob=1e-6, debug=False,
                 video_bias=1.0, audio_bias=1.0, libreface_bias=1.0):
        self.floor_prob = floor_prob
        self.debug = debug
        self.video_bias = video_bias
        self.audio_bias = audio_bias
        self.libreface_bias = libreface_bias

    def update_params(self, floor_prob=None, debug=None,
                      audio_bias=None, video_bias=None, libreface_bias=None):
        if floor_prob is not None:
            self.floor_prob = floor_prob
        if debug is not None:
            self.debug = debug
        if audio_bias is not None:
            self.audio_bias = audio_bias
        if video_bias is not None:
            self.video_bias = video_bias
        if libreface_bias is not None:
            self.libreface_bias = libreface_bias

    def _single_to_probs(self, label, score, label_set):
        """Convert a single label+score into a full probability dict."""
        probs = {l: self.floor_prob for l in label_set}
        probs[label] = max(score, self.floor_prob)
        return probs

    def fuse(self, video_results, audio_results):
        """
        Fuse video, audio, and LibreFace FER results per frame.
        LibreFace predictions are embedded in video_results as libreface_label/libreface_score.
        """
        fused_results = []

        # Collect all possible labels
        label_set = set()

        for v in video_results:
            if "face_emotion_label" in v:
                label_set.add(v["face_emotion_label"])
            label_set.add(v["label"])
            if "libreface_label" in v:
                label_set.add(v["libreface_label"])

        for a in audio_results:
            label_set.add(a["emotion_label"])

        label_set = list(label_set)

        def find_audio_for_time(t):
            return next(
                (a for a in audio_results if a["start_time"] <= t < a["end_time"]),
                None,
            )

        for v in video_results:
            # Video probabilities (face emotion from YOLO)
            if "face_emotion_label" in v and "face_emotion_score" in v:
                video_label = v["face_emotion_label"]
                video_score = v["face_emotion_score"]
            else:
                video_label = v["label"]
                video_score = v["score"]

            video_probs = self._single_to_probs(video_label, video_score, label_set)
            video_weight = video_score * self.video_bias

            # Audio prediction for this time
            audio = find_audio_for_time(v["time"])
            if audio:
                audio_probs = self._single_to_probs(
                    audio["emotion_label"],
                    audio["confidence_score"],
                    label_set,
                )
                audio_weight = audio["confidence_score"] * self.audio_bias
            else:
                audio_probs = {l: self.floor_prob for l in label_set}
                audio_weight = 0.0

            # LibreFace FER prediction (the third modality)
            libreface_probs = None
            libreface_weight = 0.0

            if "libreface_label" in v and "libreface_score" in v:
                libreface_probs = self._single_to_probs(
                    v["libreface_label"],
                    v["libreface_score"],
                    label_set,
                )
                libreface_weight = v["libreface_score"] * self.libreface_bias

            # ---- Fusion logic ----
            if libreface_probs:
                # Triple fusion: video + audio + LibreFace
                total_weight = video_weight + audio_weight + libreface_weight
                if total_weight > 0:
                    vw = video_weight / total_weight
                    aw = audio_weight / total_weight
                    lfw = libreface_weight / total_weight

                    fused_probs = {
                        lbl: vw * video_probs[lbl]
                        + aw * audio_probs[lbl]
                        + lfw * libreface_probs[lbl]
                        for lbl in label_set
                    }
                else:
                    fused_probs = {
                        lbl: (video_probs[lbl] + audio_probs[lbl] + libreface_probs[lbl]) / 3
                        for lbl in label_set
                    }
            else:
                # Dual fusion: video + audio only
                total_weight = video_weight + audio_weight
                if total_weight > 0:
                    vw = video_weight / total_weight
                    aw = audio_weight / total_weight

                    fused_probs = {
                        lbl: vw * video_probs[lbl] + aw * audio_probs[lbl]
                        for lbl in label_set
                    }
                else:
                    fused_probs = video_probs

            # Normalize probabilities
            total = sum(fused_probs.values())
            if total > 0:
                fused_probs = {k: v / total for k, v in fused_probs.items()}

            # Final prediction
            label = max(fused_probs, key=fused_probs.get)
            score = fused_probs[label]

            fused_result = {
                "frame": v["frame"],
                "time": v["time"],
                "x": v["x"],
                "y": v["y"],
                "width": v["width"],
                "height": v["height"],
                "label": label,
                "score": score,
            }

            # Preserve face emotion predictions
            if "face_emotion_label" in v:
                fused_result["face_emotion_label"] = v["face_emotion_label"]
                fused_result["face_emotion_score"] = v["face_emotion_score"]

            # Preserve LibreFace predictions
            if "libreface_label" in v:
                fused_result["libreface_label"] = v["libreface_label"]
                fused_result["libreface_score"] = v["libreface_score"]
            if "libreface_aus" in v:
                fused_result["libreface_aus"] = v["libreface_aus"]

            # Preserve OpenFace data (not used in fusion, just for debugging/comparison)
            if "openface_label" in v:
                fused_result["openface_label"] = v["openface_label"]
                fused_result["openface_score"] = v["openface_score"]
            if "openface_aus" in v:
                fused_result["openface_aus"] = v["openface_aus"]
            if "gaze_direction" in v:
                fused_result["gaze_direction"] = v["gaze_direction"]

            # Debug info
            if self.debug:
                debug_info = {
                    "video_probs": video_probs,
                    "audio_probs": audio_probs if audio else None,
                    "fused_probs": fused_probs,
                    "weights": {
                        "video": video_weight,
                        "audio": audio_weight if audio else 0,
                    },
                }

                if libreface_probs:
                    debug_info["libreface_probs"] = libreface_probs
                    debug_info["weights"]["libreface"] = libreface_weight

                if "libreface_aus" in v:
                    debug_info["libreface_aus"] = v["libreface_aus"]
                if "openface_aus" in v:
                    debug_info["openface_aus"] = v["openface_aus"]
                if "gaze_direction" in v:
                    debug_info["gaze_direction"] = v["gaze_direction"]

                fused_result["debug"] = debug_info

            fused_results.append(fused_result)

        return fused_results


class ConfidenceFusionEngine:
    """Dual fusion engine: video + audio (no LibreFace)."""

    def __init__(self, floor_prob=1e-6, debug=False,
                 video_bias=1.0, audio_bias=1.0):
        self.floor_prob = floor_prob
        self.debug = debug
        self.video_bias = video_bias
        self.audio_bias = audio_bias

    def update_params(self, floor_prob=None, debug=None,
                      audio_bias=None, video_bias=None):
        if floor_prob is not None:
            self.floor_prob = floor_prob
        if debug is not None:
            self.debug = debug
        if audio_bias is not None:
            self.audio_bias = audio_bias
        if video_bias is not None:
            self.video_bias = video_bias

    def _single_to_probs(self, label, score, label_set):
        probs = {l: self.floor_prob for l in label_set}
        probs[label] = max(score, self.floor_prob)
        return probs

    def fuse(self, video_results, audio_results):
        fused_results = []

        label_set = list({
            *[v["label"] for v in video_results],
            *[a["emotion_label"] for a in audio_results],
        })

        def find_audio_for_time(t):
            return next(
                (a for a in audio_results if a["start_time"] <= t < a["end_time"]),
                None,
            )

        for v in video_results:
            audio = find_audio_for_time(v["time"])
            video_probs = self._single_to_probs(v["label"], v["score"], label_set)
            video_weight = v["score"] * self.video_bias

            if audio:
                audio_probs = self._single_to_probs(
                    audio["emotion_label"],
                    audio["confidence_score"],
                    label_set,
                )
                audio_weight = audio["confidence_score"] * self.audio_bias

                total_weight = video_weight + audio_weight
                vw = video_weight / total_weight if total_weight > 0 else 0.5
                aw = audio_weight / total_weight if total_weight > 0 else 0.5

                fused_probs = {
                    lbl: vw * video_probs[lbl] + aw * audio_probs[lbl]
                    for lbl in label_set
                }
            else:
                fused_probs = video_probs

            total = sum(fused_probs.values())
            if total > 0:
                fused_probs = {k: v / total for k, v in fused_probs.items()}

            label = max(fused_probs, key=fused_probs.get)
            score = fused_probs[label]

            fused_result = {
                "frame": v["frame"],
                "time": v["time"],
                "x": v["x"],
                "y": v["y"],
                "width": v["width"],
                "height": v["height"],
                "label": label,
                "score": score,
            }

            if self.debug:
                fused_result["debug"] = {
                    "video_probs": video_probs,
                    "audio_probs": audio_probs if audio else None,
                    "fused_probs": fused_probs,
                    "weights": {
                        "video": video_weight,
                        "audio": audio["confidence_score"] if audio else None,
                    },
                }

            fused_results.append(fused_result)

        return fused_results


class TemporalSmoother:
    def __init__(self, beta=0.8, window_size=0.5, fps=30):
        self.beta = beta
        self.window_frames = int(window_size * fps)
        self._ema_probs = None
        self._prev_label = None
        self._label_streak = 0

    def update_params(self, beta=None, window_size=None, fps=None):
        if beta is not None:
            self.beta = beta
        if window_size is not None or fps is not None:
            if window_size is None:
                window_size = self.window_frames / fps
            if fps is None:
                fps = self.window_frames / window_size
            self.window_frames = int(window_size * fps)

    def reset(self):
        self._ema_probs = None
        self._prev_label = None
        self._label_streak = 0

    def smooth(self, video_results):
        smoothed_results = []
        label_set = {r["label"] for r in video_results}

        for r in video_results:
            probs = {l: 1e-6 for l in label_set}
            probs[r["label"]] = max(r["score"], 1e-6)

            if self._ema_probs is None:
                self._ema_probs = probs
            else:
                self._ema_probs = {
                    k: self.beta * self._ema_probs.get(k, 0)
                    + (1 - self.beta) * probs[k]
                    for k in label_set
                }

            label = max(self._ema_probs, key=self._ema_probs.get)
            score = self._ema_probs[label]

            if label == self._prev_label:
                self._label_streak += 1
            else:
                self._label_streak = 1

            if (
                self._prev_label
                and label != self._prev_label
                and self._label_streak < self.window_frames
            ):
                label = self._prev_label
                score = self._ema_probs[label]

            self._prev_label = label
            smoothed_results.append({**r, "label": label, "score": score})

        return smoothed_results


class EmotionFusionEngine:
    """Main fusion engine with configuration support."""

    DEFAULT_CONFIG = {
        "beta": 0.8,
        "min_duration": 0.5,
        "fps": 30,
        "floor_prob": 1e-6,
        "debug": False,
        "video_bias": 1.0,
        "audio_bias": 1.0,
        "libreface_bias": 1.0,
    }

    PARAM_LIMITS = {
        "beta": (0.0, 1.0),
        "min_duration": (0.1, 5.0),
        "fps": (1, 120),
        "floor_prob": (1e-12, 1e-2),
        "debug": (False, True),
        "video_bias": (0.0, 10.0),
        "audio_bias": (0.0, 10.0),
        "libreface_bias": (0.0, 10.0),
    }

    def __init__(self, config_path="config/fusion_config.json", smoothing=False):
        self.config_path = config_path
        self.smoothing = smoothing
        self.params = self._load_config()

        if not os.path.exists(self.config_path):
            self._save_config()

        self._init_components()

    def _init_components(self):
        beta = self.params["beta"]
        min_duration = self.params["min_duration"]
        fps = self.params["fps"]
        floor_prob = self.params["floor_prob"]
        debug = self.params["debug"]
        video_bias = self.params["video_bias"]
        audio_bias = self.params["audio_bias"]
        libreface_bias = self.params.get("libreface_bias", 1.0)

        self.smoother = (
            TemporalSmoother(beta=beta, window_size=min_duration, fps=fps)
            if self.smoothing
            else None
        )

        # Check if LibreFace is enabled for triple fusion
        use_libreface = False
        try:
            from Functions.Shared_objects.model_instances import predictor
            use_libreface = getattr(predictor, "use_libreface", False)
        except Exception as e:
            print(
                f"[WARNING] Could not inspect predictor flags, defaulting to dual fusion: {e}",
                file=sys.stderr,
            )

        if use_libreface:
            self.fusion = TripleFusionEngine(
                floor_prob=floor_prob,
                debug=debug,
                video_bias=video_bias,
                audio_bias=audio_bias,
                libreface_bias=libreface_bias,
            )
        else:
            self.fusion = ConfidenceFusionEngine(
                floor_prob=floor_prob,
                debug=debug,
                video_bias=video_bias,
                audio_bias=audio_bias,
            )

    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    cfg = json.load(f)
                print(f"[INFO] Loaded config from {self.config_path}", file=sys.stderr)
                merged_config = {**self.DEFAULT_CONFIG, **cfg}
                return merged_config
            except Exception as e:
                print(
                    f"[WARNING] Failed to load config, using defaults: {e}",
                    file=sys.stderr,
                )
        return self.DEFAULT_CONFIG.copy()

    def _save_config(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.params, f, indent=2)
            print(f"[INFO] Saved config to {self.config_path}", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}", file=sys.stderr)

    def get_params(self):
        return self._load_config()

    def update_params(self, **kwargs):
        updated = False
        for k, v in kwargs.items():
            if k not in self.DEFAULT_CONFIG:
                print(f"[WARNING] Ignoring unknown param: {k}", file=sys.stderr)
                continue

            if k in self.PARAM_LIMITS:
                lo, hi = self.PARAM_LIMITS[k]
                if isinstance(lo, (int, float)):
                    v = max(lo, min(v, hi))

            if self.params.get(k) != v:
                print(
                    f"[INFO] Updating {k}: {self.params.get(k)} -> {v}",
                    file=sys.stderr,
                )
                self.params[k] = v
                updated = True

        if updated:
            self._init_components()
            self._save_config()

    def reset(self):
        if self.smoother:
            self.smoother.reset()

    def fuse(self, video_results, audio_results):
        """Main fusion method."""
        if self.smoother:
            video_results = self.smoother.smooth(video_results)

        return self.fusion.fuse(video_results, audio_results)
from Functions.Helpers.set_parameters import set_param, get_param
import cv2
from Functions.Helpers.set_models import get_face_emotion_model, get_face_model, set_face_emotion_model, set_face_model
import torch
from ultralytics import YOLO
import sys
import logging as lg
from Functions.Shared_objects.shared_objects import global_stop_event, is_processing
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import time
from typing import List, Dict, Optional, Tuple
from openface_video_processor import OpenFaceVideoProcessor
from libreface_video_processor import LibreFaceVideoProcessor

# LibreFace outputs capitalized emotion names - map to YOLO's lowercase labels
LIBREFACE_LABEL_MAP = {
    "Happiness": "happy",
    "Sadness": "sad",
    "Anger": "anger",
    "Fear": "fear",
    "Disgust": "disgust",
    "Surprise": "surprise",
    "Neutral": "neutral",
    "Contempt": "content",
    "Happy": "happy",
    "Sad": "sad",
    "Angry": "anger",
    "Fearful": "fear",
    "Disgusted": "disgust",
    "Surprised": "surprise",
}

class EmotionPredictor:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        face_model_name = get_face_model()
        emotion_model_name = get_face_emotion_model() 
        
        base_dir = "/app/models"
        face_model_path = f"{base_dir}/face_models/{face_model_name}"
        emotion_model_path = f"{base_dir}/emotion_models/{emotion_model_name}"

        # Core parameters
        self.frame_step = get_param("frame_step")
        self.face_conf = get_param("face_conf")
        self.emotion_conf = get_param("emotion_conf")
        
        # Processing mode parameters
        try:
            self.processing_mode = get_param("processing_mode")
        except:
            self.processing_mode = "sparse"
            
        try:
            self.temporal_context_window = get_param("temporal_context_window")
        except:
            self.temporal_context_window = 0.5
            
        try:
            self.dense_window = get_param("dense_window")
        except:
            self.dense_window = False
            
        try:
            self.window_size = get_param("window_size")
        except:
            self.window_size = 1.0

        # flags for OpenFace vs LibreFace
        try:
            self.use_openface = get_param("use_openface")
        except Exception:
            self.use_openface = False

        try:
            self.use_libreface = get_param("use_libreface")
        except Exception:
            self.use_libreface = False

        # Initialize models
        self.face_model = YOLO(face_model_path)
        self.face_model.to(self.device)
        self.emotion_model = YOLO(emotion_model_path)
        self.emotion_model.to(self.device)
        self.labels = self.emotion_model.names

        # AU video processors + DataFrames
        self.openface_df = None
        self.libreface_df = None

        self.openface_processor = OpenFaceVideoProcessor()
        self.libreface_processor = LibreFaceVideoProcessor()

    def set_models(self, face_model_name=None, face_emotion_model_name=None):
        if is_processing.is_set():
            lg.warning("Cannot change models while processing is active.")
            return False

        base_dir = "/app/models"

        if face_model_name:
            set_face_model(face_model_name)
            face_model_path = f"{base_dir}/face_models/{face_model_name}"
            lg.info(f"Reloading face model: {face_model_path}")
            self.face_model = YOLO(face_model_path)
            self.face_model.to(self.device)

        if face_emotion_model_name:
            set_face_emotion_model(face_emotion_model_name)
            emotion_model_path = f"{base_dir}/emotion_models/{face_emotion_model_name}"
            lg.info(f"Reloading face emotion model: {emotion_model_path}")
            self.emotion_model = YOLO(emotion_model_path)
            self.emotion_model.to(self.device)
            self.labels = self.emotion_model.names

        return True
    
    def set_frame_step(self, new_step: int):
        if is_processing.is_set():
            lg.warning("Cannot change frame_step while processing is active")
            return
        if new_step < 1:
            lg.warning("frame_step must be at least 1")
            return
        self.frame_step = new_step
        set_param("frame_step", new_step)
        lg.info(f"frame_step updated to {new_step}")

    def set_face_conf(self, new_conf: float):
        if is_processing.is_set():
            lg.warning("Cannot change face_conf while processing is active")
            return
        if not 0.0 <= new_conf <= 1.0:
            lg.warning("face_conf must be between 0.0 and 1.0")
            return
        self.face_conf = new_conf
        set_param("face_conf", new_conf)
        lg.info(f"face_conf updated to {new_conf}")

    def set_emotion_conf(self, new_conf: float):
        if is_processing.is_set():
            lg.warning("Cannot change emotion_conf while processing is active")
            return
        if not 0.0 <= new_conf <= 1.0:
            lg.warning("emotion_conf must be between 0.0 and 1.0")
            return
        self.emotion_conf = new_conf
        set_param("emotion_conf", new_conf)
        lg.info(f"emotion_conf updated to {new_conf}")

    def set_processing_mode(self, mode: str):
        if is_processing.is_set():
            lg.warning("Cannot change processing_mode while processing is active")
            return
        
        valid_modes = ['sparse', 'temporal_context', 'dense_window']
        if mode not in valid_modes:
            lg.warning(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
            return
        
        self.processing_mode = mode
        set_param("processing_mode", mode)
        lg.info(f"processing_mode set to '{mode}'")
        
        if mode == 'dense_window':
            self.dense_window = True
            set_param("dense_window", True)
        else:
            self.dense_window = False
            set_param("dense_window", False)

    def set_temporal_context_window(self, window_seconds: float):
        if is_processing.is_set():
            lg.warning("Cannot change temporal_context_window while processing is active")
            return
        if window_seconds <= 0 or window_seconds > 2.0:
            lg.warning("temporal_context_window must be between 0 and 2.0 seconds")
            return
        self.temporal_context_window = window_seconds
        set_param("temporal_context_window", window_seconds)
        lg.info(f"temporal_context_window updated to {window_seconds}s")

    def set_dense_window(self, enabled: bool):
        if is_processing.is_set():
            lg.warning("Cannot change dense_window while processing is active")
            return
        
        if enabled:
            self.set_processing_mode('dense_window')
        else:
            self.set_processing_mode('sparse')

    def set_window_size(self, size: float):
        if is_processing.is_set():
            lg.warning("Cannot change window_size while processing is active")
            return
        if size <= 0 or size > 5.0:
            lg.warning("window_size must be between 0 and 5.0 seconds")
            return
        self.window_size = size
        set_param("window_size", size)
        lg.info(f"window_size updated to {size}")

    def set_use_openface(self, enabled: bool):
        if is_processing.is_set():
            lg.warning("Cannot change use_openface while processing is active")
            return
        self.use_openface = bool(enabled)
        set_param("use_openface", self.use_openface)
        lg.info(f"use_openface set to {self.use_openface}")

    def set_use_libreface(self, enabled: bool):
        if is_processing.is_set():
            lg.warning("Cannot change use_libreface while processing is active")
            return
        self.use_libreface = bool(enabled)
        set_param("use_libreface", self.use_libreface)
        lg.info(f"use_libreface set to {self.use_libreface}")

    def _get_libreface_row_for_frame(self, frame_number: int):
        """Map a video frame index to the corresponding LibreFace row."""
        if self.libreface_df is None:
            return None

        df = self.libreface_df

        # LibreFace CSV uses 'frame_idx' column (0-based)
        if hasattr(df, "columns") and "frame_idx" in df.columns:
            try:
                matches = df[df["frame_idx"] == frame_number]
                if not matches.empty:
                    return matches.iloc[0]
            except Exception as e:
                lg.warning(f"[LibreFace] Failed to filter by frame_idx: {e}")

        return None

    def get_au_predictions_separated(self, frame_number: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Get AU predictions from both OpenFace and LibreFace separately.
        
        Returns:
            Tuple of (openface_pred, libreface_pred) - either can be None
        """
        openface_pred = None
        libreface_pred = None

        # OpenFace
        if self.use_openface and self.openface_df is not None:
            result = self.openface_processor.get_frame_aus(self.openface_df, frame_number)
            if result:
                openface_pred = {
                    "label": result.get("label", "unknown"),
                    "score": result.get("score", 0.0),
                    "aus": result.get("au", {}),
                    "gaze_direction": result.get("gaze", {}).get("direction", "unknown"),
                }

        # LibreFace
        if self.use_libreface and self.libreface_df is not None:
            row = self._get_libreface_row_for_frame(frame_number)
            if row is not None:
                # Grab AU intensity columns
                aus = {}
                for col in getattr(row, "index", []):
                    if isinstance(col, str) and col.startswith("au_") and col.endswith("_intensity"):
                        try:
                            aus[col] = float(row[col])
                        except Exception:
                            continue

                # Get raw label and map to YOLO format
                raw_label = "unknown"
                if hasattr(row, "__getitem__") and "facial_expression" in row.index:
                    raw_label = row["facial_expression"]
                
                mapped_label = LIBREFACE_LABEL_MAP.get(raw_label, raw_label.lower())

                # Get actual probability from LibreFace
                prob = 0.85  # fallback
                if hasattr(row, "__getitem__") and "facial_expression_prob" in row.index:
                    try:
                        prob = float(row["facial_expression_prob"])
                    except Exception:
                        pass

                libreface_pred = {
                    "label": mapped_label,
                    "score": prob,
                    "aus": aus,
                }

        return openface_pred, libreface_pred

    def _add_au_data_to_result(self, result: Dict, frame_number: int) -> None:
        """Add separated OpenFace and LibreFace data to result dict."""
        openface_pred, libreface_pred = self.get_au_predictions_separated(frame_number)
        
        if openface_pred:
            result["openface_label"] = openface_pred["label"]
            result["openface_score"] = openface_pred["score"]
            result["openface_aus"] = openface_pred["aus"]
            result["gaze_direction"] = openface_pred["gaze_direction"]
        
        if libreface_pred:
            result["libreface_label"] = libreface_pred["label"]
            result["libreface_score"] = libreface_pred["score"]
            result["libreface_aus"] = libreface_pred["aus"]

    def run(self, video_path: str) -> Tuple[List, int, float]:
        """Main entry point - chooses processing mode based on configuration"""
        mode = self.processing_mode
        
        if self.dense_window and mode == 'sparse':
            mode = 'dense_window'
        
        if mode == 'temporal_context':
            lg.info(f"Using temporal context processing (context_window={self.temporal_context_window}s)")
            return self.run_temporal_context(video_path)
        elif mode == 'dense_window':
            lg.info("Using dense window processing")
            return self.run_dense_window(video_path)
        else:
            lg.info("Using sparse sampling")
            return self.run_sparse(video_path)

    def run_sparse(self, video_path: str) -> Tuple[List, int, float]:
        """Sparse sampling method"""
        frame_step = self.frame_step

        self.openface_df = None
        self.libreface_df = None

        if self.use_openface:
            try:
                lg.info("[Predictor] Processing video with OpenFace...")
                self.openface_df = self.openface_processor.process_video(video_path)
                if self.openface_df is not None:
                    lg.info(f"[Predictor] ✓ OpenFace processed {len(self.openface_df)} frames")
            except Exception as e:
                lg.warning(f"[Predictor] OpenFace processing failed: {e}")
                self.openface_df = None

        if self.use_libreface:
            try:
                lg.info("[Predictor] Processing video with LibreFace...")
                self.libreface_df = self.libreface_processor.process_video(video_path)
                if self.libreface_df is not None:
                    lg.info(f"[Predictor] ✓ LibreFace processed {len(self.libreface_df)} rows")
            except Exception as e:
                lg.warning(f"[Predictor] LibreFace processing failed: {e}")
                self.libreface_df = None

        cap = cv2.VideoCapture(video_path)
        
        try:
            is_processing.set()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 1
            duration = frame_count / fps

            results = []
            frame_idx = -1

            while cap.isOpened():
                if global_stop_event.is_set():
                    lg.info("Aborting video analysis")
                    return None, None, None
                    
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                if frame_idx % frame_step != 0:
                    continue

                faces = self.face_model(frame, conf=self.face_conf)

                for det in faces:
                    if global_stop_event.is_set():
                        return None, None, None
                        
                    for box in det.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                            continue

                        try:
                            emo_res = self.emotion_model(crop, conf=self.emotion_conf)[0]
                            if emo_res is None or emo_res.boxes is None or len(emo_res.boxes) == 0:
                                continue

                            best = max(emo_res.boxes, key=lambda b: b.conf[0])
                            cls_id = int(best.cls[0])
                            face_emotion_label = self.labels.get(cls_id, "unknown")
                            face_emotion_score = float(best.conf[0])
                        except Exception as e:
                            print(f"WARNING: Emotion detection failed: {e}", file=sys.stderr, flush=True)
                            continue

                        result = {
                            "frame": frame_idx + 1,
                            "time": (frame_idx + 1) * (duration / frame_count),
                            "x": (x1 / frame.shape[1]) * 100,
                            "y": (y1 / frame.shape[0]) * 100,
                            "width": ((x2 - x1) / frame.shape[1]) * 100,
                            "height": ((y2 - y1) / frame.shape[0]) * 100,
                            "label": face_emotion_label,
                            "score": face_emotion_score,
                            "face_emotion_label": face_emotion_label,
                            "face_emotion_score": face_emotion_score
                        }
                        
                        if self.use_openface or self.use_libreface:
                            self._add_au_data_to_result(result, frame_idx)
                        
                        results.append(result)
                        
            return results, frame_count, duration
            
        finally:
            is_processing.clear()
            cap.release()

    def run_temporal_context(self, video_path: str) -> Tuple[List, int, float]:
        """Temporal context mode"""
        self.openface_df = None
        self.libreface_df = None

        if self.use_openface:
            try:
                lg.info("[Predictor] Processing video with OpenFace...")
                self.openface_df = self.openface_processor.process_video(video_path)
                if self.openface_df is not None:
                    lg.info(f"[Predictor] ✓ OpenFace processed {len(self.openface_df)} frames")
            except Exception as e:
                lg.warning(f"[Predictor] OpenFace processing failed: {e}")
                self.openface_df = None

        if self.use_libreface:
            try:
                lg.info("[Predictor] Processing video with LibreFace...")
                self.libreface_df = self.libreface_processor.process_video(video_path)
                if self.libreface_df is not None:
                    lg.info(f"[Predictor] ✓ LibreFace processed {len(self.libreface_df)} rows")
            except Exception as e:
                lg.warning(f"[Predictor] LibreFace processing failed: {e}")
                self.libreface_df = None

        cap = cv2.VideoCapture(video_path)
        
        try:
            is_processing.set()
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            context_frames_count = int(self.temporal_context_window * fps)
            half_context = context_frames_count // 2
            
            lg.info(f"Temporal context mode: context_window={self.temporal_context_window}s "
                   f"({context_frames_count} frames), frame_step={self.frame_step}")
            
            sampled_indices = list(range(0, total_frames, self.frame_step))
            results = []
            
            for sample_idx in sampled_indices:
                if global_stop_event.is_set():
                    return None, None, None
                
                start_idx = max(0, sample_idx - half_context)
                end_idx = min(total_frames, sample_idx + half_context + 1)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                context_frames = []
                for _ in range(end_idx - start_idx):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    context_frames.append(frame)
                
                if not context_frames:
                    continue
                
                center_frame_idx = sample_idx - start_idx
                if center_frame_idx >= len(context_frames):
                    center_frame_idx = len(context_frames) - 1
                
                timestamp = sample_idx / fps
                prediction = self._analyze_temporal_context(
                    context_frames,
                    center_frame_idx,
                    sample_idx + 1,
                    timestamp,
                    (video_height, video_width)
                )
                
                if prediction:
                    results.append(prediction)
            
            return results, total_frames, duration
            
        finally:
            is_processing.clear()
            cap.release()

    def run_dense_window(self, video_path: str) -> Tuple[List, int, float]:
        """Dense window processing"""
        self.openface_df = None
        self.libreface_df = None

        if self.use_openface:
            try:
                lg.info("[Predictor] Processing video with OpenFace...")
                self.openface_df = self.openface_processor.process_video(video_path)
                if self.openface_df is not None:
                    lg.info(f"[Predictor] ✓ OpenFace processed {len(self.openface_df)} frames")
            except Exception as e:
                lg.warning(f"[Predictor] OpenFace processing failed: {e}")
                self.openface_df = None

        if self.use_libreface:
            try:
                lg.info("[Predictor] Processing video with LibreFace...")
                self.libreface_df = self.libreface_processor.process_video(video_path)
                if self.libreface_df is not None:
                    lg.info(f"[Predictor] ✓ LibreFace processed {len(self.libreface_df)} rows")
            except Exception as e:
                lg.warning(f"[Predictor] LibreFace processing failed: {e}")
                self.libreface_df = None

        cap = cv2.VideoCapture(video_path)

        try:
            is_processing.set()

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            window_frames = int(self.window_size * fps)
            overlap_frames = window_frames // 2

            results = []
            frame_buffer = []
            frame_indices = []
            frame_idx = -1

            lg.info(f"Dense window mode: window={self.window_size}s ({window_frames} frames), frame_step={self.frame_step}")

            while cap.isOpened():
                if global_stop_event.is_set():
                    return None, None, None

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                
                if frame_idx % self.frame_step != 0:
                    continue

                frame_buffer.append(frame)
                frame_indices.append(frame_idx)

                if len(frame_buffer) >= window_frames:
                    window_results = self.analyze_window(
                        frame_buffer, frame_indices, frame_count, duration
                    )
                    aggregated = self.aggregate_window_predictions(window_results)
                    if aggregated:
                        results.extend(aggregated)

                    frame_buffer = frame_buffer[overlap_frames:]
                    frame_indices = frame_indices[overlap_frames:]

            if frame_buffer:
                window_results = self.analyze_window(
                    frame_buffer, frame_indices, frame_count, duration
                )
                aggregated = self.aggregate_window_predictions(window_results)
                if aggregated:
                    results.extend(aggregated)

            return results, frame_count, duration

        finally:
            is_processing.clear()
            cap.release()

    def _analyze_temporal_context(
        self, 
        context_frames: List[np.ndarray], 
        center_frame_idx: int,
        frame_number: int,
        timestamp: float,
        video_shape: Tuple[int, int]
    ) -> Optional[Dict]:
        """Analyze temporal context window and return single context-aware prediction."""
        if not context_frames or center_frame_idx >= len(context_frames):
            return None
        
        center_frame = context_frames[center_frame_idx]
        faces = self.face_model(center_frame, conf=self.face_conf)
        
        for det in faces:
            if global_stop_event.is_set():
                return None
            
            for box in det.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                center_crop = center_frame[int(y1):int(y2), int(x1):int(x2)]
                if center_crop.size == 0 or center_crop.shape[0] < 10 or center_crop.shape[1] < 10:
                    continue
                
                context_predictions = []
                
                for frame_idx, frame in enumerate(context_frames):
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        continue
                    
                    try:
                        emo_res = self.emotion_model(crop, conf=self.emotion_conf)[0]
                        if emo_res is None or emo_res.boxes is None or len(emo_res.boxes) == 0:
                            continue
                        
                        best = max(emo_res.boxes, key=lambda b: b.conf[0])
                        cls_id = int(best.cls[0])
                        label = self.labels.get(cls_id, "unknown")
                        score = float(best.conf[0])
                        
                        distance_from_center = abs(frame_idx - center_frame_idx)
                        weight = 1.0 / (1.0 + 0.3 * distance_from_center)
                        
                        context_predictions.append({
                            'label': label,
                            'score': score,
                            'weight': weight
                        })
                        
                    except Exception as e:
                        lg.warning(f"Context frame emotion detection failed: {e}")
                        continue
                
                if not context_predictions:
                    continue
                
                label_scores = {}
                total_weight = 0.0
                
                for pred in context_predictions:
                    label = pred['label']
                    weighted_score = pred['score'] * pred['weight']
                    label_scores[label] = label_scores.get(label, 0.0) + weighted_score
                    total_weight += pred['weight']
                
                if total_weight > 0:
                    label_scores = {k: v / total_weight for k, v in label_scores.items()}
                
                best_label = max(label_scores, key=label_scores.get)
                best_score = label_scores[best_label]

                result = {
                    "frame": frame_number,
                    "time": timestamp,
                    "x": (x1 / video_shape[1]) * 100,
                    "y": (y1 / video_shape[0]) * 100,
                    "width": ((x2 - x1) / video_shape[1]) * 100,
                    "height": ((y2 - y1) / video_shape[0]) * 100,
                    "label": best_label,
                    "score": best_score,
                    "face_emotion_label": best_label,
                    "face_emotion_score": best_score,
                    "context_size": len(context_predictions)
                }
                
                if self.use_openface or self.use_libreface:
                    self._add_au_data_to_result(result, frame_number - 1)
                
                return result
        
        return None

    def analyze_window(self, frames: List, frame_indices: List[int], frame_count: int, duration: float) -> List[Dict]:
        """Analyze a window of frames and return per-frame predictions"""
        results = []
        
        for idx, frame in enumerate(frames):
            if global_stop_event.is_set():
                return results
            
            frame_idx = frame_indices[idx]
            faces = self.face_model(frame, conf=self.face_conf)
            
            for det in faces:
                if global_stop_event.is_set():
                    return results
                
                for box in det.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue
                    
                    try:
                        emo_res = self.emotion_model(crop, conf=self.emotion_conf)[0]
                        if emo_res is None or emo_res.boxes is None or len(emo_res.boxes) == 0:
                            continue
                        
                        best = max(emo_res.boxes, key=lambda b: b.conf[0])
                        cls_id = int(best.cls[0])
                        face_emotion_label = self.labels.get(cls_id, "unknown")
                        face_emotion_score = float(best.conf[0])
                    except Exception:
                        continue
                    
                    result = {
                        "frame": frame_idx + 1,
                        "time": (frame_idx + 1) * (duration / frame_count),
                        "x": (x1 / frame.shape[1]) * 100,
                        "y": (y1 / frame.shape[0]) * 100,
                        "width": ((x2 - x1) / frame.shape[1]) * 100,
                        "height": ((y2 - y1) / frame.shape[0]) * 100,
                        "label": face_emotion_label,
                        "score": face_emotion_score,
                        "face_emotion_label": face_emotion_label,
                        "face_emotion_score": face_emotion_score
                    }
                    
                    if self.use_openface or self.use_libreface:
                        self._add_au_data_to_result(result, frame_idx)
                    
                    results.append(result)
        
        return results

    def aggregate_window_predictions(self, window_results: List[Dict]) -> List[Dict]:
        """Aggregate predictions within a window"""
        if not window_results:
            return []
        
        frame_groups = {}
        for r in window_results:
            frame_num = r["frame"]
            if frame_num not in frame_groups:
                frame_groups[frame_num] = []
            frame_groups[frame_num].append(r)
        
        aggregated_results = []
        
        for frame_num in sorted(frame_groups.keys()):
            group = frame_groups[frame_num]
            if not group:
                continue
            
            # Vote on face emotion
            label_votes = {}
            for item in group:
                label = item.get('face_emotion_label', item['label'])
                label_votes[label] = label_votes.get(label, 0) + item['score']
            best_emotion = max(label_votes, key=label_votes.get)
            
            avg_x = sum(item['x'] for item in group) / len(group)
            avg_y = sum(item['y'] for item in group) / len(group)
            avg_width = sum(item['width'] for item in group) / len(group)
            avg_height = sum(item['height'] for item in group) / len(group)
            avg_score = sum(item['score'] for item in group) / len(group)
            
            middle_idx = len(group) // 2
            
            result = {
                "frame": frame_num,
                "time": group[middle_idx]['time'],
                "x": avg_x,
                "y": avg_y,
                "width": avg_width,
                "height": avg_height,
                "label": best_emotion,
                "score": avg_score,
                "face_emotion_label": group[middle_idx].get('face_emotion_label', best_emotion),
                "face_emotion_score": group[middle_idx].get('face_emotion_score', avg_score),
                "window_size": len(group)
            }
            
            # Aggregate OpenFace
            of_labels = [item.get('openface_label') for item in group if 'openface_label' in item]
            of_scores = [item.get('openface_score') for item in group if 'openface_score' in item]
            if of_labels:
                of_votes = {}
                for lbl, sc in zip(of_labels, of_scores):
                    of_votes[lbl] = of_votes.get(lbl, 0) + sc
                result["openface_label"] = max(of_votes, key=of_votes.get)
                result["openface_score"] = sum(of_scores) / len(of_scores)
                
                # Average OpenFace AUs
                of_aus_list = [item.get('openface_aus', {}) for item in group if 'openface_aus' in item]
                if of_aus_list:
                    all_keys = set()
                    for av in of_aus_list:
                        all_keys.update(av.keys())
                    result["openface_aus"] = {}
                    for key in all_keys:
                        values = [av.get(key, 0) for av in of_aus_list if key in av]
                        if values:
                            result["openface_aus"][key] = sum(values) / len(values)
            
            # Aggregate LibreFace
            lf_labels = [item.get('libreface_label') for item in group if 'libreface_label' in item]
            lf_scores = [item.get('libreface_score') for item in group if 'libreface_score' in item]
            if lf_labels:
                lf_votes = {}
                for lbl, sc in zip(lf_labels, lf_scores):
                    lf_votes[lbl] = lf_votes.get(lbl, 0) + sc
                result["libreface_label"] = max(lf_votes, key=lf_votes.get)
                result["libreface_score"] = sum(lf_scores) / len(lf_scores)
                
                # Average LibreFace AUs
                lf_aus_list = [item.get('libreface_aus', {}) for item in group if 'libreface_aus' in item]
                if lf_aus_list:
                    all_keys = set()
                    for av in lf_aus_list:
                        all_keys.update(av.keys())
                    result["libreface_aus"] = {}
                    for key in all_keys:
                        values = [av.get(key, 0) for av in lf_aus_list if key in av]
                        if values:
                            result["libreface_aus"][key] = sum(values) / len(values)
            
            # Gaze direction (most common)
            gaze_dirs = [item.get('gaze_direction') for item in group if 'gaze_direction' in item]
            if gaze_dirs:
                result["gaze_direction"] = max(set(gaze_dirs), key=gaze_dirs.count)
            
            aggregated_results.append(result)
        
        return aggregated_results
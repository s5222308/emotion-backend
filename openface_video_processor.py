import requests
import pandas as pd
import os
import logging as lg
from pathlib import Path
import numpy as np
from typing import Dict, Optional

class AUToEmotionMapper:
    """Maps Action Units to emotions based on FACS emotion signatures"""
    
    EMOTION_SIGNATURES = {
        "happy": {"AU06": 0.45, "AU12": 0.55},
        "content": {"AU06": 0.35, "AU12": 0.50, "AU25": 0.15},
        "sad": {"AU01": 0.30, "AU04": 0.35, "AU15": 0.35, "AU17": 0.15},
        "anger": {"AU04": 0.40, "AU05": 0.15, "AU07": 0.30, "AU10": 0.10, "AU23": 0.30},
        "fear": {"AU01": 0.25, "AU02": 0.25, "AU04": 0.10, "AU05": 0.30, "AU07": 0.15, "AU20": 0.30, "AU25": 0.15, "AU26": 0.10},
        "surprise": {"AU01": 0.30, "AU02": 0.30, "AU05": 0.30, "AU25": 0.15, "AU26": 0.35},
        "disgust": {"AU09": 0.50, "AU10": 0.30, "AU15": 0.15, "AU17": 0.10},
        "calm": {"AU06": 0.25, "AU12": 0.30, "AU25": 0.10},
        "neutral": {}
    }
    
    GAZE_MODIFIERS = {
        "down": {"sad": 1.35, "fear": 0.75, "happy": 0.70, "anger": 0.85, "content": 0.95},
        "left": {"fear": 1.20, "disgust": 1.15, "anger": 0.85, "sad": 1.10, "happy": 0.95},
        "right": {"fear": 1.20, "disgust": 1.15, "anger": 0.85, "sad": 1.10, "happy": 0.95},
        "up": {"surprise": 1.20, "sad": 0.85, "content": 1.08, "happy": 1.05, "fear": 1.05},
        "center": {"anger": 1.25, "happy": 1.10, "surprise": 1.10, "content": 1.05, "disgust": 0.90}
    }
    
    def calculate_neutral_score(self, aus: Dict) -> float:
        """
        High AU activation => very unlikely to be neutral.
        Make neutral only strong when the face is genuinely quiet.
        """
        values = list(aus.values())
        if not values:
            return 1.0

        mean_activation = np.mean(values)

        # If the face is strongly active, basically no neutral
        if mean_activation >= 0.7:
            return 0.05

        # For low-activation faces, neutral decays from ~0.4 -> 0.1
        neutral_raw = 0.4 * (1.0 - (mean_activation / 0.7))
        neutral_raw = max(0.0, min(neutral_raw, 0.4))
        return float(neutral_raw)

    def map_aus_to_emotion(self, aus: Dict, gaze_direction: Optional[str] = None) -> Dict:
        emotion_scores = {}
        
        for emotion, signature in self.EMOTION_SIGNATURES.items():
            if emotion == "neutral":
                emotion_scores[emotion] = self.calculate_neutral_score(aus)
            elif not signature:
                emotion_scores[emotion] = 0.0
            else:
                score = 0.0
                weight_sum = 0.0
                for au, weight in signature.items():
                    au_value = aus.get(au, 0.0)
                    score += au_value * weight
                    weight_sum += weight
                if weight_sum > 0:
                    score = score / weight_sum
                emotion_scores[emotion] = min(score / 5.0, 1.0)
        
        # Apply gaze modifiers
        if gaze_direction and gaze_direction in self.GAZE_MODIFIERS:
            for emotion, modifier in self.GAZE_MODIFIERS[gaze_direction].items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] = min(emotion_scores[emotion] * modifier, 1.0)

        # Slightly down-weight neutral so it doesn't drown everything
        if "neutral" in emotion_scores:
            emotion_scores["neutral"] *= 0.7

        # Normalize
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v / total for k, v in emotion_scores.items()}
        
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        return {
            "label": top_emotion,
            "score": float(emotion_scores[top_emotion]),
            "all_scores": {k: float(v) for k, v in emotion_scores.items()}
        }


class OpenFaceVideoProcessor:
    def __init__(self, au_service_url="http://openface-service:9000"):
        self.cache = {}
        self.au_service_url = au_service_url
        self.au_mapper = AUToEmotionMapper()

    def process_video(self, video_path: str) -> pd.DataFrame:
        if video_path in self.cache:
            lg.info(f"Using cached OpenFace results for {video_path}")
            return self.cache[video_path]

        try:
            lg.info(f"Processing video with OpenFace (dynamic calibration): {video_path}")

            response = requests.post(
                f"{self.au_service_url}/process_video",
                json={"video_path": video_path},
                timeout=300
            )

            if response.status_code != 200:
                lg.error(f"AU service returned status {response.status_code}: {response.text}")
                return None

            result = response.json()
            
            if result.get('status') != 'success':
                lg.error(f"AU service error: {result.get('error', 'Unknown error')}")
                return None

            csv_path = result['csv_path']
            df = pd.read_csv(csv_path)
            
            # Clean up CSV after reading
            try:
                os.remove(csv_path)
                print(f"[OpenFace] Cleaned up: {csv_path}", flush=True)
            except Exception as e:
                lg.warning(f"[OpenFace] Failed to clean up {csv_path}: {e}")
            
            self.cache[video_path] = df

            au25 = (df['AU25_r'] > 0).sum()
            au26 = (df['AU26_r'] > 0).sum()
            lg.info(f"✓ OpenFace processed {len(df)} frames: AU25 in {au25} frames, AU26 in {au26} frames")

            if au25 == 0 and au26 == 0:
                lg.warning("No mouth AUs detected - OpenFace may have issues")

            return df

        except requests.Timeout:
            lg.error("OpenFace processing timeout (>300s)")
            return None
        except Exception as e:
            lg.error(f"OpenFace processing error: {e}")
            return None

    def get_frame_aus(self, df: pd.DataFrame, frame_idx: int) -> dict:
        """Extract AU values for specific frame (0-indexed) and map to emotion"""
        if df is None or frame_idx >= len(df):
            return None

        row = df.iloc[frame_idx]

        # Extract AU intensities
        aus = {}
        for col in df.columns:
            if col.endswith('_r') and col.startswith('AU'):
                au_name = col.replace('_r', '')
                aus[au_name] = float(row[col])

        # Determine gaze direction
        gaze_dir = self._gaze_dir(row)
        
        # Map AUs to emotion
        emotion_prediction = self.au_mapper.map_aus_to_emotion(aus, gaze_dir)

        return {
            'au': aus,
            'label': emotion_prediction['label'],
            'score': emotion_prediction['score'],
            'all_scores': emotion_prediction.get('all_scores', {}),
            'gaze': {
                'direction': gaze_dir,
                'left_eye': {
                    'x': float(row.get('gaze_0_x', 0)),
                    'y': float(row.get('gaze_0_y', 0)),
                    'z': float(row.get('gaze_0_z', 0))
                },
                'right_eye': {
                    'x': float(row.get('gaze_1_x', 0)),
                    'y': float(row.get('gaze_1_y', 0)),
                    'z': float(row.get('gaze_1_z', 0))
                }
            },
            'head_pose': {
                'pitch': float(row.get('pose_Rx', 0)),
                'yaw': float(row.get('pose_Ry', 0)),
                'roll': float(row.get('pose_Rz', 0))
            },
            'confidence': float(row.get('confidence', 0))
        }

    def _gaze_dir(self, row):
        """Determine gaze direction from gaze vectors"""
        x = (row.get('gaze_0_x', 0) + row.get('gaze_1_x', 0)) / 2
        y = (row.get('gaze_0_y', 0) + row.get('gaze_1_y', 0)) / 2

        if abs(x) > 0.15:
            return "right" if x > 0 else "left"
        if abs(y) > 0.15:
            return "up" if y < 0 else "down"
        return "center"
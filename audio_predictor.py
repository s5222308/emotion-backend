import subprocess
import tempfile
import os
import numpy as np
import sys
from Functions.Helpers.set_models import set_audio_emotion_model, get_audio_emotion_model
import librosa
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from Functions.Shared_objects.shared_objects import global_stop_event, is_processing
from Functions.Helpers.set_parameters import set_param, get_param
from typing import List, Dict, Optional

class AudioEmotionRecognizer:
    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AudioEmotionRecognizer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.segment_duration = get_param("segment_duration")
        
        # New parameters for sliding window mode
        try:
            self.use_sliding_window = get_param("audio_sliding_window")
        except:
            self.use_sliding_window = False
            
        try:
            self.window_size = get_param("audio_window_size")
        except:
            self.window_size = 1.0
            
        try:
            self.window_overlap = get_param("audio_window_overlap")
        except:
            self.window_overlap = 0.5

        # Load model and feature extractor once
        model_name = get_audio_emotion_model()
        model_path = f"/app/models/audio_models/{model_name}"
        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.id2label = self.model.config.id2label

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AudioEmotionRecognizer] Device for audio is: {self.device}", file=sys.stderr, flush=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Audio label mapping (from README)
        self.AUDIO_TO_VIDEO_LABEL_MAP = {
            "fearful": "fear",
            "surprised": "surprise",
            "angry": "anger",
            "disgusted": "disgust",
            "sad": "sad",
            "happy": "happy",
            "neutral": "neutral",
            "content": "content",
            "ANG": "anger",
            "CAL": "calm",
            "DIS": "disgust",
            "FEA": "fear",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
            "SUR": "surprise"
        }
    
    def normalize_label(self, label: str) -> str:
        """Normalize audio labels to match video labels"""
        return self.AUDIO_TO_VIDEO_LABEL_MAP.get(label, label)
    
    def normalize_labels(self, audio_results: List[Dict]) -> List[Dict]:
        """Normalize all labels in audio results"""
        for item in audio_results:
            item['emotion_label'] = self.normalize_label(item['emotion_label'])
        return audio_results
    
    def set_model(self, name):
        if is_processing.is_set():
            print("[AudioEmotionRecognizer] Model reload attempted while processing – skipping.", file=sys.stderr, flush=True)
            return

        # Update JSON config
        set_audio_emotion_model(name)

        # Load the new model
        model_path = f"/app/models/audio_models/{name}"
        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.id2label = self.model.config.id2label

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AudioEmotionRecognizer] Device for audio is: {self.device}", file=sys.stderr, flush=True)
        self.model.to(self.device)
        self.model.eval()

    def change_segment_duration(self, new_duration: float):
        """Change the segment duration used for audio processing."""
        if is_processing.is_set():
            print("[AudioEmotionRecognizer] Attempted to change segment_duration while processing – skipping.", file=sys.stderr, flush=True)
            return
        if new_duration <= 0:
            print("[AudioEmotionRecognizer] segment_duration must be positive – skipping.", file=sys.stderr, flush=True)
            return
        if new_duration >= 10:
            print("[AudioEmotionRecognizer] segment_duration must be less than 10 – skipping.", file=sys.stderr, flush=True)
            return
        self.segment_duration = new_duration
        print(f"[AudioEmotionRecognizer] segment_duration updated to {self.segment_duration} seconds", file=sys.stderr, flush=True)
        set_param("segment_duration", new_duration)

    def set_sliding_window(self, enabled: bool):
        """Enable/disable sliding window mode for temporal alignment"""
        if is_processing.is_set():
            print("[AudioEmotionRecognizer] Cannot change sliding window mode while processing", file=sys.stderr, flush=True)
            return
        self.use_sliding_window = enabled
        set_param("audio_sliding_window", enabled)
        print(f"[AudioEmotionRecognizer] Sliding window mode: {enabled}", file=sys.stderr, flush=True)

    def set_window_params(self, window_size: float = None, overlap: float = None):
        """Set sliding window parameters"""
        if is_processing.is_set():
            print("[AudioEmotionRecognizer] Cannot change window params while processing", file=sys.stderr, flush=True)
            return
        
        if window_size is not None:
            if window_size <= 0 or window_size > 5:
                print("[AudioEmotionRecognizer] window_size must be between 0 and 5 seconds", file=sys.stderr, flush=True)
                return
            self.window_size = window_size
            set_param("audio_window_size", window_size)
            print(f"[AudioEmotionRecognizer] Window size: {window_size}s", file=sys.stderr, flush=True)
        
        if overlap is not None:
            if overlap < 0 or overlap >= 1:
                print("[AudioEmotionRecognizer] overlap must be between 0 and 1", file=sys.stderr, flush=True)
                return
            self.window_overlap = overlap
            set_param("audio_window_overlap", overlap)
            print(f"[AudioEmotionRecognizer] Window overlap: {overlap}", file=sys.stderr, flush=True)

    def predict_emotion(self, chunk_wave: np.ndarray, sampling_rate: int) -> tuple:
        """Predict emotion from audio chunk"""
        if global_stop_event.is_set():
            return None, None
        
        inputs = self.feature_extractor(chunk_wave, sampling_rate=sampling_rate, return_tensors="pt")
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)
        
        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits
        
        scores = torch.nn.functional.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(scores).cpu().item())
        label = self.id2label[pred_id]
        score = float(scores[pred_id].cpu().item())
        
        # Normalize label
        normalized_label = self.normalize_label(label)
        
        return normalized_label, score

    def ProcessAudioWindowed(self, input_path: str) -> Optional[List[Dict]]:
        """
        Process audio with sliding windows for better temporal alignment.
        Each window is centered around specific timestamps.
        """
        # Extract audio from video
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-vn",
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[AudioEmotionRecognizer] Error: ffmpeg not found in PATH.", file=sys.stderr, flush=True)
            return None
        except subprocess.CalledProcessError:
            print(f"[AudioEmotionRecognizer] Error: Failed to extract audio from {input_path}.", file=sys.stderr, flush=True)
            return None

        if global_stop_event.is_set():
            os.remove(temp_wav_path)
            return None

        # Load the extracted audio
        target_sr = self.feature_extractor.sampling_rate
        audio, sr = librosa.load(temp_wav_path, sr=target_sr)
        os.remove(temp_wav_path)

        if audio is None or len(audio) == 0 or global_stop_event.is_set():
            print("[AudioEmotionRecognizer] Error: No audio data found.", file=sys.stderr, flush=True)
            return None

        # Calculate window parameters
        window_samples = int(self.window_size * target_sr)
        hop_samples = int((self.window_size - self.window_size * self.window_overlap) * target_sr)
        
        results = []
        
        # Process with sliding windows
        for start_idx in range(0, len(audio) - window_samples + 1, hop_samples):
            if global_stop_event.is_set():
                return None
            
            is_processing.set()
            
            end_idx = start_idx + window_samples
            chunk = audio[start_idx:end_idx]
            
            # Get emotion prediction
            label, conf = self.predict_emotion(chunk, target_sr)
            
            if label is None:
                continue
            
            # Calculate time boundaries
            start_time = start_idx / target_sr
            end_time = end_idx / target_sr
            
            results.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "emotion_label": label,
                "confidence_score": round(conf, 3)
            })
            
            print(f"[AudioEmotionRecognizer] Window {start_time:.2f}-{end_time:.2f}s: {label} ({conf:.3f})", 
                  file=sys.stderr, flush=True)
        
        # Handle last segment if needed
        remaining_samples = len(audio) - (len(results) * hop_samples)
        if remaining_samples > window_samples // 2:  # Process if more than half window
            start_idx = len(audio) - window_samples
            if start_idx >= 0:
                chunk = audio[start_idx:]
                if len(chunk) < window_samples:
                    chunk = np.pad(chunk, (0, window_samples - len(chunk)))
                
                label, conf = self.predict_emotion(chunk, target_sr)
                if label:
                    start_time = start_idx / target_sr
                    end_time = len(audio) / target_sr
                    
                    results.append({
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "emotion_label": label,
                        "confidence_score": round(conf, 3)
                    })
        
        is_processing.clear()
        return results

    def ProcessAudio(self, input_path):
        """Main entry point - uses sliding window or fixed segments based on configuration"""
        if self.use_sliding_window:
            print(f"[AudioEmotionRecognizer] Using sliding window mode: {self.window_size}s windows, {self.window_overlap} overlap", 
                  file=sys.stderr, flush=True)
            return self.ProcessAudioWindowed(input_path)
        else:
            print(f"[AudioEmotionRecognizer] Using fixed segment mode: {self.segment_duration}s segments", 
                  file=sys.stderr, flush=True)
            return self.ProcessAudioFixed(input_path)

    def ProcessAudioFixed(self, input_path):
        """Original fixed segment processing (backward compatibility)"""
        # 1. Extract audio from the video using ffmpeg
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-vn",
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[AudioEmotionRecognizer] Error: ffmpeg not found in PATH.", file=sys.stderr, flush=True)
            return None
        except subprocess.CalledProcessError:
            print(f"[AudioEmotionRecognizer] Error: Failed to extract audio from {input_path}.", file=sys.stderr, flush=True)
            return None

        if global_stop_event.is_set():
            os.remove(temp_wav_path)
            return None

        # 2. Load the extracted audio
        target_sr = self.feature_extractor.sampling_rate
        audio, sr = librosa.load(temp_wav_path, sr=target_sr)
        os.remove(temp_wav_path)

        if audio is None or len(audio) == 0 or global_stop_event.is_set():
            print("[AudioEmotionRecognizer] Error: No audio data found.", file=sys.stderr, flush=True)
            return None

        # 3. Segment parameters
        segment_duration = self.segment_duration
        segment_samples = int(segment_duration * target_sr)
        total_samples = len(audio)
        num_full_segments = total_samples // segment_samples
        remainder = total_samples % segment_samples

        current_label = None
        current_confs = []
        segment_start_time = 0.0
        results = []

        # Helper: run model
        try:
            # 4. Full segments
            for i in range(num_full_segments):
                is_processing.set()
                print("[AudioEmotionRecognizer] Processing audio segment", file=sys.stderr, flush=True)
                if global_stop_event.is_set():
                    return None
                start_idx = i * segment_samples
                end_idx = start_idx + segment_samples
                chunk = audio[start_idx:end_idx]
                label, conf = self.predict_emotion(chunk, target_sr)
                if label is None:
                    return None
                timestamp = i * segment_duration

                if i == 0:
                    current_label = label
                    current_confs = [conf]
                    segment_start_time = 0.0
                else:
                    if label == current_label:
                        current_confs.append(conf)
                    else:
                        avg_conf = sum(current_confs) / len(current_confs)
                        results.append((segment_start_time, timestamp, current_label, avg_conf))
                        current_label = label
                        current_confs = [conf]
                        segment_start_time = timestamp

            # 5. Remainder
            if global_stop_event.is_set():
                return None
            if remainder > 0:
                last_start_time = num_full_segments * segment_duration
                last_chunk = audio[num_full_segments * segment_samples:]
                if len(last_chunk) < segment_samples:
                    last_chunk = np.pad(last_chunk, (0, segment_samples - len(last_chunk)))
                last_label, last_conf = self.predict_emotion(last_chunk, target_sr)
                if current_label is None:
                    current_label = last_label
                    current_confs = [last_conf]
                    segment_start_time = 0.0
                if last_label == current_label:
                    current_confs.append(last_conf)
                    avg_conf = sum(current_confs) / len(current_confs)
                    results.append((segment_start_time, float(num_full_segments * segment_duration + (remainder / target_sr)), current_label, avg_conf))
                else:
                    avg_conf = sum(current_confs) / len(current_confs)
                    results.append((segment_start_time, last_start_time, current_label, avg_conf))
                    results.append((last_start_time, float(num_full_segments * segment_duration + (remainder / target_sr)), last_label, last_conf))
            else:
                if current_label is not None:
                    end_time = num_full_segments * segment_duration
                    avg_conf = sum(current_confs) / len(current_confs)
                    results.append((segment_start_time, float(end_time), current_label, avg_conf))

            # 6. Format results
            formatted_results = []
            for (start, end, label, conf) in results:
                formatted_results.append({
                    "start_time": round(start, 2),
                    "end_time": round(end, 2),
                    "emotion_label": label,
                    "confidence_score": round(conf, 3)
                })

            return formatted_results
        finally:
            is_processing.clear()
import json
import queue
import sys
from threading import Thread
import time
from flask import jsonify, request
import logging as lg
from concurrent.futures import ThreadPoolExecutor, as_completed

from Functions.Helpers.get_labelstudio_sdk import get_labelstudio_sdk
from group_frames import frame_by_frame_regions
from multimodal_fushion import EmotionFusionEngine
from Functions.Shared_objects.shared_objects import global_stop_event, prediction_queue, task_progress_lock, task_progress, incoming_tasks, get_executor, is_processing, is_aborting
from Functions.Shared_objects.model_instances import predictor, recognizer, engine


lg.basicConfig(level=lg.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_label_config(path="/label_config.xml"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def project_listener_worker():
    while True:
        ls = get_labelstudio_sdk()
        label_config = load_label_config("label_config.xml")
        projects = None
        try:
            try:
                projects = ls.projects.list()
            except Exception as e:
                print(e)
            if projects:
                for p in projects:
                    needs_update = False
                    if not p.label_config or "<Choices" not in p.label_config:
                        needs_update = True
                    if not getattr(p, "ml_backend", None):
                        needs_update = True
                    if needs_update:
                        print(f"Configuring project: {p.title}", sys.stderr, flush=True)
                        ls.projects.update(
                            id=p.id,
                            label_config=label_config,
                            evaluate_predictions_automatically=False,
                            reveal_preannotations_interactively=False,
                            show_collab_predictions=False
                        )
                        existing_backends = ls.ml.list(project=p.id)
                        if not existing_backends:
                            ml_backend = ls.ml.create(
                                project=p.id,
                                title="Emotion Backend",
                                url="http://emotion-backend:9090",
                            )

        except Exception as e:
            lg.exception(f"Error checking projects: {e}")

        time.sleep(10)


def queue_worker():
    while True:
        ls = get_labelstudio_sdk()
        if global_stop_event.is_set():
            lg.info("Queue worker detected global abort, clearing pending tasks")
            while not prediction_queue.empty():
                try:
                    prediction_queue.get_nowait()
                    prediction_queue.task_done()
                except queue.Empty:
                    break
            with task_progress_lock:
                task_progress["submitted"] = 0
                task_progress["completed"] = 0
            time.sleep(0.5)
            continue

        try:
            task_id, regions = prediction_queue.get(timeout=1)
        except queue.Empty:
            time.sleep(0.5)
            continue

        try:
            ls.predictions.create(task=task_id, result=regions, model_version="multimodal_fusion_v1")
            with task_progress_lock:
                task_progress["completed"] += 1
            lg.info(f"[{task_id}] Posted prediction from queue")
        except Exception as e:
            lg.exception(f"[{task_id}] Failed to post prediction: {e}")
        finally:
            prediction_queue.task_done()

def task_worker():
    while True:
        task = incoming_tasks.get()

        while is_aborting:
            time.sleep(0.1)

        if global_stop_event.is_set():
            lg.info("Global abort detected, skipping task")
            incoming_tasks.task_done()
            continue

        try:
            stage_two(task)
        except Exception as e:
            lg.exception(f"Failed to process task {task.get('id')}: {e}")
        finally:
            incoming_tasks.task_done()

Thread(target=task_worker, daemon=True).start()
Thread(target=project_listener_worker, daemon=True).start()
Thread(target=queue_worker, daemon=True).start()

def predict():
    try:
        payload = request.get_json(force=True)
        tasks = payload.get("tasks", [])
        if not tasks:
            return jsonify({"error": "No tasks provided"}), 400

        for task in tasks:
            incoming_tasks.put(task)

        return jsonify({
            "results": [],
            "model_version": "queued",
            "task": "processing"
        })

    except Exception as e:
        lg.error("Error in /predict", e)
        return jsonify({"error": str(e)}), 500


def stage_two(task):
    task_id = task.get("id")
    video_path = task.get("data", {}).get("video")
    
    if video_path.startswith("/data/upload/"):
        video_path = video_path.replace("/data/upload/", "/label-studio/data/media/upload/")

    with task_progress_lock:
        task_progress["submitted"] += 1
        lg.info(f"Task queued. Progress: {task_progress['submitted']} submitted")
    
    executor = get_executor()
    executor.submit(process_and_inject_prediction, task_id, video_path)


def process_and_inject_prediction(task_id, video_path):
    try:
        is_processing.set()
        
        with ThreadPoolExecutor(max_workers=2) as inner_pool:
            futures = {
                inner_pool.submit(run_video, video_path): "video",
                inner_pool.submit(run_audio, video_path): "audio",
            }

            raw_video_results, frames_count, duration = None, None, None
            raw_audio_results = None

            for future in as_completed(futures):
                if global_stop_event.is_set():
                    lg.info(f"[{task_id}] Global abort detected, stopping task")
                name = futures[future]
                try:
                    result = future.result()
                    if name == "video":
                        if result == (None, None, None):
                            lg.info(f"Aborting task [{task_id}]", sys.stderr, flush=True)
                            raw_video_results, frames_count, duration = None, None, None
                        raw_video_results, frames_count, duration = result
                    elif name == "audio":
                        raw_audio_results = result
                        raw_audio_results = normalize_labels(raw_audio_results)
                        print(f"[{task_id}] Audio analysis complete, {raw_audio_results[:10]}", sys.stderr, flush=True)
                except Exception as e:
                    lg.exception(f"[{task_id}] Error in {name} analysis: {e}")

        if raw_video_results is None or global_stop_event.is_set():
            if global_stop_event.is_set():
                lg.info(f"Task aborted before fusion")
            lg.error(f"[{task_id}] Skipping task - video analysis failed")
            return

        # Fuse results
        engine.reset()
        fused = engine.fuse(raw_video_results, raw_audio_results or [])
        lg.info(f"[{task_id}] Fusion complete with {len(fused)} frames")

        # Merge fused results back into raw_video_results
        for i, v in enumerate(raw_video_results):
            if i < len(fused):
                v['fused_label'] = fused[i]['label']
                v['fused_score'] = fused[i]['score']
                # Remove redundant 'label' field (it's same as yolo_label)
                if 'label' in v:
                    del v['label']
                if 'score' in v:
                    del v['score']

        # Debug output
        print(f"DEBUG FUSED: {[(f.get('frame'), f.get('fused_label'), f.get('fused_score'), f.get('yolo_score'), f.get('libreface_score')) for f in raw_video_results[:5]]}", file=sys.stderr, flush=True)

        # Convert to Label Studio format
        regions = frame_by_frame_regions(
            results=fused, frames_count=frames_count, duration=duration
        )

        # Audio debug table
        regions.append({
            "from_name": "audio_table",
            "to_name": "video",
            "type": "textarea",
            "value": {
                "text": json.dumps(
                    {"audio_results": raw_audio_results},
                    indent=1,
                )
            },
        })

        # Video debug table - single clean output
        regions.append({
            "from_name": "video_table",
            "to_name": "video",
            "type": "textarea",
            "value": {
                "text": json.dumps(
                    {"video_results": raw_video_results},
                    indent=1,
                )
            },
        })

        prediction_queue.put((task_id, regions))
        lg.info(f"[{task_id}] Enqueued prediction")
        
    except Exception as e:
        lg.exception(f"[{task_id}] Fatal error injecting prediction: {e}")
    finally:
        is_processing.clear()


def run_video(video_path):
    lg.info("Running video analysis")
    return predictor.run(video_path)


def run_audio(video_path):
    lg.info("Running audio analysis")
    return recognizer.ProcessAudio(video_path)


def normalize_labels(audio_results):
    AUDIO_TO_VIDEO_LABEL_MAP = {
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
    for item in audio_results:
        item['emotion_label'] = AUDIO_TO_VIDEO_LABEL_MAP.get(item['emotion_label'], item['emotion_label'])
    return audio_results
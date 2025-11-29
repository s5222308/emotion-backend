import os
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
import libreface
from libreface.Facial_Expression_Recognition.solver_inference_image import solver_inference_image, Facial_Expression_Dataset
from libreface.Facial_Expression_Recognition.inference import ConfigObject, set_seed, facial_expr_idx_to_class
from torch.utils.data import DataLoader

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LibreFace Service")


# ---------- Models ----------

class VideoRequest(BaseModel):
    video_path: str


# ---------- Helpers ----------

def _slim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce LibreFace's huge output down to:
    - AU flags (au_1, au_4, ...)
    - AU intensities (au_1_intensity, ...)
    - frame/time indices if present
    - facial_expression + facial_expression_prob
    """
    cols = df.columns

    keep_cols = [
        c for c in cols
        if c.startswith("au_")
        or c in [
            "frame",
            "frame_idx",
            "frame_index",
            "time",
            "timestamp",
            "facial_expression",
            "facial_expression_prob",
        ]
    ]

    if not keep_cols:
        # fallback: just return whatever we got
        return df

    return df[keep_cols]


def get_facial_expression_with_probs(aligned_frames_path, device="cpu", 
                                      batch_size=256, num_workers=2,
                                      weights_download_dir="./weights_libreface"):
    """
    Custom FER inference that returns both labels and probabilities.
    This replaces libreface.get_facial_expression_video which discards probabilities.
    """
    opts = ConfigObject({
        'seed': 0,
        'train_csv': 'training_filtered.csv',
        'test_csv': 'validation_filtered.csv',
        'data_root': '',
        'ckpt_path': f'{weights_download_dir}/Facial_Expression_Recognition/weights/resnet.pt',
        'weights_download_id': '1PeoPj8rga4vU2nuh_PciyX3HqaXp6LP7',
        'data': 'AffectNet',
        'num_workers': num_workers,
        'image_size': 224,
        'num_labels': 8,
        'dropout': 0.1,
        'hidden_dim': 128,
        'sigma': 10.0,
        'student_model_name': 'resnet',
        'student_model_choices': [
            'resnet_heatmap', 'resnet', 'swin', 'mae', 'emotionnet_mae', 'gh_feat'
        ],
        'alpha': 1.0,
        'T': 1.0,
        'fm_distillation': True,
        'grad': True,
        'interval': 500,
        'threshold': 0.0,
        'loss': 'unweighted',
        'num_epochs': 50,
        'batch_size': batch_size,
        'learning_rate': '3e-5',
        'weight_decay': '1e-4',
        'clip': 1.0,
        'when': 10,
        'patience': 10,
        'device': 'cpu'
    })
    
    set_seed(opts.seed)
    opts.device = device
    
    solver = solver_inference_image(opts).to(device)
    
    # Create dataset and dataloader
    dataset = Facial_Expression_Dataset(
        aligned_frames_path, 
        (solver.config.image_size, solver.config.image_size)
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=solver.config.batch_size,
        num_workers=solver.config.num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False
    )
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        solver.student_model.eval()
        for input_image in loader:
            input_image = input_image.to(solver.device)
            logits, _ = solver.student_model(input_image)
            
            # Get probabilities via softmax
            probs = F.softmax(logits, dim=1)
            
            # Get predicted labels and their corresponding probabilities
            pred_labels = torch.argmax(probs, dim=1)
            pred_probs = probs.gather(1, pred_labels.unsqueeze(1)).squeeze(1)
            
            all_labels.extend(pred_labels.cpu().tolist())
            all_probs.extend(pred_probs.cpu().tolist())
    
    # Convert indices to class names
    labels_str = [facial_expr_idx_to_class(idx) for idx in all_labels]
    
    return pd.DataFrame({
        "facial_expression": labels_str,
        "facial_expression_prob": all_probs
    })


def get_facial_attributes_with_probs(video_path, device="cpu"):
    """
    Modified version of libreface.get_facial_attributes that includes FER probabilities.
    """
    import tempfile
    from libreface import (
        get_aligned_video_frames,
        get_frames_from_video_ffmpeg,
        get_au_intensities_and_detect_aus_video
    )
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Extract frames
        frames_df = get_frames_from_video_ffmpeg(video_path, temp_dir=temp_dir)
        
        # Step 2: Align faces
        aligned_frames_path_list, headpose_list, landmarks_3d_list = get_aligned_video_frames(
            frames_df,
            temp_dir=temp_dir
        )
        
        # Step 3: Get AU detection and intensities
        detected_aus, au_intensities = get_au_intensities_and_detect_aus_video(
            aligned_frames_path_list,
            device=device
        )
        
        # Step 4: Get facial expressions WITH probabilities (our custom function)
        facial_expression_df = get_facial_expression_with_probs(
            aligned_frames_path_list,
            device=device
        )
        
        # Combine all results
        frames_df = frames_df.join(detected_aus)
        frames_df = frames_df.join(au_intensities)
        frames_df = frames_df.join(facial_expression_df)
        
        return frames_df
        
    finally:
        # Cleanup temp directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# ---------- Routes ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Debug / test endpoint:
    - Accepts an uploaded *image or video* file (multipart/form-data)
    - Runs LibreFace
    - Returns slimmed JSON (no 400+ landmark columns)
    """
    suffix = os.path.splitext(file.filename)[1] or ".mp4"

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        logger.info(f"Running LibreFace /analyze on temp file: {tmp_path}")
        result = get_facial_attributes_with_probs(tmp_path, device="cpu")

        if isinstance(result, pd.DataFrame):
            result = _slim_columns(result)
            data = result.to_dict(orient="records")
        else:
            data = str(result)

        return JSONResponse({"results": data})
    except Exception as e:
        logger.exception(f"LibreFace /analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/process_video")
async def process_video(req: VideoRequest):
    """
    Production-style endpoint:
    - Takes a JSON body: { "video_path": "/path/on/shared/volume/video.mp4" }
    - Runs LibreFace on the entire video
    - Slims columns to AU + expression + frame/time
    - Writes CSV to /label-studio/data/libreface_cache/<video>_libreface.csv
    - Returns that csv_path
    """
    video_path = req.video_path

    if not video_path or not os.path.exists(video_path):
        msg = f"Invalid video path: {video_path}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    output_dir = "/label-studio/data/libreface_cache"
    os.makedirs(output_dir, exist_ok=True)

    video_name = Path(video_path).stem
    csv_path = os.path.join(output_dir, f"{video_name}_libreface.csv")

    try:
        logger.info(f"Processing video with LibreFace: {video_path}")

        # Use our custom function that includes probabilities
        df = get_facial_attributes_with_probs(video_path, device="cpu")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("LibreFace did not return a pandas DataFrame")

        df_slim = _slim_columns(df)
        df_slim.to_csv(csv_path, index=False)

        if not os.path.exists(csv_path):
            raise RuntimeError("CSV not generated by LibreFace")

        logger.info(f"LibreFace CSV written: {csv_path}")
        return {"status": "success", "csv_path": csv_path}
    except Exception as e:
        logger.exception(f"LibreFace /process_video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
from flask import Flask, request, jsonify
import subprocess
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'status': 'error', 'error': f'Invalid video path: {video_path}'}), 400
    
    try:
        output_dir = "/label-studio/data/openface_cache"
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing video: {video_path}")
        
        cmd = [
            "/opt/OpenFace/build/bin/FeatureExtraction",
            "-f", video_path,
            "-out_dir", output_dir,
            "-aus", "-gaze", "-pose"
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode != 0:
            return jsonify({'status': 'error', 'error': f'OpenFace failed: {result.stderr.decode()}'}), 500
        
        video_name = Path(video_path).stem
        csv_path = f"{output_dir}/{video_name}.csv"
        
        if not os.path.exists(csv_path):
            return jsonify({'status': 'error', 'error': 'CSV not generated'}), 500
        
        logger.info(f"✓ Video processed: {csv_path}")
        return jsonify({'status': 'success', 'csv_path': csv_path}), 200
        
    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'error': 'Processing timeout'}), 500
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting OpenFace video service on port 9000...")
    app.run(host='0.0.0.0', port=9000)
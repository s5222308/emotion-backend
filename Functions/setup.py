from flask import jsonify
def setup():
    return jsonify({
        "description": "Emotion recognition backend",
        "type": "video",
        "tags": ["yolo", "emotion", "faces"],
        "model_version": "1.0"
    })

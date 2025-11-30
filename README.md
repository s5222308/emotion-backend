## Installation

### Prerequisites
- Docker and Docker Compose (v2.20 or later)
- Git
- Node.js (v20 or later) and npm
- At least 20GB of free disk space

### Installation Steps

1. **Clone the repository**
```bash
   git clone https://github.com/s5222308/emotion-backend.git
   cd emotion-backend
```

2. **Install frontend dependencies**
```bash
   cd 3821ICT-annotation-tool
   npm install
   cd ..
```

3. **Add model files**
   
   Contact your system administrator to obtain the `models/` folder and place it in the project root:
```
   emotion-backend/
   ├── models/
   │   ├── face_models/
   │   │   ├── yolov11s-face.pt
   │   │   └── yolo11s-emotion.pt
   │   └── audio_models/
   │       └── wave2vec-english-speech-recognition-by-r-f/
```

4. **Start the system**
```bash
   ./start.sh
```

5. **Access the application**
   - Annotation Tool: http://localhost:3000
   - Label Studio: http://localhost:8026

6. **Configure Label Studio API Key**
   - Open Label Studio at http://localhost:8026
   - Create account / login
   - Go to Account & Settings → Access Token
   - Copy the token
   - Paste it in the Dashboard at http://localhost:3000
   - Click Save

## Model Requirements

### Face & Emotion Models
- Must be YOLO-based (.pt format)
- Emotion models must output 7 emotions: happy, content, sad, disgust, anger, neutral, surprise

### Audio Models
- Must be classification-based (e.g., from HuggingFace)
- Output labels are automatically normalized to match the system

### Audio Label Normalization
The system accepts various audio model outputs and normalizes them:
```python
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
```

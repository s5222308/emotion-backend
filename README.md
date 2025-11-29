### Clone this repo
``git clone <repo-url>``


### Pull in the submodule
`` git submodule update --init --recursive  ``


### Change directory into the submodule
`` cd 3821ICT-annotation-tool``

### Run this command(it installs node modules and vite etc)
`` npm install ``

### Add models folder, structure is 
Models/face_models, emotion_models, audio_models

``copy and paste models folder``

face_models and emotion_models have to be Yolo based. Emotion models must output the 7 emotions(happy, content, sad, disgust, anger, neutral and suprised). The audio_models have to be similar as well in terms of output, but they need to be classficiation based. Search on huggingface for audio_classification etc.


### To start
``./start.sh`` , then connect to localhost:3000


#### Mapping system: The function below shows only the acceptable outputs for audio models, meaning a model that would output SUR, is acceptable, and will be normalized to fit the systme

```
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
        item['emotion_label'] = AUDIO_TO_VIDEO_LABEL_MAP.get(item['emotion_label'],item['emotion_label'])
    return audio_results
```

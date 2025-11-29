LABEL_STUDIO_URL = 'http://label-studio:8080'

import os
from label_studio_sdk.client import LabelStudio
from Functions.Helpers.get_labelstudio_api_key import get_labelstudio_api_key

_ls_instance = None
_api_key = get_labelstudio_api_key()

def set_labelstudio_sdk_key(new_key: str):
    """Update the API key and recreate the SDK instance."""
    key_path = "/config/labelstudio_api_key.txt"    
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    with open(key_path, "w") as f:
        f.write(new_key.strip())
        

    global _api_key, _ls_instance
    _api_key = new_key.strip()
    _ls_instance = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=_api_key)
    return _ls_instance

def get_labelstudio_sdk():
    """Return the current SDK instance, creating if necessary."""
    global _ls_instance, _api_key
    if _ls_instance is None:
        _ls_instance = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=_api_key)
    return _ls_instance

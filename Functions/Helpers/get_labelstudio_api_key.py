def get_labelstudio_api_key():
    try:
        with open("/config/labelstudio_api_key.txt", "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Could not read API key: {e}")
        return None

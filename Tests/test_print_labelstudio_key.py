def test_print_labelstudio_key():
    with open("/config/labelstudio_api_key.txt") as f:
        key = f.read().strip()
    assert key  # Optional: ensure it's not empty
from Functions.Helpers.get_labelstudio_sdk import get_labelstudio_sdk, set_labelstudio_sdk_key
from flask import jsonify, request
import os
import logging as lg

def set_labelstudio_key():
     # Direct path, no env var
    data = request.get_json(force=True)
    new_key = data.get("api_key")

    if not new_key:
        return jsonify({"error": "Missing 'api_key'"}), 400

    try:
        # Ensure directory exists (just in case)

        ls = set_labelstudio_sdk_key(new_key.strip())  # re-reads the new token
        return jsonify({"message": "API key updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to write key: {e}"}), 500
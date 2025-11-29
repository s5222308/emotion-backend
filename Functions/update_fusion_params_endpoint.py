import sys
from flask import request, jsonify
import logging as lg
from Functions.Shared_objects.model_instances import engine


def update_fusion_params_endpoint():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "No JSON payload provided"}), 400

        valid_keys = engine.DEFAULT_CONFIG.keys()
        update_kwargs = {k: payload[k] for k in payload if k in valid_keys}

        if not update_kwargs:
            return jsonify({"error": "No valid parameters to update"}), 400

        engine.update_params(**update_kwargs)

        current_params = engine.get_params()
        print(f"Updated fusion params: {current_params}", file=sys.stderr, flush=True)
        return jsonify({"status": "success", "params": current_params})

    except Exception as e:
        print(f"Failed to update fusion params: {e}", file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500


def get_fusion_params_endpoint():
    try:
        current_params = engine.get_params()
        response = {
            "status": "success",
            "params": current_params,
            "limits": engine.PARAM_LIMITS
        }
        print(f"Fusion params response: {response}", file=sys.stderr, flush=True)  # log to stderr
        return jsonify(response)
    except Exception as e:
        print(f"Failed to get fusion params: {e}", file=sys.stderr, flush=True)
        return jsonify({"error": str(e)}), 500
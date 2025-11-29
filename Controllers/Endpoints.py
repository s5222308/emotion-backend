from Functions import abort_all, get_models, get_progress, health, predict, set_labelstudio_key, set_models_endpoint, setup, get_labelstudio_key, get_fusion_params_endpoint, update_fusion_params_endpoint

def register_routes(app):

    # Get Requests
    app.add_url_rule("/health", view_func=health, methods=["GET"])
    
    # Returns a dummy key(Fake key for display/visual)
    app.add_url_rule("/get-labelstudio-key", view_func=get_labelstudio_key, methods=["GET"])
    app.add_url_rule("/get_progress", view_func=get_progress, methods=["GET"])
    app.add_url_rule("/get_models", view_func=get_models, methods=["GET"])
    app.add_url_rule("/get-fusion_params", view_func=get_fusion_params_endpoint, methods=["GET"])
    
    # POST REQUESTS
    app.add_url_rule("/set-labelstudio-key", view_func=set_labelstudio_key, methods=["POST"])
    app.add_url_rule("/predict", view_func=predict, methods=["POST"])
    app.add_url_rule("/abort_all", view_func=abort_all, methods=["POST"])
    app.add_url_rule("/set_models", view_func=set_models_endpoint,methods=["POST"])
    app.add_url_rule("/update-fusion_params",view_func=update_fusion_params_endpoint, methods=["POST"])

    # Mix of GET AND POST
    app.add_url_rule("/setup", view_func=setup, methods=["GET", "POST"])






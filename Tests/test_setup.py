def test_setup_endpoint(client):
    response = client.get("/setup")
    assert response.status_code == 200

    data = response.get_json()
    assert data["description"] == "Emotion recognition backend"
    assert data["type"] == "video"
    assert "yolo" in data["tags"]
    assert data["model_version"] == "1.0"

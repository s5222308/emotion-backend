def test_get_models_success(client):
    response = client.get("/get_models")
    assert response.status_code == 200

    data = response.get_json()
    assert "current" in data
    assert "available" in data

    current = data["current"]
    assert "face" in current
    assert "face_emotion" in current
    assert "audio_emotion" in current
    assert "parameters" in current
    assert isinstance(current["parameters"], dict)

    available = data["available"]
    assert "face" in available
    assert isinstance(available["face"], list)


from unittest.mock import patch

@patch("Functions.Helpers.set_models.get_face_model", side_effect=Exception("fail"))
@patch("Functions.Helpers.set_models.get_face_emotion_model", return_value="emo_v1")
@patch("Functions.Helpers.set_models.get_audio_emotion_model", return_value="aud_v1")
def test_get_models_fallback(mock_audio, mock_emotion, mock_face, client):
    response = client.get("/get_models")
    data = response.get_json()
    assert "current" in data
    assert data["current"]["face"] is not None
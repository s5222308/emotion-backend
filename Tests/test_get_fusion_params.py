def test_get_fusion_params_success(client):
    response = client.get("/get-fusion_params")
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "success"
    assert "params" in data
    assert "limits" in data
    assert isinstance(data["params"], dict)
    assert isinstance(data["limits"], dict)


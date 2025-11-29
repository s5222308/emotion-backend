import string


def test_get_labelstudio_key_obfuscation(client):
    response = client.get("/get-labelstudio-key")
    assert response.status_code == 200

    obfuscated_key = response.get_json()
    assert isinstance(obfuscated_key, str)
    assert len(obfuscated_key) > 5  # Should be longer than original due to mutations

    # Optional: check that it contains alphanumeric noise
    noise = any(c in string.ascii_letters + string.digits for c in obfuscated_key)
    assert noise
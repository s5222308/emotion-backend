
import pytest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from server import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

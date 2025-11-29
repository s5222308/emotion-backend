import time
from server import app
from Functions.get_progress import get_progress
from Functions.Shared_objects import shared_objects as so
import Functions.get_progress as gp  # for accessing last_activity_time

def test_get_progress_logic():
    with app.app_context():
        with so.task_progress_lock:
            so.task_progress["submitted"] = 3
            so.task_progress["completed"] = 3

        gp.last_activity_time = time.time()  # simulate recent activity

        response = get_progress()
        data = response.get_json()
        assert data["submitted"] == 3
        assert data["completed"] == 3
        assert "Completed" in data["message"]

def test_progress_active_tasks(client):
    with client.application.app_context():
        with so.task_progress_lock:
            so.task_progress["submitted"] = 5
            so.task_progress["completed"] = 2

        gp.last_activity_time = time.time()  # simulate recent activity

        response = client.get("/get_progress")
        data = response.get_json()
        assert data["submitted"] == 5
        assert data["completed"] == 2
        assert "Processing" in data["message"]
        assert "2 of 5" in data["message"]

from flask import jsonify
import time
from Functions.Shared_objects.shared_objects import task_progress, task_progress_lock

QUEUE_EMPTY_RESET_DELAY = 30
last_activity_time = time.time()

def get_progress():
    global last_activity_time
    with task_progress_lock:
        submitted = task_progress.get("submitted", 0)
        completed = task_progress.get("completed", 0)

        # Reset after tasks completed
        if submitted > 0 and completed >= submitted:
            message = f"Completed {completed} tasks"
            # Only reset after QUEUE_EMPTY_RESET_DELAY seconds
            if time.time() - last_activity_time > QUEUE_EMPTY_RESET_DELAY:
                task_progress["submitted"] = 0
                task_progress["completed"] = 0
                message = "No tasks in queue"
        elif submitted == 0:
            # Queue empty logic
            if time.time() - last_activity_time > QUEUE_EMPTY_RESET_DELAY:
                message = "No tasks in queue"
            else:
                message = "Waiting for new tasks..."
        else:
            message = f"Processing {completed} of {submitted} tasks"
            last_activity_time = time.time()  # reset activity timestamp on active tasks

    return jsonify({
        "submitted": submitted,
        "completed": completed,
        "message": message
    })

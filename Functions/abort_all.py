import time
import logging as lg
from flask import jsonify
from Functions.Shared_objects import shared_objects as so

def abort_all():
    lg.info("Global abort requested")

    so.is_aborting = True        # block new submissions
    so.global_stop_event.set()   # signal running tasks to stop

    # Clear incoming queue
    while not so.incoming_tasks.empty():
        try:
            so.incoming_tasks.get_nowait()
            so.incoming_tasks.task_done()
        except Exception:
            break

    # Clear prediction queue
    while not so.prediction_queue.empty():
        try:
            so.prediction_queue.get_nowait()
            so.prediction_queue.task_done()
        except Exception:
            break

    # ✅ Reset executor using helper
    so.reset_executor(max_workers=1)

    # Reset progress counters
    with so.task_progress_lock:
        so.task_progress["submitted"] = 0
        so.task_progress["completed"] = 0

    # Wait for any in-flight processing to finish
    while so.is_processing.is_set():
        lg.info("Waiting for current task to stop...")
        time.sleep(0.5)

    # Allow new tasks
    so.global_stop_event.clear()
    so.is_aborting = False

    lg.info("Abort complete, system reset and ready for new tasks")
    return jsonify({"status": "aborted_all"})

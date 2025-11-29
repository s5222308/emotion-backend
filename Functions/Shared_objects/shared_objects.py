from threading import Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

global_stop_event = Event()
is_processing = Event()
is_aborting = False  # <-- new flag

incoming_tasks = Queue()
prediction_queue = Queue()
task_progress_lock = Lock()
task_progress = {"submitted": 0, "completed": 0}


_executor = ThreadPoolExecutor(max_workers=1)
_executor_lock = Lock()

def get_executor():
    """Return a live executor, recreating if it was shut down."""
    global _executor
    with _executor_lock:
        if getattr(_executor, "_shutdown", False):
            _executor = ThreadPoolExecutor(max_workers=1)
        return _executor

def reset_executor(max_workers=1):
    """Forcefully shut down and recreate the executor."""
    global _executor
    with _executor_lock:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = ThreadPoolExecutor(max_workers=max_workers)
    return _executor
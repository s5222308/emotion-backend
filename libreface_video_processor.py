import os
from typing import Dict

import logging as lg
import pandas as pd
import requests


class LibreFaceVideoProcessor:
    """
    Thin wrapper around the libreface-service container.

    - Sends video_path to /process_video
    - Reads the returned CSV (AUs + facial_expression)
    - Caches results per video_path
    """

    def __init__(self, libreface_service_url: str = "http://libreface-service:9001"):
        self.base_url = libreface_service_url.rstrip("/")
        self.cache: Dict[str, pd.DataFrame] = {}

    def process_video(self, video_path: str, timeout: int = 600) -> pd.DataFrame:
        if video_path in self.cache:
            lg.info(f"[LibreFace] Using cached results for {video_path}")
            return self.cache[video_path]

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"[LibreFace] Video does not exist: {video_path}")

        lg.info(f"[LibreFace] Requesting processing for video: {video_path}")

        resp = requests.post(
            f"{self.base_url}/process_video",
            json={"video_path": video_path},
            timeout=timeout,
        )

        try:
            resp.raise_for_status()
        except Exception as e:
            lg.error(
                f"[LibreFace] HTTP error {resp.status_code} from service: {resp.text}"
            )
            raise

        data = resp.json()
        csv_path = data.get("csv_path")

        if not csv_path:
            raise RuntimeError(f"[LibreFace] No csv_path in response: {data}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[LibreFace] CSV path does not exist on backend: {csv_path}"
            )

        lg.info(f"[LibreFace] Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Clean up CSV after reading
        try:
            os.remove(csv_path)
            print(f"[LibreFace] Cleaned up: {csv_path}", flush=True)
        except Exception as e:
            lg.warning(f"[LibreFace] Failed to clean up {csv_path}: {e}")

        self.cache[video_path] = df
        return df

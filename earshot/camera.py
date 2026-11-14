"""The table camera: latest frame from the Logitech webcam, selected by name.

macOS numbers cameras differently in different libraries, and the number changes when a
phone or another camera appears. ffmpeg can open a camera by its name, so one ffmpeg process
streams JPEG frames into a reader thread and everyone asks this module for the newest one.
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

from earshot import config

_SOI, _EOI = b"\xff\xd8", b"\xff\xd9"


class Camera:
    """Keeps the most recent frame from the named camera."""

    def __init__(self, name: str = config.CAMERA_NAME, size: tuple[int, int] = config.CAMERA_SIZE, fps: int = 10):
        self.name, self.size, self.fps = name, size, fps
        self._jpeg: Optional[bytes] = None
        self._at = 0.0
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stop = threading.Event()

    def start(self) -> "Camera":
        """Start streaming; returns once the first frame has arrived."""
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-framerate", "30",
               "-video_size", f"{self.size[0]}x{self.size[1]}", "-i", f"{self.name}:none",
               "-r", str(self.fps), "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._read, daemon=True).start()
        deadline = time.time() + 8
        while self._jpeg is None and time.time() < deadline:
            time.sleep(0.05)
        if self._jpeg is None:
            self.stop()
            raise RuntimeError(f'No picture from the camera "{self.name}". Is it plugged in, and not open in another program?')
        return self

    def _read(self) -> None:
        buf = b""
        while not self._stop.is_set() and self._proc and self._proc.stdout:
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                a = buf.find(_SOI)
                b = buf.find(_EOI, a + 2) if a >= 0 else -1
                if a < 0 or b < 0:
                    break
                with self._lock:
                    self._jpeg, self._at = buf[a:b + 2], time.time()
                buf = buf[b + 2:]

    def jpeg(self) -> Optional[bytes]:
        """Newest frame as JPEG bytes (for the screen and for the vision models)."""
        with self._lock:
            return self._jpeg

    def frame(self) -> np.ndarray:
        """Newest frame as a BGR image."""
        data = self.jpeg()
        if data is None:
            raise RuntimeError("The camera has not produced a frame yet.")
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def fresh_frame(self, newer_than_s: float = 0.4) -> np.ndarray:
        """A frame captured after this call, so it shows the table as it is now."""
        t0 = time.time()
        while self._at < t0 and time.time() - t0 < newer_than_s + 2:
            time.sleep(0.03)
        return self.frame()

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            self._proc = None

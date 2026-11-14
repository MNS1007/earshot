"""The live screen: a small FastAPI app that shows earshot.state.state in a browser.

    from earshot.ui import server
    server.set_frame_provider(lambda: latest_jpeg_bytes)   # optional; a placeholder is shown otherwise
    server.run()                                           # background thread, returns the URL
    while True:
        act = server.next_voice_action(timeout=0.2)        # {"action", "args", "said", "source"} or None

Routes
    GET  /             the page                GET  /api/state   full state as JSON
    WS   /ws           full state on connect and on every change
    GET  /frame.jpg    latest camera frame (or a placeholder)
    POST /api/voice    {"action", "args", "said"} from the voice process
    POST /api/command  {"action", "args"} from the on-screen buttons
Both POSTs land on the same queue, read with next_voice_action().

This module never opens a camera, microphone or serial port.
"""
from __future__ import annotations

import asyncio
import queue
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import config
from ..state import state

__all__ = ["app", "run", "stop", "set_frame_provider", "next_voice_action", "put_action", "COMMANDS"]

STATIC = Path(__file__).resolve().parent / "static"

# What the on-screen buttons may send (the voice process may send any action name).
COMMANDS = ("start_teaching", "stop_teaching", "help_me", "your_turn", "pause", "resume", "load_task")

_BUTTON_WORDS = {
    "start_teaching": "Start teaching", "stop_teaching": "Stop teaching", "help_me": "Help me",
    "your_turn": "Your turn", "pause": "Pause", "resume": "Resume", "load_task": "Load task",
}

# ----------------------------------------------------------------------------- action queue

_actions: "queue.Queue[dict]" = queue.Queue()


def put_action(action: str, args: Optional[dict] = None, said: str = "", source: str = "code") -> dict:
    """Queue an action as if it had been spoken. Returns the queued item."""
    item = {"action": str(action), "args": dict(args or {}), "said": str(said or ""),
            "source": source, "t": time.time()}
    _actions.put(item)
    return item


def next_voice_action(timeout: Optional[float] = None) -> Optional[dict]:
    """Next action from the voice process or the buttons, oldest first.

    Returns {"action": str, "args": dict, "said": str, "source": "voice"|"button"|"code", "t": float},
    or None if nothing arrived within `timeout` seconds (timeout=0: do not wait; None: wait forever).
    """
    try:
        if timeout is not None and timeout <= 0:
            return _actions.get_nowait()
        return _actions.get(timeout=timeout)
    except queue.Empty:
        return None


# ----------------------------------------------------------------------------- camera frame

_frame_provider: Optional[Callable[[], Optional[bytes]]] = None
_placeholder: Optional[tuple[bytes, str]] = None


def set_frame_provider(provider: Optional[Callable[[], Optional[bytes]]]) -> None:
    """Register a function returning the latest camera frame as JPEG bytes (or None if there is
    none yet). It is called from a server worker thread about five times a second, so it should
    just hand back the newest already-encoded frame. Pass None to go back to the placeholder."""
    global _frame_provider
    _frame_provider = provider


def _make_placeholder() -> tuple[bytes, str]:
    """A neutral 'no camera' image drawn in code: JPEG via OpenCV, or an SVG if that is missing."""
    try:
        import cv2
        import numpy as np

        w, h = 960, 540
        img = np.full((h, w, 3), 232, np.uint8)
        cv2.rectangle(img, (12, 12), (w - 13, h - 13), (170, 170, 170), 2)
        for text, scale, thick, y in (("NO CAMERA", 2.0, 4, h // 2 - 5), ("waiting for the first frame", 0.9, 2, h // 2 + 50)):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(img, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (95, 95, 95), thick, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            return buf.tobytes(), "image/jpeg"
    except Exception:
        pass
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 540"><rect width="960" height="540" '
           'fill="#e8e8e8"/><text x="480" y="285" font-family="sans-serif" font-size="56" font-weight="bold" '
           'fill="#5f5f5f" text-anchor="middle">NO CAMERA</text></svg>')
    return svg.encode(), "image/svg+xml"


def _frame() -> tuple[bytes, str, bool]:
    global _placeholder
    provider = _frame_provider
    if provider is not None:
        try:
            data = provider()
            if data:
                return bytes(data), "image/jpeg", True
        except Exception:
            pass
    if _placeholder is None:
        _placeholder = _make_placeholder()
    return _placeholder[0], _placeholder[1], False


# ----------------------------------------------------------------------------- websocket fan-out

_loop: Optional[asyncio.AbstractEventLoop] = None
_dirty: "set[asyncio.Event]" = set()      # one per connected browser; touched only on the loop


def _wake_all() -> None:
    for ev in _dirty:
        ev.set()


def _on_state_change(_snapshot: dict) -> None:
    """Called by AppState on whichever thread changed it; hop onto the server's event loop."""
    loop = _loop
    if loop is not None and not loop.is_closed():
        try:
            loop.call_soon_threadsafe(_wake_all)
        except RuntimeError:
            pass


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    unsubscribe = state.subscribe(_on_state_change)
    try:
        yield
    finally:
        unsubscribe()
        _loop = None


app = FastAPI(title="Earshot", lifespan=_lifespan, docs_url=None, redoc_url=None, openapi_url=None)

_NO_STORE = {"Cache-Control": "no-store, max-age=0"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers=_NO_STORE)


@app.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(state.snapshot(), headers=_NO_STORE)


@app.get("/frame.jpg")
def frame_jpg() -> Response:
    data, media, live = _frame()
    return Response(data, media_type=media, headers={**_NO_STORE, "X-Earshot-Live": "1" if live else "0"})


async def _read_action(request: Request) -> tuple[Optional[dict], Optional[JSONResponse]]:
    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict) or not isinstance(body.get("action"), str) or not body["action"].strip():
        return None, JSONResponse({"ok": False, "error": 'expected JSON like {"action": "pause", "args": {}}'},
                                  status_code=400)
    args = body.get("args")
    return {"action": body["action"].strip(), "args": args if isinstance(args, dict) else {},
            "said": str(body.get("said") or "")}, None


@app.post("/api/voice")
async def api_voice(request: Request) -> JSONResponse:
    body, err = await _read_action(request)
    if err:
        return err
    item = put_action(body["action"], body["args"], body["said"], source="voice")
    if item["action"] not in ("transcript", "nothing"):   # the main loop shows the words; no event for an echo
        state.heard(f"“{item['said']}” → {item['action']}" if item["said"] else f"→ {item['action']}")
    return JSONResponse({"ok": True, "queued": item})


@app.post("/api/command")
async def api_command(request: Request) -> JSONResponse:
    body, err = await _read_action(request)
    if err:
        return err
    if body["action"] not in COMMANDS:
        return JSONResponse({"ok": False, "error": f"unknown command; use one of {list(COMMANDS)}"}, status_code=400)
    words = _BUTTON_WORDS[body["action"]]
    query = str(body["args"].get("query") or "").strip()
    if body["action"] == "load_task" and query:
        words += f": {query}"
    item = put_action(body["action"], body["args"], body["said"] or words, source="button")
    state.heard(f"[button] {words}")
    return JSONResponse({"ok": True, "queued": item})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    dirty = asyncio.Event()
    dirty.set()                               # send the full state straight away
    _dirty.add(dirty)
    try:
        while True:
            try:
                await asyncio.wait_for(dirty.wait(), timeout=10)   # also a keep-alive every 10 s
            except asyncio.TimeoutError:
                pass
            dirty.clear()                     # several changes in a burst collapse into one send
            await websocket.send_json(state.snapshot())
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    except Exception:
        pass
    finally:
        _dirty.discard(dirty)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ----------------------------------------------------------------------------- run in a thread

_server = None
_thread: Optional[threading.Thread] = None


def run(host: str = "127.0.0.1", port: Optional[int] = None, wait: float = 10.0) -> str:
    """Start the screen in a background (daemon) thread and return its URL once it is serving.

    Raises RuntimeError if the port is already in use or the server does not come up.
    Calling it again while it is running just returns the URL.
    """
    global _server, _thread
    import uvicorn

    port = int(port or config.UI_PORT)
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{port}/"
    if _thread is not None and _thread.is_alive():
        return url

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as e:
        raise RuntimeError(f"Port {port} is already in use ({e}). Is another Earshot screen running? "
                           f"Find it with: lsof -i :{port}") from e
    finally:
        probe.close()

    _server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False))
    _thread = threading.Thread(target=_server.run, name="earshot-ui", daemon=True)
    _thread.start()
    deadline = time.time() + wait
    while not _server.started:
        if not _thread.is_alive() or time.time() > deadline:
            raise RuntimeError(f"The screen server did not start on {url}")
        time.sleep(0.02)
    return url


def stop() -> None:
    """Ask the background server to shut down (optional; the thread is a daemon)."""
    global _server, _thread
    if _server is not None:
        _server.should_exit = True
    if _thread is not None:
        _thread.join(timeout=3)
    _server, _thread = None, None

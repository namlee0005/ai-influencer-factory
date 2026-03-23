"""
Async ComfyUI client: HTTP submission + WebSocket progress tracking.
GPU watchdog polls nvidia-smi every 15s; pauses queue at VRAM > 90%.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
import websockets
from websockets.exceptions import WebSocketException

logger = logging.getLogger(__name__)

COMFYUI_BASE = "http://localhost:8188"
COMFYUI_WS = "ws://localhost:8188"
VRAM_PAUSE_THRESHOLD = 0.90   # fraction
GPU_POLL_INTERVAL = 15.0      # seconds
WS_TIMEOUT = 120.0            # seconds per generation
HTTP_TIMEOUT = 30.0


class ComfyUIError(Exception):
    """Base error for ComfyUI client failures."""


class ComfyUITimeoutError(ComfyUIError):
    """Raised when a generation job exceeds WS_TIMEOUT."""


class ComfyUIVRAMError(ComfyUIError):
    """Raised when VRAM usage exceeds threshold and queue is paused."""


# ---------------------------------------------------------------------------
# GPU Watchdog
# ---------------------------------------------------------------------------

@dataclass
class GPUWatchdog:
    pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self.pause_event.set()  # not paused initially
        self._task = asyncio.create_task(self._poll(), name="gpu-watchdog")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _poll(self) -> None:
        while True:
            try:
                usage = await asyncio.to_thread(self._read_vram_fraction)
                if usage > VRAM_PAUSE_THRESHOLD:
                    if self.pause_event.is_set():
                        logger.warning(
                            "gpu_watchdog",
                            extra={"event": "pausing", "vram_fraction": round(usage, 3)},
                        )
                        self.pause_event.clear()  # signal pause
                else:
                    if not self.pause_event.is_set():
                        logger.info(
                            "gpu_watchdog",
                            extra={"event": "resuming", "vram_fraction": round(usage, 3)},
                        )
                        self.pause_event.set()  # clear pause
            except Exception as exc:
                logger.error("gpu_watchdog_error", extra={"error": str(exc)})
            await asyncio.sleep(GPU_POLL_INTERVAL)

    @staticmethod
    def _read_vram_fraction() -> float:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return 0.0  # no GPU or nvidia-smi absent — allow queue to run
        used, total = result.stdout.strip().split(",")
        return int(used.strip()) / int(total.strip())


# ---------------------------------------------------------------------------
# ComfyUI Client
# ---------------------------------------------------------------------------

class ComfyUIClient:
    def __init__(self, client_id: str | None = None) -> None:
        self.client_id = client_id or str(uuid.uuid4())
        self.watchdog = GPUWatchdog()
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ComfyUIClient":
        self._http = httpx.AsyncClient(base_url=COMFYUI_BASE, timeout=HTTP_TIMEOUT)
        self.watchdog.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.watchdog.stop()
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(self, workflow_json: dict) -> str:
        """
        Enqueue a workflow. Returns ComfyUI prompt_id.
        Raises ComfyUIVRAMError if watchdog has paused the queue.
        """
        if not self.watchdog.pause_event.is_set():
            raise ComfyUIVRAMError("GPU VRAM > 90% — queue paused. Retry later.")

        assert self._http is not None
        payload = {"prompt": workflow_json, "client_id": self.client_id}
        logger.info("comfyui_submit", extra={"client_id": self.client_id})
        resp = await self._http.post("/prompt", json=payload)
        resp.raise_for_status()
        data = resp.json()
        prompt_id: str = data["prompt_id"]
        logger.info("comfyui_queued", extra={"prompt_id": prompt_id})
        return prompt_id

    async def wait_for_completion(self, prompt_id: str) -> list[str]:
        """
        Wait for a prompt to complete via WebSocket.
        Returns list of output image filenames.
        Falls back to HTTP polling if WebSocket connection fails.
        """
        try:
            return await self._wait_ws(prompt_id)
        except WebSocketException as exc:
            logger.warning("ws_fallback", extra={"reason": str(exc)})
            return await self._wait_poll(prompt_id)

    # ------------------------------------------------------------------
    # Internal: WebSocket path
    # ------------------------------------------------------------------

    async def _wait_ws(self, prompt_id: str) -> list[str]:
        ws_url = f"{COMFYUI_WS}/ws?clientId={self.client_id}"
        async with websockets.connect(ws_url) as ws:
            try:
                async with asyncio.timeout(WS_TIMEOUT):
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "executing":
                            data = msg.get("data", {})
                            if data.get("node") is None and data.get("prompt_id") == prompt_id:
                                # execution finished
                                return await self._fetch_outputs(prompt_id)
                        elif msg.get("type") == "execution_error":
                            data = msg.get("data", {})
                            if data.get("prompt_id") == prompt_id:
                                raise ComfyUIError(f"Execution error: {data.get('exception_message')}")
            except TimeoutError:
                raise ComfyUITimeoutError(f"prompt_id={prompt_id} timed out after {WS_TIMEOUT}s")
        return []

    # ------------------------------------------------------------------
    # Internal: HTTP polling fallback
    # ------------------------------------------------------------------

    async def _wait_poll(self, prompt_id: str) -> list[str]:
        assert self._http is not None
        deadline = asyncio.get_event_loop().time() + WS_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            resp = await self._http.get("/history", params={"prompt_id": prompt_id})
            resp.raise_for_status()
            history = resp.json()
            if prompt_id in history:
                return await self._fetch_outputs(prompt_id)
            await asyncio.sleep(2.0)
        raise ComfyUITimeoutError(f"prompt_id={prompt_id} polling timed out")

    async def _fetch_outputs(self, prompt_id: str) -> list[str]:
        assert self._http is not None
        resp = await self._http.get("/history", params={"prompt_id": prompt_id})
        resp.raise_for_status()
        history = resp.json()
        outputs = history.get(prompt_id, {}).get("outputs", {})
        filenames: list[str] = []
        for node_output in outputs.values():
            for img in node_output.get("images", []):
                filenames.append(img["filename"])
        return filenames

    async def download_image(self, filename: str) -> bytes:
        assert self._http is not None
        resp = await self._http.get("/view", params={"filename": filename})
        resp.raise_for_status()
        return resp.content
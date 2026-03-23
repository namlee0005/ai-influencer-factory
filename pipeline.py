"""
Core pipeline orchestrator for AI Influencer Factory.

Flow per job:
  WorkflowRequest
    → CharacterRegistry (lookup profile)
    → VisualArtist (generate six-layer prompt)
    → WorkflowBuilder (patch Jinja2 template)
    → ComfyUIClient (submit + wait)
    → MediaPostProcessor (resize, watermark, face-similarity gate)
    → GenerationJob (status = complete | failed)

Single asyncio.Queue consumer enforces one GPU job at a time.
Each job exposes a per-job event queue for SSE streaming.
"""
from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

from character_registry import CharacterRegistry
from comfyui_client import ComfyUIClient, ComfyUIVRAMError
from models import CharacterProfile, GenerationJob, WorkflowRequest
from workflow_builder import build_workflow

logger = logging.getLogger(__name__)

MAX_FACE_RETRIES = 2
FACE_SIMILARITY_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Stubs — replace when visual_artist.py and media_processor.py land
# ---------------------------------------------------------------------------

async def _generate_prompt(profile: CharacterProfile, request: WorkflowRequest) -> str:
    """Stub: returns a bare scene description until VisualArtist is wired in."""
    try:
        from visual_artist import VisualArtist  # type: ignore[import]
        artist = VisualArtist()
        return await artist.generate(profile=profile, request=request)
    except ImportError:
        logger.warning("visual_artist not available — using bare scene description")
        return request.scene_description


async def _post_process(
    image_bytes: bytes,
    profile: CharacterProfile,
    request: WorkflowRequest,
    job_id: str,
    attempt: int,
) -> tuple[list[str], float | None]:
    """
    Stub: saves raw bytes to /output and returns (paths, similarity_score).
    Real implementation delegates to MediaPostProcessor.
    """
    try:
        from media_processor import MediaPostProcessor  # type: ignore[import]
        processor = MediaPostProcessor()
        return await processor.process(
            image_bytes=image_bytes,
            profile=profile,
            request=request,
            job_id=job_id,
        )
    except ImportError:
        import os
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/{job_id}_attempt{attempt}.png"
        with open(path, "wb") as f:
            f.write(image_bytes)
        logger.warning("media_processor not available — raw image saved to %s", path)
        return [path], None  # None = skip face gate


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState:
    """Shared mutable state: job registry + per-job SSE event queues."""

    def __init__(self) -> None:
        self._jobs: dict[str, GenerationJob] = {}
        self._events: dict[str, asyncio.Queue[str]] = {}

    def register(self, job: GenerationJob) -> asyncio.Queue[str]:
        self._jobs[job.job_id] = job
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        self._events[job.job_id] = q
        return q

    def get_job(self, job_id: str) -> GenerationJob | None:
        return self._jobs.get(job_id)

    def update(self, job: GenerationJob) -> None:
        self._jobs[job.job_id] = job

    def emit(self, job_id: str, event: str) -> None:
        q = self._events.get(job_id)
        if q:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # SSE consumer too slow; drop event

    def close_stream(self, job_id: str) -> None:
        self.emit(job_id, "__done__")

    async def stream(self, job_id: str) -> AsyncIterator[str]:
        q = self._events.get(job_id)
        if q is None:
            return
        while True:
            event = await q.get()
            if event == "__done__":
                break
            yield event


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, registry: CharacterRegistry | None = None) -> None:
        self._queue: asyncio.Queue[tuple[WorkflowRequest, GenerationJob]] = asyncio.Queue()
        self._registry = registry or CharacterRegistry()
        self._state = PipelineState()
        self._worker_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background worker. Call once at app startup."""
        self._worker_task = asyncio.create_task(self._worker(), name="pipeline-worker")
        logger.info("pipeline_started")

    async def stop(self) -> None:
        """Drain and shut down the worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("pipeline_stopped")

    # ------------------------------------------------------------------
    # Public submit
    # ------------------------------------------------------------------

    async def submit(self, request: WorkflowRequest) -> GenerationJob:
        """
        Enqueue a generation request.
        Returns a GenerationJob immediately; track progress via stream().
        """
        job = GenerationJob(
            job_id=str(uuid.uuid4()),
            prompt_id="",        # set after ComfyUI enqueue
            status="queued",
            created_at=datetime.now(timezone.utc),
        )
        self._state.register(job)
        await self._queue.put((request, job))
        logger.info("job_queued", extra={"job_id": job.job_id, "scenario": request.scenario})
        return job

    def stream(self, job_id: str) -> AsyncIterator[str]:
        """Async iterator of SSE event strings for a given job."""
        return self._state.stream(job_id)

    def get_job(self, job_id: str) -> GenerationJob | None:
        return self._state.get_job(job_id)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _worker(self) -> None:
        async with ComfyUIClient() as client:
            while True:
                request, job = await self._queue.get()
                try:
                    await self._run_job(request, job, client)
                except Exception as exc:
                    logger.error(
                        "job_failed_unhandled",
                        extra={"job_id": job.job_id, "error": str(exc)},
                        exc_info=True,
                    )
                    job = job.model_copy(update={"status": "failed"})
                    self._state.update(job)
                    self._state.emit(job.job_id, f"error:{exc}")
                    self._state.close_stream(job.job_id)
                finally:
                    self._queue.task_done()

    async def _run_job(
        self,
        request: WorkflowRequest,
        job: GenerationJob,
        client: ComfyUIClient,
    ) -> None:
        job_id = job.job_id
        self._state.emit(job_id, "status:running")

        # 1. Resolve character profile
        profile = self._registry.get(request.character_id)
        if profile is None:
            raise ValueError(f"Character '{request.character_id}' not found in registry")

        # 2. Generate prompt
        self._state.emit(job_id, "status:generating_prompt")
        prompt = await _generate_prompt(profile, request)
        logger.info("prompt_generated", extra={"job_id": job_id, "chars": len(prompt)})

        # 3. Face consistency retry loop
        seed = random.randint(0, 2**32 - 1)
        last_paths: list[str] = []
        last_score: float | None = None

        for attempt in range(MAX_FACE_RETRIES + 1):
            if attempt > 0:
                seed = random.randint(0, 2**32 - 1)  # fresh seed on retry
                self._state.emit(job_id, f"status:retry_{attempt}")
                logger.info("face_retry", extra={"job_id": job_id, "attempt": attempt})

            # 4. Build workflow
            workflow = build_workflow(request, profile, prompt, seed)

            # 5. Check VRAM + submit
            try:
                self._state.emit(job_id, "status:submitting")
                prompt_id = await client.submit(workflow)
            except ComfyUIVRAMError as exc:
                raise  # propagate — let caller surface 503

            # Update job with ComfyUI prompt_id
            job = job.model_copy(update={"prompt_id": prompt_id, "status": "running"})
            self._state.update(job)

            # 6. Wait for ComfyUI
            self._state.emit(job_id, f"status:generating_seed_{seed}")
            filenames = await client.wait_for_completion(prompt_id)
            if not filenames:
                raise RuntimeError(f"ComfyUI returned no output files for prompt_id={prompt_id}")

            # 7. Download first image
            image_bytes = await client.download_image(filenames[0])

            # 8. Post-process + face gate
            self._state.emit(job_id, "status:post_processing")
            paths, score = await _post_process(image_bytes, profile, request, job_id, attempt)
            last_paths = paths
            last_score = score

            if score is None or score >= FACE_SIMILARITY_THRESHOLD:
                break  # passed or no face gate active

            logger.warning(
                "face_similarity_below_threshold",
                extra={"job_id": job_id, "score": score, "threshold": FACE_SIMILARITY_THRESHOLD},
            )
            if attempt == MAX_FACE_RETRIES:
                logger.error(
                    "face_retry_exhausted",
                    extra={"job_id": job_id, "final_score": score},
                )
                # Persist best-effort result rather than dropping the job
                break

        # 9. Finalise
        job = job.model_copy(update={
            "status": "complete",
            "output_paths": last_paths,
            "face_similarity_score": last_score,
        })
        self._state.update(job)
        self._state.emit(job_id, f"status:complete")
        self._state.emit(job_id, f"output:{','.join(last_paths)}")
        self._state.close_stream(job_id)
        logger.info(
            "job_complete",
            extra={
                "job_id": job_id,
                "paths": last_paths,
                "face_score": last_score,
            },
        )
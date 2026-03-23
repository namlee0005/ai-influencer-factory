"""
FastAPI application: characters, jobs, and SSE progress streaming.

Endpoints:
  POST   /characters              — create/upsert CharacterProfile
  GET    /characters/{id}         — retrieve profile
  DELETE /characters/{id}         — remove profile
  POST   /jobs                    — submit WorkflowRequest → GenerationJob
  GET    /jobs/{id}               — poll job status
  GET    /jobs/{id}/stream        — SSE progress stream
  GET    /assets/{filename}       — serve output images
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from character_registry import CharacterRegistry
from models import CharacterProfile, GenerationJob, WorkflowRequest
from pipeline import Pipeline

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.path.realpath(os.environ.get("OUTPUT_ROOT", str(Path(__file__).parent / "output")))

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

registry = CharacterRegistry()
pipeline = Pipeline(registry=registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.start()
    logger.info("app_startup")
    yield
    await pipeline.stop()
    logger.info("app_shutdown")


app = FastAPI(title="AI Influencer Factory", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------

@app.post("/characters", response_model=CharacterProfile, status_code=201)
async def create_character(profile: CharacterProfile):
    logger.info("create_character", extra={"id": profile.id})
    try:
        return registry.save(profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/characters/{character_id}", response_model=CharacterProfile)
async def get_character(character_id: str):
    profile = registry.get(character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")
    return profile


@app.delete("/characters/{character_id}", status_code=204)
async def delete_character(character_id: str):
    deleted = registry.delete(character_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Character '{character_id}' not found")


@app.get("/characters", response_model=list[CharacterProfile])
async def list_characters():
    return registry.list_all()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=GenerationJob, status_code=202)
async def submit_job(request: WorkflowRequest):
    logger.info(
        "submit_job",
        extra={"character_id": request.character_id, "scenario": request.scenario},
    )
    # Validate character exists before queuing
    if registry.get(request.character_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{request.character_id}' not found",
        )
    job = await pipeline.submit(request)
    return job


@app.get("/jobs/{job_id}", response_model=GenerationJob)
async def get_job(job_id: str):
    job = pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    """Server-Sent Events endpoint for real-time job progress."""
    job = pipeline.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def event_generator():
        async for event in pipeline.stream(job_id):
            if await request.is_disconnected():
                break
            yield f"data: {event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

@app.get("/assets/{filename}")
async def serve_asset(filename: str):
    """Serve generated images from OUTPUT_ROOT with path traversal protection."""
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    resolved = os.path.realpath(os.path.join(OUTPUT_ROOT, safe_name))
    allowed = OUTPUT_ROOT
    if not resolved.startswith(allowed + os.sep) and resolved != allowed:
        raise HTTPException(status_code=400, detail="Path traversal detected")

    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="Asset not found")

    return FileResponse(resolved, media_type="image/png")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}
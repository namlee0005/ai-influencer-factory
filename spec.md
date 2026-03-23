# AI Influencer Factory — Project Spec

## Executive Summary

Build a **modular monolith** image generation pipeline on a single GPU node that produces consistent, high-quality images for an AI Influencer profile on TikTok/Facebook.

Use **IP-Adapter** for character consistency in V1 — defer LoRA training until engagement is validated. Maintain exactly **5 hand-crafted ComfyUI workflow templates**; no LLM-generated node graphs. Ship **image-only V1**; video (AnimateDiff/SVD) is gated behind real audience signal.

The project's highest risk is content strategy, not infrastructure. The minimum viable pipeline ships in 6 weeks. Validate engagement before scaling.

---

## Recommended Tech Stack

| Component | Choice | Reasoning |
|---|---|---|
| Language / Runtime | Python 3.12 | ComfyUI ecosystem; native async |
| API Framework | FastAPI | Pydantic v2 integration; SSE support |
| Job Queue | `asyncio.Queue` (in-process) | Single GPU = one consumer; no distribution benefit |
| Persistence | SQLite + JSON files | No concurrent writes; zero ops overhead |
| Job Progress | SSE from pipeline coroutine | One-way, proxy-safe, no broker needed |
| ComfyUI Client | `httpx.AsyncClient` + WebSocket | Async HTTP + fallback polling |
| Character Consistency | IP-Adapter (V1) / LoRA optional | Eliminates training dependency for V1 |
| Face Validation | `insightface` on CPU | Avoids VRAM conflict with SD pipeline |
| Post-processing | PIL + FFmpeg subprocess | Standard, well-documented |
| Workflow Templates | Jinja2 JSON (version-locked) | Parameterize prompts only; never node topology |

**Explicitly deferred:** Celery, Redis, PostgreSQL — all held until a second GPU node exists.

---

## Architecture Overview

```
User / Orchestrator
       │ WorkflowRequest (Pydantic)
       ▼
  VisualArtist Agent (Claude)
  → produces text prompts only — never node IDs or workflow JSON
       │ six-layer prompt: subject, style, lighting, camera, quality, negative
       ▼
  WorkflowBuilder
  → patches 1 of 5 locked Jinja2 templates (travel, food, fashion, gym, studio)
  → swaps: {prompt}, {seed}, {lora_weight}, {ip_adapter_ref}
       │ ComfyUI JSON payload
       ▼
  ComfyUI Server (localhost:8188)
  → GPU watchdog polls nvidia-smi every 15s
  → pauses queue when VRAM > 90%; returns 503 on new submissions
       │ raw PNG
       ▼
  MediaPostProcessor (PIL, FFmpeg)
  → insightface similarity check on CPU (threshold >= 0.75, max 2 retries)
  → resize + watermark per platform spec (TikTok 9:16, Facebook 1:1)
       │
       ▼
  CharacterRegistry (SQLite)  │  Content Store (local /output)
```

### Module Breakdown

| Module | Responsibility |
|---|---|
| `models.py` | Pydantic v2 contracts — `CharacterProfile`, `WorkflowRequest`, `GenerationJob` |
| `character_registry.py` | SQLite-backed identity store; path traversal guard on all writes |
| `comfyui_client.py` | Async HTTP + WebSocket client; typed timeout errors; GPU watchdog |
| `workflow_builder.py` | Jinja2 template selection and patching per scenario/aspect-ratio/quality |
| `visual_artist.py` | Claude prompt engineer; cached system prompt; text-only output |
| `media_processor.py` | PIL resize, FFmpeg encode, insightface CPU similarity gate |
| `pipeline.py` | End-to-end orchestration with face-consistency retry loop (max 2) |

### Key Data Contracts

```python
class CharacterProfile(BaseModel):
    id: str
    name: str
    lora_path: str | None = None
    ip_adapter_reference_image: str | None = None
    lora_weight: float = Field(ge=0.0, le=2.0, default=0.8)
    trigger_words: list[str]
    base_model: str
    negative_prompt: str

    @model_validator(mode="after")
    def must_have_one_consistency_method(self) -> "CharacterProfile":
        if not self.lora_path and not self.ip_adapter_reference_image:
            raise ValueError("Provide lora_path or ip_adapter_reference_image")
        return self

class WorkflowRequest(BaseModel):
    character_id: str
    scene_description: str
    scenario: Literal["travel", "food", "fashion", "lifestyle", "gym", "studio"]
    aspect_ratio: Literal["9:16", "1:1", "16:9"]
    output_type: Literal["image"]  # video locked to V2
    quality_preset: Literal["draft", "standard", "high"]

class GenerationJob(BaseModel):
    job_id: str
    prompt_id: str  # ComfyUI queue ID
    status: Literal["queued", "running", "complete", "failed"]
    output_paths: list[str] = []
    face_similarity_score: float | None = None
    created_at: datetime
```
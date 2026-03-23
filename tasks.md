# AI Influencer Factory — Implementation Plan

## Phase 1 — Core Pipeline (Weeks 1–2)

**Goal:** Generate one consistent image end-to-end through the full pipeline.

- [ ] **1.1** Define `models.py` — Pydantic v2 contracts: `CharacterProfile`, `WorkflowRequest`, `GenerationJob`
- [ ] **1.2** Build `character_registry.py` — SQLite-backed CRUD with path traversal guard
- [ ] **1.3** Build `comfyui_client.py` — async HTTP + WebSocket client + GPU watchdog (nvidia-smi, pause >90% VRAM)
- [ ] **1.4** Build `workflow_builder.py` — 3 Jinja2 templates (travel, food, fashion); variables only, node topology locked
- [ ] **1.5** Wire `pipeline.py` — asyncio.Queue orchestration skeleton

**Milestone:** Single `WorkflowRequest` → ComfyUI → raw PNG saved to `/output`

---

## Phase 2 — Quality Gate & Automation (Weeks 3–4)

**Goal:** Full pipeline with consistency validation and prompt automation.

- [ ] **2.1** Build `visual_artist.py` — Claude prompt engineer; six-layer text prompts only; cached system prompt
- [ ] **2.2** Build `media_processor.py` — insightface CPU similarity gate (threshold ≥0.75, max 2 retries); PIL resize; watermark; FFmpeg
- [ ] **2.3** Add SSE progress endpoint `GET /jobs/{id}/status` — stream from pipeline coroutine, no broker
- [ ] **2.4** Wire FastAPI app: `POST /characters`, `POST /jobs`, `GET /assets/{character_id}`

**Milestone:** Full pipeline with IP-Adapter consistency gate; rejected images surface with diagnosis

---

## Phase 3 — Content Loop Validation (Weeks 5–6)

**Goal:** Real audience signal before any V2 investment.

- [ ] **3.1** Add 2 remaining templates: `gym_9x16.json.j2`, `studio_1x1.json.j2`
- [ ] **3.2** Generate first content batch: 10 images across 3+ scenarios
- [ ] **3.3** Publish to TikTok and Facebook; record baseline engagement metrics
- [ ] **3.4** Document top-performing scenarios and prompt patterns

**Milestone:** Audience engagement data acquired — V2 decision gate reached

---

## V2 Gate (Conditional on Phase 3 engagement signal)

- [ ] LoRA training pipeline (50–200 curated images, training runs, quality gate)
- [ ] AnimateDiff / SVD video generation (24GB+ VRAM node required)
- [ ] Celery + PostgreSQL upgrade for multi-GPU scale
- [ ] `CharacterProfile.base_model` migration tooling for SD ecosystem upgrades
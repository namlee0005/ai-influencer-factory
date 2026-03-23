"""
Pydantic v2 data contracts for AI Influencer Factory.
All datetimes are UTC-aware. Decimal for any financial fields if added later.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
            raise ValueError(
                "CharacterProfile requires at least one of: lora_path, ip_adapter_reference_image"
            )
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
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def created_at_must_be_utc(self) -> "GenerationJob":
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC)")
        return self
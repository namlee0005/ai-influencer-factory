"""
VisualArtist: Claude-powered six-layer prompt engineer.

Produces text-only prompts — never node IDs, workflow JSON, or ComfyUI internals.
System prompt is cached (ephemeral) to minimize token costs across repeated calls.
Uses claude-opus-4-6 with adaptive thinking for prompt quality.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import anthropic

from models import CharacterProfile, WorkflowRequest

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """You are a professional AI image prompt engineer for a social media influencer pipeline.

Your ONLY job is to produce a six-layer positive prompt for Stable Diffusion / SDXL.
Output format — exactly six comma-separated layers on ONE line, nothing else:
  SUBJECT, STYLE, LIGHTING, CAMERA, QUALITY, NEGATIVE

Rules:
- SUBJECT: describe the person's pose, expression, action, and immediate environment
- STYLE: art direction (photorealistic, editorial, cinematic, etc.)
- LIGHTING: specific lighting setup (golden hour, studio softbox, neon ambient, etc.)
- CAMERA: lens + framing (85mm portrait lens, wide-angle establishing shot, etc.)
- QUALITY: technical quality keywords (8k, sharp focus, award-winning photography)
- NEGATIVE: comma-separated negative descriptors (blurry, deformed, watermark, text, logo)

Never output JSON, node IDs, workflow keys, or markdown.
Never describe the character's name or identity — only visual attributes.
Never exceed 200 tokens total."""


@lru_cache(maxsize=1)
def _get_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


async def generate_prompt(
    profile: CharacterProfile,
    request: WorkflowRequest,
) -> str:
    """
    Generate a six-layer Stable Diffusion prompt.

    Returns a single comma-separated string.
    Raises ValueError if the response is empty or contains disallowed content.
    """
    client = _get_client()

    user_content = (
        f"Scenario: {request.scenario}\n"
        f"Scene description: {request.scene_description}\n"
        f"Aspect ratio: {request.aspect_ratio}\n"
        f"Quality preset: {request.quality_preset}\n"
        f"Character trigger words: {', '.join(profile.trigger_words)}\n"
        f"Base negative: {profile.negative_prompt}"
    )

    logger.info(
        "visual_artist_request",
        extra={"scenario": request.scenario, "quality": request.quality_preset},
    )

    # Stream with prompt caching on the system prompt
    async with client.messages.stream(
        model=MODEL,
        max_tokens=256,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cached after first call
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        final = await stream.get_final_message()

    text_blocks = [b.text for b in final.content if b.type == "text"]
    if not text_blocks:
        raise ValueError("VisualArtist returned no text output")

    prompt = text_blocks[0].strip()

    # Safety gate: reject if it looks like workflow JSON leaked through
    if any(kw in prompt for kw in ("{", "}", "class_type", "node_id", "prompt_id")):
        raise ValueError(f"VisualArtist output contains disallowed content: {prompt[:80]}")

    logger.info(
        "visual_artist_response",
        extra={"prompt_length": len(prompt), "cached": final.usage.cache_read_input_tokens > 0},
    )
    return prompt


class VisualArtist:
    """Thin wrapper for use as an injectable dependency."""

    async def generate(
        self,
        profile: CharacterProfile,
        request: WorkflowRequest,
    ) -> str:
        return await generate_prompt(profile, request)
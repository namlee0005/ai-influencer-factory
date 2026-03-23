"""
Jinja2-based ComfyUI workflow template builder.

Rules:
- Templates live in workflows/templates/ as *.json.j2
- Only prompt inputs are parameterised — node topology is NEVER modified
- render() validates output is parseable JSON before returning
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from models import CharacterProfile, WorkflowRequest

TEMPLATE_DIR = Path(__file__).parent / "workflows" / "templates"
SCENARIO_TEMPLATE_MAP: dict[str, str] = {
    "travel":    "travel.json.j2",
    "food":      "food.json.j2",
    "fashion":   "fashion.json.j2",
    "lifestyle": "fashion.json.j2",  # reuse fashion template for V1
    "gym":       "fashion.json.j2",  # same; add dedicated template in V2
    "studio":    "fashion.json.j2",
}

QUALITY_STEPS: dict[str, int] = {"draft": 20, "standard": 30, "high": 50}
QUALITY_CFG: dict[str, float]  = {"draft": 5.0, "standard": 7.0, "high": 8.5}

ASPECT_DIMS: dict[str, tuple[int, int]] = {
    "9:16":  (768, 1344),
    "1:1":   (1024, 1024),
    "16:9":  (1344, 768),
}

_jinja_env: Environment | None = None


def _get_env() -> Environment:
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            undefined=StrictUndefined,
            autoescape=False,
        )
    return _jinja_env


def build_workflow(
    request: WorkflowRequest,
    profile: CharacterProfile,
    prompt: str,
    seed: int,
) -> dict:
    """
    Render and return a ComfyUI workflow dict.

    Args:
        request:  WorkflowRequest from the caller.
        profile:  CharacterProfile with LoRA / IP-Adapter refs.
        prompt:   Six-layer positive prompt from VisualArtist.
        seed:     Deterministic integer seed.

    Returns:
        Parsed workflow dict ready for ComfyUIClient.submit().

    Raises:
        ValueError: if template is missing or render produces invalid JSON.
    """
    template_name = SCENARIO_TEMPLATE_MAP.get(request.scenario)
    if not template_name:
        raise ValueError(f"No template mapped for scenario '{request.scenario}'")

    width, height = ASPECT_DIMS[request.aspect_ratio]
    steps = QUALITY_STEPS[request.quality_preset]
    cfg   = QUALITY_CFG[request.quality_preset]

    context = {
        "positive_prompt":       prompt,
        "negative_prompt":       profile.negative_prompt,
        "seed":                  seed,
        "steps":                 steps,
        "cfg":                   cfg,
        "width":                 width,
        "height":                height,
        "base_model":            profile.base_model,
        "lora_path":             profile.lora_path or "",
        "lora_weight":           profile.lora_weight,
        "ip_adapter_ref":        profile.ip_adapter_reference_image or "",
        "use_lora":              bool(profile.lora_path),
        "use_ip_adapter":        bool(profile.ip_adapter_reference_image),
        "trigger_words":         " ".join(profile.trigger_words),
    }

    env = _get_env()
    try:
        template = env.get_template(template_name)
    except Exception as exc:
        raise ValueError(f"Template '{template_name}' not found in {TEMPLATE_DIR}: {exc}") from exc

    rendered = template.render(**context)
    try:
        return json.loads(rendered)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Rendered template is not valid JSON: {exc}") from exc
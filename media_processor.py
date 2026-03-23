"""
MediaPostProcessor: PIL resize/watermark + insightface CPU face-similarity gate.

insightface runs on CPU to avoid VRAM conflict with the SD pipeline.
FFmpeg subprocess handles any video encode in V2; image-only for V1.
All output paths are sanitized against OUTPUT_ROOT.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from models import CharacterProfile, WorkflowRequest

logger = logging.getLogger(__name__)

OUTPUT_ROOT = os.path.realpath(os.environ.get("OUTPUT_ROOT", str(Path(__file__).parent / "output")))

PLATFORM_SPECS: dict[str, tuple[int, int]] = {
    "9:16":  (1080, 1920),
    "1:1":   (1080, 1080),
    "16:9":  (1920, 1080),
}

WATERMARK_TEXT = "@ai_influencer"
FACE_THRESHOLD = 0.75

# insightface model loaded lazily to avoid import-time cost
_face_app = None


def _get_face_app():
    global _face_app
    if _face_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            _face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
            _face_app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 = CPU
        except ImportError:
            logger.warning("insightface not installed — face similarity gate disabled")
    return _face_app


def _safe_output_path(job_id: str, suffix: str) -> str:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    filename = f"{job_id}_{suffix}.png"
    # Strip any path components from job_id to prevent traversal
    safe_filename = os.path.basename(filename)
    resolved = os.path.realpath(os.path.join(OUTPUT_ROOT, safe_filename))
    if not resolved.startswith(OUTPUT_ROOT + os.sep) and resolved != OUTPUT_ROOT:
        raise ValueError(f"Path traversal detected for job_id='{job_id}'")
    return resolved


def _resize_image(img: Image.Image, aspect_ratio: str) -> Image.Image:
    target_w, target_h = PLATFORM_SPECS.get(aspect_ratio, (1080, 1080))
    return img.resize((target_w, target_h), Image.LANCZOS)


def _add_watermark(img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=36)
    except (IOError, OSError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    margin = 20
    x = w - text_w - margin
    y = h - text_h - margin

    # Shadow for legibility
    draw.text((x + 2, y + 2), WATERMARK_TEXT, fill=(0, 0, 0, 128), font=font)
    draw.text((x, y), WATERMARK_TEXT, fill=(255, 255, 255, 200), font=font)
    return img


def _compute_face_similarity(img_bytes: bytes, reference_path: str) -> float | None:
    """Return cosine similarity between primary face in image and reference, or None."""
    face_app = _get_face_app()
    if face_app is None:
        return None

    import numpy as np
    import cv2

    try:
        # Decode generated image
        nparr = np.frombuffer(img_bytes, np.uint8)
        gen_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gen_faces = face_app.get(gen_img)
        if not gen_faces:
            logger.warning("no_face_detected_in_generated_image")
            return None

        # Decode reference image
        ref_img = cv2.imread(reference_path)
        if ref_img is None:
            logger.warning("reference_image_not_readable", extra={"path": reference_path})
            return None
        ref_faces = face_app.get(ref_img)
        if not ref_faces:
            logger.warning("no_face_detected_in_reference")
            return None

        gen_emb = gen_faces[0].normed_embedding
        ref_emb = ref_faces[0].normed_embedding
        similarity = float(np.dot(gen_emb, ref_emb))
        return similarity

    except Exception as exc:
        logger.error("face_similarity_error", extra={"error": str(exc)}, exc_info=True)
        return None


class MediaPostProcessor:
    async def process(
        self,
        image_bytes: bytes,
        profile: CharacterProfile,
        request: WorkflowRequest,
        job_id: str,
    ) -> tuple[list[str], float | None]:
        """
        Resize, watermark, and face-gate an image.

        Returns (output_paths, face_similarity_score).
        score is None when insightface is unavailable or no reference image exists.
        """
        # Run CPU-bound work off the event loop
        return await asyncio.to_thread(
            self._process_sync, image_bytes, profile, request, job_id
        )

    def _process_sync(
        self,
        image_bytes: bytes,
        profile: CharacterProfile,
        request: WorkflowRequest,
        job_id: str,
    ) -> tuple[list[str], float | None]:
        import io

        # Face similarity gate
        score: float | None = None
        if profile.ip_adapter_reference_image:
            ref_path = os.path.realpath(profile.ip_adapter_reference_image)
            score = _compute_face_similarity(image_bytes, ref_path)
            logger.info(
                "face_similarity",
                extra={"job_id": job_id, "score": score, "threshold": FACE_THRESHOLD},
            )

        # PIL processing
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        img = _resize_image(img, request.aspect_ratio)
        img = _add_watermark(img)

        # Convert back to RGB for PNG save
        final = Image.new("RGB", img.size, (255, 255, 255))
        final.paste(img, mask=img.split()[3])

        out_path = _safe_output_path(job_id, request.aspect_ratio.replace(":", "x"))
        final.save(out_path, "PNG", optimize=True)
        logger.info("image_saved", extra={"path": out_path})

        return [out_path], score
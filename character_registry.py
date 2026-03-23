"""
SQLite-backed CRUD for CharacterProfile.

Path traversal guard: all lora_path / ip_adapter_reference_image values are
resolved with os.path.realpath() and asserted to start with ALLOWED_ASSET_ROOT.
Set ALLOWED_ASSET_ROOT via env var ASSET_ROOT (defaults to ./assets).
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from models import CharacterProfile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_DEFAULT_ASSET_ROOT = str(Path(__file__).parent / "assets")
ALLOWED_ASSET_ROOT: str = os.environ.get("ASSET_ROOT", _DEFAULT_ASSET_ROOT)
DB_PATH: str = os.environ.get("CHARACTER_DB_PATH", str(Path(__file__).parent / "characters.db"))

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS characters (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    data        TEXT NOT NULL  -- full JSON blob
);
"""


# ---------------------------------------------------------------------------
# Path guard
# ---------------------------------------------------------------------------

def _safe_asset_path(raw: str | None) -> str | None:
    """Resolve and validate that a path stays within ALLOWED_ASSET_ROOT."""
    if raw is None:
        return None
    resolved = os.path.realpath(raw)
    allowed = os.path.realpath(ALLOWED_ASSET_ROOT)
    if not resolved.startswith(allowed + os.sep) and resolved != allowed:
        raise ValueError(
            f"Path traversal detected: '{raw}' resolves outside asset root '{allowed}'"
        )
    return resolved


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(_CREATE_TABLE)
        conn.commit()
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class CharacterRegistry:
    """Synchronous SQLite registry. Async wrappers can be added via run_in_executor."""

    def save(self, profile: CharacterProfile) -> CharacterProfile:
        """Insert or replace a CharacterProfile. Validates asset paths before write."""
        # Validate paths before persisting
        _safe_asset_path(profile.lora_path)
        _safe_asset_path(profile.ip_adapter_reference_image)

        data = profile.model_dump(mode="json")
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO characters (id, name, data) VALUES (?, ?, ?)",
                (profile.id, profile.name, json.dumps(data)),
            )
            conn.commit()
        return profile

    def get(self, character_id: str) -> CharacterProfile | None:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT data FROM characters WHERE id = ?", (character_id,)
            ).fetchone()
        if row is None:
            return None
        return CharacterProfile.model_validate(json.loads(row["data"]))

    def list_all(self) -> list[CharacterProfile]:
        with _get_conn() as conn:
            rows = conn.execute("SELECT data FROM characters").fetchall()
        return [CharacterProfile.model_validate(json.loads(r["data"])) for r in rows]

    def delete(self, character_id: str) -> bool:
        """Returns True if a record was deleted."""
        with _get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM characters WHERE id = ?", (character_id,)
            )
            conn.commit()
        return cursor.rowcount > 0

    def update(self, profile: CharacterProfile) -> CharacterProfile:
        """Update an existing profile; raises KeyError if not found."""
        if self.get(profile.id) is None:
            raise KeyError(f"Character '{profile.id}' not found")
        return self.save(profile)
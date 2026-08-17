"""SQLite-backed dataset of agent runs for RL / post-training.

Each row is one saved agent run: the system prompt, the user request, the full
project-state observation, the complete tool/reasoning trace, the final message
transcript, and references to the inspect JPEGs the agent saw. This is enough to
replay a run as an RL trajectory (observation → actions → outcome) offline.

The DB lives at reports/rl_dataset.db by default (override with AGENT_RL_DB).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


def default_db_path() -> Path:
    env = (os.environ.get("AGENT_RL_DB") or "").strip()
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[1]
    return root / "reports" / "rl_dataset.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rl_samples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    project_id    TEXT,
    video_id      TEXT,
    frame_idx     INTEGER,
    provider      TEXT,
    model         TEXT,
    system_prompt TEXT,
    user_text     TEXT,
    state_json    TEXT,   -- project/video observation (overview)
    trace_json    TEXT,   -- full recorded SSE event trace
    messages_json TEXT,   -- LLM transcript (images stripped to metadata)
    images_json   TEXT,   -- [{file, path, frame_idx, video_id}]
    dump_dir      TEXT,
    reward        REAL,   -- optional scalar reward/label
    label         TEXT,   -- optional categorical label
    notes         TEXT
);
"""


class RLDatasetStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def insert_sample(self, sample: dict[str, Any]) -> int:
        cols = (
            "created_at", "project_id", "video_id", "frame_idx", "provider", "model",
            "system_prompt", "user_text", "state_json", "trace_json", "messages_json",
            "images_json", "dump_dir", "reward", "label", "notes",
        )
        row = {c: sample.get(c) for c in cols}
        row["created_at"] = row.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
        for jkey in ("state_json", "trace_json", "messages_json", "images_json"):
            v = row.get(jkey)
            if v is not None and not isinstance(v, str):
                row[jkey] = json.dumps(v, default=str)
        placeholders = ", ".join("?" for _ in cols)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO rl_samples ({', '.join(cols)}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )
            conn.commit()
            return int(cur.lastrowid)

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM rl_samples").fetchone()
            return int(n)

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM rl_samples").fetchone()
            (labeled,) = conn.execute(
                "SELECT COUNT(*) FROM rl_samples WHERE reward IS NOT NULL"
            ).fetchone()
        return {"count": int(n), "labeled": int(labeled), "db_path": str(self.db_path)}


_STORE: Optional[RLDatasetStore] = None


def get_store() -> RLDatasetStore:
    global _STORE
    if _STORE is None:
        _STORE = RLDatasetStore()
    return _STORE

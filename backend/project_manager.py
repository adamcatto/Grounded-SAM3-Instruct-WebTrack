"""
Project and video metadata management.
Projects are stored in ~/.sam3_zero_projects/<project-id>-<name>/config.json
"""

import re
import uuid
import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path.home() / ".sam3_zero_projects"

OBJECT_COLORS = [
    "#5B8DD9",  # blue
    "#E8A445",  # orange
    "#52C41A",  # green
    "#722ED1",  # purple
    "#F5222D",  # red
    "#13C2C2",  # teal
    "#FA8C16",  # dark orange
    "#EB2F96",  # pink
]


class ProjectManager:
    def __init__(self):
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def _get_lock(self, pid: str) -> threading.Lock:
        with self._locks_lock:
            if pid not in self._locks:
                self._locks[pid] = threading.Lock()
            return self._locks[pid]

    # ─── Projects ────────────────────────────────────────────────────────────

    def _slugify(self, name: str) -> str:
        """Convert a project name to a filesystem-safe slug."""
        slug = re.sub(r"[^\w\-.]", "_", name).strip("_.-")
        return slug[:64]

    def _find_video_dir(self, pid: str, vid: str) -> Optional[Path]:
        """Locate the video directory by scanning for dirs starting with vid."""
        videos_root = self._project_dir(pid) / "videos"
        for d in videos_root.iterdir():
            if d.name == vid or d.name.startswith(vid + "_"):
                return d
        return None

    def _find_project_dir(self, pid: str) -> Optional[Path]:
        """Locate the project directory by scanning for dirs starting with pid."""
        for d in BASE_DIR.iterdir():
            if d.name == pid or d.name.startswith(pid + "-"):
                return d
        return None

    def _project_dir(self, pid: str) -> Path:
        d = self._find_project_dir(pid)
        if d is None:
            raise ValueError(f"Project {pid} not found")
        return d

    def list_projects(self) -> list[dict]:
        projects = []
        for d in sorted(BASE_DIR.iterdir()):
            cfg = d / "config.json"
            if cfg.exists():
                try:
                    projects.append(json.loads(cfg.read_text()))
                except Exception:
                    pass
        return projects

    def create_project(self, name: str) -> dict:
        pid = str(uuid.uuid4())[:8]
        slug = self._slugify(name)
        dir_name = f"{pid}-{slug}" if slug else pid
        project_dir = BASE_DIR / dir_name
        project_dir.mkdir(parents=True)
        (project_dir / "videos").mkdir()
        config = {
            "id": pid,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "videos": {},
        }
        self._save_config(pid, config)
        return config

    def get_project(self, pid: str) -> Optional[dict]:
        d = self._find_project_dir(pid)
        if d is None:
            return None
        cfg = d / "config.json"
        if not cfg.exists():
            return None
        return json.loads(cfg.read_text())

    def update_project(self, pid: str, updates: dict) -> dict:
        config = self.get_project(pid)
        if config is None:
            raise ValueError(f"Project {pid} not found")
        for k, v in updates.items():
            if k not in ("id", "created_at", "videos"):
                config[k] = v
        self._save_config(pid, config)
        return config

    def delete_project(self, pid: str):
        d = self._find_project_dir(pid)
        if d is not None:
            shutil.rmtree(d)

    # ─── Videos ──────────────────────────────────────────────────────────────

    def add_video(self, pid: str, name: str, source_path: str,
                  num_frames: int, fps: float, width: int, height: int) -> dict:
        config = self.get_project(pid)
        if config is None:
            raise ValueError(f"Project {pid} not found")
        vid = str(uuid.uuid4())[:8]
        slug = self._slugify(name)
        dir_name = f"{vid}_{slug}" if slug else vid
        video_dir = self._project_dir(pid) / "videos" / dir_name
        video_dir.mkdir(parents=True)
        (video_dir / "frames").mkdir()
        (video_dir / "masks").mkdir()
        (video_dir / "bboxes").mkdir()
        video_meta = {
            "id": vid,
            "name": name,
            "source_path": source_path,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "start_frame": 0,
            "objects": {},
            "point_prompts": {},
            "sam3_session_id": None,
            "propagated_frames": [],
            "propagation_complete": False,
            "frames_extracted": False,
            "all_frames_extracted": False,
            "preview_indices": [],
            "annotated_anchors": [],
        }
        config["videos"][vid] = video_meta
        self._save_config(pid, config)
        return video_meta

    def get_video(self, pid: str, vid: str) -> Optional[dict]:
        config = self.get_project(pid)
        if config is None:
            return None
        return config["videos"].get(vid)

    def update_video(self, pid: str, vid: str, updates: dict):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found in project {pid}")
        config["videos"][vid].update(updates)
        self._save_config(pid, config)
        return config["videos"][vid]

    def remove_video(self, pid: str, vid: str):
        config = self.get_project(pid)
        if config is None:
            raise ValueError(f"Project {pid} not found")
        config["videos"].pop(vid, None)
        self._save_config(pid, config)
        video_dir = self._find_video_dir(pid, vid)
        if video_dir is not None and video_dir.exists():
            shutil.rmtree(video_dir)

    # ─── Objects ─────────────────────────────────────────────────────────────

    def add_object(
        self,
        pid: str,
        vid: str,
        name: str,
        color: Optional[str] = None,
        description: str = "",
        min_instances: int = 1,
        max_instances: int = 1,
    ) -> dict:
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        existing_ids = list(config["videos"][vid]["objects"].keys())
        new_id = str(len(existing_ids) + 1)
        while new_id in existing_ids:
            new_id = str(int(new_id) + 1)
        if color is None:
            idx = len(existing_ids) % len(OBJECT_COLORS)
            color = OBJECT_COLORS[idx]
        obj = {
            "id": new_id,
            "name": name,
            "color": color,
            "description": description,
            "min_instances": min_instances,
            "max_instances": max_instances,
        }
        config["videos"][vid]["objects"][new_id] = obj
        # Initialize instance_groups entry: first SAM obj_id slot = int(new_id)
        # For multi-instance objects, additional slots are allocated on demand.
        config["videos"][vid].setdefault("instance_groups", {})[new_id] = [int(new_id)]
        self._save_config(pid, config)
        return obj

    def update_object(self, pid: str, vid: str, obj_id: str, **kwargs) -> dict:
        """Update any field of an object (name, color, description, min_instances, max_instances)."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        obj = config["videos"][vid]["objects"].get(obj_id)
        if obj is None:
            raise ValueError(f"Object {obj_id} not found in video {vid}")
        allowed = {"name", "color", "description", "min_instances", "max_instances"}
        for k, v in kwargs.items():
            if k in allowed:
                obj[k] = v
        self._save_config(pid, config)
        return obj

    def remove_object(self, pid: str, vid: str, obj_id: str):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        config["videos"][vid]["objects"].pop(obj_id, None)
        config["videos"][vid]["point_prompts"].pop(obj_id, None)
        self._save_config(pid, config)

    def rename_object(self, pid: str, vid: str, obj_id: str, new_name: str):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        if obj_id in config["videos"][vid]["objects"]:
            config["videos"][vid]["objects"][obj_id]["name"] = new_name
            self._save_config(pid, config)

    def get_object_description(self, pid: str, vid: str, obj_id: str) -> str:
        """Return effective text prompt for an object (description if set, else name)."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            return ""
        obj = config["videos"][vid]["objects"].get(str(obj_id), {})
        return obj.get("description") or obj.get("name", "")

    # ─── Point Prompts ────────────────────────────────────────────────────────

    def save_point_prompts(self, pid: str, vid: str, obj_id: str,
                           frame_idx: int, points: list, labels: list):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        prompts = config["videos"][vid].setdefault("point_prompts", {})
        obj_prompts = prompts.setdefault(str(obj_id), {})
        obj_prompts[str(frame_idx)] = {"points": points, "labels": labels}
        self._save_config(pid, config)

    def clear_point_prompts(self, pid: str, vid: str, obj_id: str):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        config["videos"][vid].get("point_prompts", {}).pop(str(obj_id), None)
        self._save_config(pid, config)

    def clear_object_frame_prompt(self, pid: str, vid: str, obj_id: str, frame_idx: int):
        """Remove point prompts for a single object on a single frame."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        obj_prompts = config["videos"][vid].get("point_prompts", {}).get(str(obj_id), {})
        if str(frame_idx) in obj_prompts:
            del obj_prompts[str(frame_idx)]
            self._save_config(pid, config)

    def clear_frame_prompts(self, pid: str, vid: str, frame_idx: int):
        """Remove all point prompts for a specific frame across all objects."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        key = str(frame_idx)
        prompts = config["videos"][vid].get("point_prompts", {})
        changed = False
        for obj_prompts in prompts.values():
            if key in obj_prompts:
                del obj_prompts[key]
                changed = True
        if changed:
            self._save_config(pid, config)

    def get_all_point_prompts(self, pid: str, vid: str) -> dict:
        """Return {obj_id: {frame_idx: {points, labels}}}"""
        video = self.get_video(pid, vid)
        if video is None:
            return {}
        return video.get("point_prompts", {})

    # ─── Propagation ─────────────────────────────────────────────────────────

    def mark_frame_propagated(self, pid: str, vid: str, frame_idx: int):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            return
        frames = config["videos"][vid].setdefault("propagated_frames", [])
        if frame_idx not in frames:
            frames.append(frame_idx)
        self._save_config(pid, config)

    def mark_propagation_complete(self, pid: str, vid: str):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            return
        config["videos"][vid]["propagation_complete"] = True
        self._save_config(pid, config)

    # ─── Paths ───────────────────────────────────────────────────────────────

    def video_dir(self, pid: str, vid: str) -> Path:
        d = self._find_video_dir(pid, vid)
        if d is None:
            raise ValueError(f"Video {vid} not found in project {pid}")
        return d

    def frames_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "frames"

    def annotated_frames_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "annotated_frames"

    def masks_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "masks"

    def bboxes_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "bboxes"

    # ─── Internal ────────────────────────────────────────────────────────────

    def _save_config(self, pid: str, config: dict):
        cfg_path = self._project_dir(pid) / "config.json"
        tmp_path = cfg_path.with_suffix(".json.tmp")
        with self._get_lock(pid):
            tmp_path.write_text(json.dumps(config, indent=2))
            os.replace(str(tmp_path), str(cfg_path))

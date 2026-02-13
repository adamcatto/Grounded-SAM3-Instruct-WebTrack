"""
Project and video metadata management.
Projects are stored in /opt/.sam3_projects/<project-id>/config.json
"""

import uuid
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path("/opt/.sam3_projects")

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

    # ─── Projects ────────────────────────────────────────────────────────────

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
        project_dir = BASE_DIR / pid
        project_dir.mkdir(parents=True)
        (project_dir / "videos").mkdir()
        display_name = f"{name} - {pid}"
        config = {
            "id": pid,
            "name": display_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "videos": {},
        }
        self._save_config(pid, config)
        return config

    def get_project(self, pid: str) -> Optional[dict]:
        cfg = BASE_DIR / pid / "config.json"
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
        project_dir = BASE_DIR / pid
        if project_dir.exists():
            shutil.rmtree(project_dir)

    # ─── Videos ──────────────────────────────────────────────────────────────

    def add_video(self, pid: str, name: str, source_path: str,
                  num_frames: int, fps: float, width: int, height: int) -> dict:
        config = self.get_project(pid)
        if config is None:
            raise ValueError(f"Project {pid} not found")
        vid = str(uuid.uuid4())[:8]
        video_dir = BASE_DIR / pid / "videos" / vid
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
            "objects": {},
            "point_prompts": {},
            "sam3_session_id": None,
            "propagated_frames": [],
            "propagation_complete": False,
            "frames_extracted": False,
            "all_frames_extracted": False,
            "preview_indices": [],
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
        video_dir = BASE_DIR / pid / "videos" / vid
        if video_dir.exists():
            shutil.rmtree(video_dir)

    # ─── Objects ─────────────────────────────────────────────────────────────

    def add_object(self, pid: str, vid: str, name: str, color: Optional[str] = None) -> dict:
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
        obj = {"id": new_id, "name": name, "color": color}
        config["videos"][vid]["objects"][new_id] = obj
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

    def frames_dir(self, pid: str, vid: str) -> Path:
        return BASE_DIR / pid / "videos" / vid / "frames"

    def annotated_frames_dir(self, pid: str, vid: str) -> Path:
        return BASE_DIR / pid / "videos" / vid / "annotated_frames"

    def masks_dir(self, pid: str, vid: str) -> Path:
        return BASE_DIR / pid / "videos" / vid / "masks"

    def bboxes_dir(self, pid: str, vid: str) -> Path:
        return BASE_DIR / pid / "videos" / vid / "bboxes"

    # ─── Internal ────────────────────────────────────────────────────────────

    def _save_config(self, pid: str, config: dict):
        cfg_path = BASE_DIR / pid / "config.json"
        cfg_path.write_text(json.dumps(config, indent=2))

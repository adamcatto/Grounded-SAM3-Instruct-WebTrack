"""
Project and video metadata management.
Projects live under a configurable root directory (see default_projects_base_dir).
"""

import re
import uuid
import json
import fcntl
import os
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from anchor_helpers import is_anchor_labeling_complete


def _env_projects_path_from_environ() -> tuple[Optional[Path], Optional[str]]:
    """Return (expanded path, env key used) from SAM3_TRACKING_PROJECTS_DIR or SAM3_PROJECTS_DIR."""
    for key in ("SAM3_TRACKING_PROJECTS_DIR", "SAM3_PROJECTS_DIR"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return Path(raw).expanduser(), key
    return None, None


def default_projects_base_dir() -> Path:
    """Filesystem root for project folders (not overridden at runtime)."""
    p, _ = _env_projects_path_from_environ()
    if p is not None:
        return p
    return Path.home() / ".sam3_zero_projects"


def env_projects_var_name() -> Optional[str]:
    _, key = _env_projects_path_from_environ()
    return key

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
        self._root_lock = threading.Lock()
        self._base_dir = default_projects_base_dir().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    def projects_base_dir(self) -> Path:
        return self._base_dir

    def get_projects_root_info(self) -> dict:
        env_path = default_projects_base_dir().resolve()
        # Repo root = parent of backend/ folder (where start_frontend.sh lives).
        app_root = Path(__file__).resolve().parents[1]
        home = Path.home().resolve()
        return {
            "active_root": str(self._base_dir),
            "env_default_root": str(env_path),
            "env_var": env_projects_var_name(),
            "app_root": str(app_root),
            "home": str(home),
        }

    def set_projects_root(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.exists():
            raise ValueError(f"Path not found: {path}")
        if not p.is_dir():
            raise ValueError(f"Not a directory: {path}")
        resolved = p.resolve()
        with self._root_lock:
            self._base_dir = resolved
            self._base_dir.mkdir(parents=True, exist_ok=True)
        return self._base_dir

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
        for d in self._base_dir.iterdir():
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
        for d in sorted(self._base_dir.iterdir()):
            cfg = d / "config.json"
            if cfg.exists():
                try:
                    raw = json.loads(cfg.read_text())
                    projects.append(self._normalize_project_config(raw))
                except Exception:
                    pass
        return projects

    def create_project(self, name: str) -> dict:
        pid = str(uuid.uuid4())[:8]
        slug = self._slugify(name)
        dir_name = f"{pid}-{slug}" if slug else pid
        project_dir = self._base_dir / dir_name
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
        config = self._normalize_project_config(json.loads(cfg.read_text()))
        # Merge propagated_frames from per-video progress files
        for vid in config.get("videos", {}):
            config["videos"][vid]["propagated_frames"] = self._read_propagated_frames(pid, vid)
        return config

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
            "anchor_labeling_timing": {"frames": {}, "video": {}},
            "anchor_labeling_complete": False,
            "whole_video_inference": {
                "status": "none",
                "updated_at": None,
                "host": None,
            },
        }
        config["videos"][vid] = video_meta
        self._save_config(pid, config)
        return video_meta

    def get_video(self, pid: str, vid: str) -> Optional[dict]:
        config = self.get_project(pid)
        if config is None:
            return None
        vm = config["videos"].get(vid)
        if vm is None:
            return None
        # Merge propagated_frames from the per-video progress file
        vm["propagated_frames"] = self._read_propagated_frames(pid, vid)
        return vm

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
        igs = config["videos"][vid].get("instance_groups") or {}
        igs.pop(obj_id, None)
        self._save_config(pid, config)

    def restore_object_entry(
        self,
        pid: str,
        vid: str,
        obj_id: str,
        object_entry: dict,
        point_prompts_for_obj: Optional[dict],
        instance_group: Optional[list],
    ):
        """Re-insert an object and its prompts after undo of removal."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        config["videos"][vid]["objects"][str(obj_id)] = dict(object_entry)
        prompts = config["videos"][vid].setdefault("point_prompts", {})
        if point_prompts_for_obj is None:
            pass
        elif not point_prompts_for_obj:
            prompts.pop(str(obj_id), None)
        else:
            prompts[str(obj_id)] = dict(point_prompts_for_obj)
        if instance_group is not None:
            config["videos"][vid].setdefault("instance_groups", {})[str(obj_id)] = list(instance_group)
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

    # ─── Propagation progress (per-video files) ────────────────────────────

    def _propagation_progress_path(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "propagation_progress.txt"

    def _read_propagated_frames(self, pid: str, vid: str) -> list[int]:
        """Read propagated frame indices from the per-video progress file."""
        try:
            path = self._propagation_progress_path(pid, vid)
        except ValueError:
            return []
        if not path.exists():
            return []
        frames: set[int] = set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    frames.add(int(line))
                except ValueError:
                    pass
        return sorted(frames)

    def mark_frame_propagated(self, pid: str, vid: str, frame_idx: int):
        """Append frame index to the per-video progress file.

        This is the hot path during propagation — uses a simple file append
        instead of a full config.json read-modify-write cycle.  Appending
        a short line (<4096 bytes) is atomic on Linux/POSIX, so concurrent
        appenders (different videos = different files) are safe without locking.
        """
        path = self._propagation_progress_path(pid, vid)
        with open(path, "a") as f:
            f.write(f"{frame_idx}\n")

    def clear_propagated_frames(self, pid: str, vid: str):
        """Delete the progress file (used during video reset)."""
        try:
            path = self._propagation_progress_path(pid, vid)
        except ValueError:
            return
        if path.exists():
            path.unlink()

    def trim_propagated_frames(self, pid: str, vid: str, keep_below: int) -> int:
        """Rewrite progress file keeping only frames < keep_below.

        Returns the number of kept frames.
        """
        frames = [f for f in self._read_propagated_frames(pid, vid) if f < keep_below]
        try:
            path = self._propagation_progress_path(pid, vid)
        except ValueError:
            return 0
        if frames:
            path.write_text("".join(f"{f}\n" for f in frames))
        elif path.exists():
            path.unlink()
        return len(frames)

    def mark_propagation_complete(self, pid: str, vid: str):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            return
        config["videos"][vid]["propagation_complete"] = True
        winf = dict(config["videos"][vid].get("whole_video_inference") or {})
        winf["status"] = "complete"
        winf["updated_at"] = datetime.now(timezone.utc).isoformat()
        config["videos"][vid]["whole_video_inference"] = winf
        self._save_config(pid, config)

    def set_video_inference_status(
        self,
        pid: str,
        vid: str,
        status: str,
        host: str | None = None,
    ) -> Optional[dict]:
        allowed = frozenset({"none", "running", "complete", "failed"})
        if status not in allowed:
            raise ValueError(f"invalid inference status: {status!r}")
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            return None
        winf = dict(config["videos"][vid].get("whole_video_inference") or {})
        winf["status"] = status
        winf["updated_at"] = datetime.now(timezone.utc).isoformat()
        if host is not None:
            winf["host"] = host
        if status == "none":
            winf["host"] = None
        config["videos"][vid]["whole_video_inference"] = winf
        self._save_config(pid, config)
        return config["videos"][vid]

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

    def masks_sqlite_path(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "masks.sqlite"

    # ─── Internal ────────────────────────────────────────────────────────────

    def _normalize_video_meta(self, vm: dict) -> dict:
        out = dict(vm)
        winf_raw = out.get("whole_video_inference") or {}
        winf = dict(winf_raw) if isinstance(winf_raw, dict) else {}
        winf.setdefault("status", "none")
        winf.setdefault("updated_at", None)
        winf.setdefault("host", None)
        if out.get("propagation_complete"):
            winf["status"] = "complete"
        out["whole_video_inference"] = winf
        inferred = is_anchor_labeling_complete(out)
        if "anchor_labeling_complete" not in vm:
            out["anchor_labeling_complete"] = inferred
        else:
            out["anchor_labeling_complete"] = bool(vm.get("anchor_labeling_complete")) or inferred
        return out

    def _normalize_project_config(self, config: dict) -> dict:
        out = dict(config)
        vids = {}
        for vid, vm in (config.get("videos") or {}).items():
            vids[vid] = self._normalize_video_meta(vm if isinstance(vm, dict) else {})
        out["videos"] = vids
        return out

    def _save_config(self, pid: str, config: dict):
        cfg_path = self._project_dir(pid) / "config.json"
        # Strip propagated_frames from config — stored in per-video progress files
        for vid_data in (config.get("videos") or {}).values():
            if isinstance(vid_data, dict):
                vid_data.pop("propagated_frames", None)
        with self._get_lock(pid):
            # Use a unique temp file to prevent cross-process write corruption.
            # Previously all writers used the same .json.tmp path, so two
            # concurrent processes would interleave bytes in that file.
            fd, tmp_name = tempfile.mkstemp(
                dir=str(cfg_path.parent),
                prefix=".config.json.tmp.",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(config, f, indent=2)
                os.replace(tmp_name, str(cfg_path))
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

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
    return Path("/opt/projects/segmentation_tracking_projects")


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
        self._config_cache_lock = threading.Lock()
        self._config_cache: dict[str, tuple[Path, int, int, dict]] = {}
        self._progress_cache_lock = threading.Lock()
        self._progress_cache: dict[Path, tuple[int, int, list[int]]] = {}

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
        with self._config_cache_lock:
            self._config_cache.clear()
        with self._progress_cache_lock:
            self._progress_cache.clear()
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

    def create_project(
        self,
        name: str,
        parent_dir: Optional[Path] = None,
        tracking_mode: str = "segmentation_tracking",
    ) -> dict:
        pid = str(uuid.uuid4())[:8]
        slug = self._slugify(name)
        dir_name = f"{pid}-{slug}" if slug else pid
        parent = (parent_dir or self._base_dir).resolve()
        project_dir = parent / dir_name
        project_dir.mkdir(parents=True)
        (project_dir / "videos").mkdir()
        config = {
            "id": pid,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tracking_mode": tracking_mode,
            "videos": {},
        }
        self._save_config(pid, config, project_dir=project_dir)
        return config

    def resolve_project_ref(self, ref: str) -> tuple[Path, dict]:
        """Resolve a project by short id, folder name, or filesystem path."""
        ref = (ref or "").strip()
        if not ref:
            raise ValueError("Empty project reference")

        p = Path(ref).expanduser()
        if p.is_dir():
            cfg_path = p / "config.json"
            if cfg_path.is_file():
                config = self._normalize_project_config(json.loads(cfg_path.read_text()))
                return p.resolve(), config

        found = self._find_project_dir(ref)
        if found is not None:
            config = self._normalize_project_config(
                json.loads((found / "config.json").read_text())
            )
            return found.resolve(), config

        env_base = default_projects_base_dir().resolve()
        if env_base != self._base_dir.resolve():
            for d in sorted(env_base.iterdir()):
                if not d.is_dir():
                    continue
                if d.name == ref or d.name.startswith(ref + "-"):
                    cfg = d / "config.json"
                    if cfg.is_file():
                        config = self._normalize_project_config(json.loads(cfg.read_text()))
                        return d.resolve(), config

        raise ValueError(f"Project not found: {ref!r}")

    def _default_merge_parent(self, left_dir: Path, right_dir: Path) -> Path:
        left_parent = left_dir.parent.resolve()
        right_parent = right_dir.parent.resolve()
        if left_parent == right_parent and left_parent.is_dir():
            return left_parent
        return self._base_dir.resolve()

    def _iter_video_entries(
        self, project_dir: Path, config: dict
    ) -> list[tuple[str, Path, dict]]:
        videos_root = project_dir / "videos"
        if not videos_root.is_dir():
            return []
        out: list[tuple[str, Path, dict]] = []
        for vid, vm in (config.get("videos") or {}).items():
            if not isinstance(vm, dict):
                continue
            video_dir = None
            for child in videos_root.iterdir():
                if not child.is_dir():
                    continue
                if child.name == vid or child.name.startswith(vid + "_"):
                    video_dir = child
                    break
            if video_dir is None:
                raise ValueError(
                    f"Video directory for {vid!r} not found under {videos_root}"
                )
            out.append((str(vid), video_dir, dict(vm)))
        out.sort(key=lambda t: (t[2].get("name") or t[0]).lower())
        return out

    def _copy_video_into_project(
        self,
        dest_project_dir: Path,
        dest_pid: str,
        src_video_dir: Path,
        video_meta: dict,
    ) -> dict:
        new_vid = str(uuid.uuid4())[:8]
        slug = self._slugify(video_meta.get("name") or "video")
        dest_name = f"{new_vid}_{slug}" if slug else new_vid
        dest_video_dir = dest_project_dir / "videos" / dest_name
        if dest_video_dir.exists():
            raise ValueError(f"Destination video directory already exists: {dest_video_dir}")

        shutil.copytree(src_video_dir, dest_video_dir, symlinks=True)

        vm = dict(video_meta)
        vm["id"] = new_vid
        vm.pop("propagated_frames", None)
        vm["sam3_session_id"] = None

        src_resolved = src_video_dir.resolve()
        old_source = (vm.get("source_path") or "").strip()
        if old_source:
            old_path = Path(old_source)
            try:
                old_path.resolve().relative_to(src_resolved)
                for candidate in dest_video_dir.glob("source.*"):
                    if candidate.is_file() or candidate.is_symlink():
                        vm["source_path"] = str(candidate)
                        break
            except ValueError:
                pass

        return self._normalize_video_meta(vm)

    def merge_projects(
        self,
        name: str,
        left_ref: str,
        right_ref: str,
        output_parent: Optional[Path] = None,
    ) -> dict:
        """
        Copy all videos from two projects into a new project. Source projects are unchanged.
        Symlinked source videos remain symlinks in the merged project.
        """
        left_dir, left_cfg = self.resolve_project_ref(left_ref)
        right_dir, right_cfg = self.resolve_project_ref(right_ref)
        if left_dir.resolve() == right_dir.resolve():
            raise ValueError("Cannot merge a project with itself")

        parent = (output_parent or self._default_merge_parent(left_dir, right_dir)).resolve()
        parent.mkdir(parents=True, exist_ok=True)

        pid = str(uuid.uuid4())[:8]
        slug = self._slugify(name)
        dir_name = f"{pid}-{slug}" if slug else pid
        project_dir = parent / dir_name
        if project_dir.exists():
            raise ValueError(f"Project directory already exists: {project_dir}")
        project_dir.mkdir(parents=True)
        (project_dir / "videos").mkdir()

        merged_config: dict = {
            "id": pid,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "videos": {},
        }

        for _src_dir, src_cfg in ((left_dir, left_cfg), (right_dir, right_cfg)):
            for _vid, video_dir, video_meta in self._iter_video_entries(_src_dir, src_cfg):
                vm = self._copy_video_into_project(
                    project_dir, pid, video_dir, video_meta
                )
                merged_config["videos"][vm["id"]] = vm

        self._save_config(pid, merged_config, project_dir=project_dir)

        if parent != self._base_dir.resolve() and left_dir.parent.resolve() == parent:
            with self._root_lock:
                self._base_dir = parent

        return self.get_project(pid) or merged_config

    def get_project(self, pid: str) -> Optional[dict]:
        config = self._load_config(pid)
        if config is None:
            return None
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

    def initialize_registration(self, pid: str, target_size: int = 1000) -> dict:
        """Create/refresh the project-level first-frame floor registration manifest."""
        config = self.get_project(pid)
        if config is None:
            raise ValueError(f"Project {pid} not found")
        previous = config.get("registration") or {}
        previous_videos = previous.get("videos") or {}
        videos: dict[str, dict] = {}
        for vid in config.get("videos", {}):
            old = previous_videos.get(vid) or {}
            videos[vid] = {
                "frame_idx": 0,
                "points": list(old.get("points") or []),
                "labels": list(old.get("labels") or []),
                "polygon_vertices": list(old.get("polygon_vertices") or []),
                "mask_source": old.get("mask_source"),
                "mask_file": old.get("mask_file"),
                "source_corners": old.get("source_corners"),
                "homography": old.get("homography"),
                "labeled": bool(old.get("labeled")),
                "registered": bool(old.get("registered")),
                "morphology_history": list(old.get("morphology_history") or []),
                "morphology_cursor": int(old.get("morphology_cursor") or 0),
                "edge_points": dict(old.get("edge_points") or {}),
                "calibration_source": old.get("calibration_source"),
                "camera_matrix": old.get("camera_matrix"),
                "distortion_coefficients": old.get("distortion_coefficients"),
                "straightness_rms_pixels": old.get("straightness_rms_pixels"),
            }
        registration = {
            "version": 2,
            "method": "floor_mask_quadrilateral_to_square_homography",
            "orientation_policy": "unspecified_dihedral_rotation_reflection_invariant_analysis",
            "target_size": max(2, int(target_size)),
            "canvas_width": previous.get("canvas_width"),
            "canvas_height": previous.get("canvas_height"),
            "canvas_offset": previous.get("canvas_offset"),
            "warp_mode": previous.get("warp_mode"),
            "status": "complete" if videos and all(v["registered"] for v in videos.values()) else "labeling",
            "videos": videos,
        }
        config["registration"] = registration
        self._save_config(pid, config)
        return registration

    def update_registration_video(self, pid: str, vid: str, updates: dict) -> dict:
        config = self.get_project(pid)
        if config is None or vid not in config.get("videos", {}):
            raise ValueError(f"Video {vid} not found in project {pid}")
        registration = config.get("registration")
        if not registration:
            registration = self.initialize_registration(pid)
            config = self.get_project(pid)
        entry = registration.setdefault("videos", {}).setdefault(vid, {"frame_idx": 0})
        entry.update(updates)
        entries = list(registration.get("videos", {}).values())
        registration["status"] = (
            "complete" if entries and all(bool(v.get("registered")) for v in entries) else "labeling"
        )
        config["registration"] = registration
        self._save_config(pid, config)
        return entry

    def delete_project(self, pid: str):
        d = self._find_project_dir(pid)
        if d is not None:
            shutil.rmtree(d)
        with self._config_cache_lock:
            self._config_cache.pop(pid, None)
        with self._progress_cache_lock:
            for path in [p for p in self._progress_cache if str(p).startswith(str(d))]:
                self._progress_cache.pop(path, None)

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
            "anchor_batch_size": 1000,
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
            "pose_objects": {},
            "pose_annotations": {},
            "pose_memory_frames": [],
            "pose_skipped_anchor_frames": [],
            "pose_tracking": {
                "status": "none",
                "start_frame": None,
                "end_frame": None,
                "tracks_file": None,
            },
        }
        config["videos"][vid] = video_meta
        if config.get("registration"):
            config["registration"].setdefault("videos", {})[vid] = {
                "frame_idx": 0,
                "points": [],
                "labels": [],
                "polygon_vertices": [],
                "mask_source": None,
                "mask_file": None,
                "source_corners": None,
                "homography": None,
                "labeled": False,
                "registered": False,
                "morphology_history": [],
                "morphology_cursor": 0,
                "edge_points": {},
                "calibration_source": None,
                "camera_matrix": None,
                "distortion_coefficients": None,
                "straightness_rms_pixels": None,
            }
            config["registration"]["status"] = "labeling"
        self._save_config(pid, config)
        return video_meta

    def add_pose_object(self, pid: str, vid: str, name: str, color: str) -> dict:
        config = self.get_project(pid)
        if config is None or vid not in config.get("videos", {}):
            raise ValueError("Video not found")
        pose_objects = config["videos"][vid].setdefault("pose_objects", {})
        oid = str(uuid.uuid4())[:8]
        obj = {"id": oid, "name": name, "color": color, "parts": {}}
        pose_objects[oid] = obj
        config["videos"][vid].setdefault("pose_annotations", {})[oid] = {}
        self._save_config(pid, config)
        return obj

    def add_pose_part(
        self, pid: str, vid: str, object_id: str, name: str, color: str,
    ) -> dict:
        config = self.get_project(pid)
        if config is None or vid not in config.get("videos", {}):
            raise ValueError("Video not found")
        obj = config["videos"][vid].setdefault("pose_objects", {}).get(object_id)
        if obj is None:
            raise ValueError("Pose object not found")
        part_id = str(uuid.uuid4())[:8]
        part = {"id": part_id, "name": name, "color": color}
        obj.setdefault("parts", {})[part_id] = part
        config["videos"][vid].setdefault("pose_annotations", {}).setdefault(object_id, {})[part_id] = {}
        self._save_config(pid, config)
        return part

    def set_pose_annotation(
        self,
        pid: str,
        vid: str,
        object_id: str,
        part_id: str,
        frame_idx: int,
        x: Optional[float],
        y: Optional[float],
        visible: bool = True,
    ) -> dict:
        config = self.get_project(pid)
        if config is None or vid not in config.get("videos", {}):
            raise ValueError("Video not found")
        obj = config["videos"][vid].setdefault("pose_objects", {}).get(object_id)
        if obj is None or part_id not in obj.get("parts", {}):
            raise ValueError("Pose part not found")
        annotation = {"visible": bool(visible)}
        if visible:
            if x is None or y is None:
                raise ValueError("Visible pose annotations require coordinates")
            annotation.update({"x": float(x), "y": float(y)})
        config["videos"][vid].setdefault("pose_annotations", {}).setdefault(
            object_id, {},
        ).setdefault(part_id, {})[str(frame_idx)] = annotation
        self._save_config(pid, config)
        return annotation

    def delete_pose_annotation(
        self,
        pid: str,
        vid: str,
        object_id: str,
        part_id: str,
        frame_idx: int,
    ) -> bool:
        config = self.get_project(pid)
        if config is None or vid not in config.get("videos", {}):
            raise ValueError("Video not found")
        obj = config["videos"][vid].setdefault("pose_objects", {}).get(object_id)
        if obj is None or part_id not in obj.get("parts", {}):
            raise ValueError("Pose part not found")
        frames = config["videos"][vid].setdefault("pose_annotations", {}).setdefault(
            object_id, {},
        ).setdefault(part_id, {})
        removed = frames.pop(str(frame_idx), None) is not None
        if removed:
            self._save_config(pid, config)
        return removed

    def clear_pose_annotations_for_frame(self, pid: str, vid: str, frame_idx: int) -> int:
        config = self.get_project(pid)
        if config is None or vid not in config.get("videos", {}):
            raise ValueError("Video not found")
        removed = 0
        for parts in config["videos"][vid].setdefault("pose_annotations", {}).values():
            for frames in parts.values():
                if frames.pop(str(frame_idx), None) is not None:
                    removed += 1
        if removed:
            self._save_config(pid, config)
        return removed

    def get_video(self, pid: str, vid: str) -> Optional[dict]:
        config = self._load_config(pid)
        if config is None:
            return None
        vm = config["videos"].get(vid)
        if vm is None:
            return None
        vm = dict(vm)
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
        if config.get("registration"):
            config["registration"].get("videos", {}).pop(vid, None)
            remaining = list(config["registration"].get("videos", {}).values())
            config["registration"]["status"] = (
                "complete"
                if remaining and all(bool(v.get("registered")) for v in remaining)
                else "labeling"
            )
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

    def clear_prompts_in_range(
        self, pid: str, vid: str,
        from_frame: Optional[int] = None,
        to_frame: Optional[int] = None,
    ) -> bool:
        """Remove point prompts for all objects on frames in [from_frame, to_frame]
        (both inclusive, None = unbounded). Object entries left empty are dropped.
        Returns True if anything changed."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        prompts = config["videos"][vid].get("point_prompts", {})
        lo = from_frame if from_frame is not None else -(10 ** 18)
        hi = to_frame if to_frame is not None else 10 ** 18
        changed = False
        for obj_id in list(prompts.keys()):
            frame_map = prompts[obj_id]
            for frame_key in list(frame_map.keys()):
                try:
                    fi = int(frame_key)
                except (TypeError, ValueError):
                    continue
                if lo <= fi <= hi:
                    del frame_map[frame_key]
                    changed = True
            if not frame_map:
                del prompts[obj_id]
        if changed:
            self._save_config(pid, config)
        return changed

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
        st = path.stat()
        with self._progress_cache_lock:
            cached = self._progress_cache.get(path)
            if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                return list(cached[2])
        frames: set[int] = set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    frames.add(int(line))
                except ValueError:
                    pass
        out = sorted(frames)
        st_after = path.stat()
        with self._progress_cache_lock:
            self._progress_cache[path] = (st_after.st_mtime_ns, st_after.st_size, out)
        return list(out)

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
        with self._progress_cache_lock:
            self._progress_cache.pop(path, None)

    def clear_propagated_frames(self, pid: str, vid: str):
        """Delete the progress file (used during video reset)."""
        try:
            path = self._propagation_progress_path(pid, vid)
        except ValueError:
            return
        if path.exists():
            path.unlink()
        with self._progress_cache_lock:
            self._progress_cache.pop(path, None)

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
        with self._progress_cache_lock:
            self._progress_cache.pop(path, None)
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

    def _load_config(self, pid: str) -> Optional[dict]:
        d = self._find_project_dir(pid)
        if d is None:
            return None
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            return None

        st = cfg_path.stat()
        mtime_ns = st.st_mtime_ns
        size = st.st_size
        with self._config_cache_lock:
            cached = self._config_cache.get(pid)
            if (
                cached is not None
                and cached[0] == cfg_path
                and cached[1] == mtime_ns
                and cached[2] == size
            ):
                return cached[3]

        config = self._normalize_project_config(json.loads(cfg_path.read_text()))
        with self._config_cache_lock:
            self._config_cache[pid] = (cfg_path, mtime_ns, size, config)
        return config

    def _save_config(self, pid: str, config: dict, project_dir: Optional[Path] = None):
        cfg_path = (project_dir or self._project_dir(pid)) / "config.json"
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
                st = cfg_path.stat()
                normalized = self._normalize_project_config(config)
                with self._config_cache_lock:
                    self._config_cache[pid] = (
                        cfg_path,
                        st.st_mtime_ns,
                        st.st_size,
                        normalized,
                    )
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

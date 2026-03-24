"""
Project and video metadata management.
Projects are stored in ~/.sam3_zero_projects/<project-id>-<name>/config.json
"""

import re
import uuid
import json
import shutil
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

    # ─── Projects ────────────────────────────────────────────────────────────

    def _slugify(self, name: str) -> str:
        """Convert a project name to a filesystem-safe slug."""
        slug = re.sub(r"[^\w\-.]", "_", name).strip("_.-")
        return slug[:64]

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
        video_dir = self._project_dir(pid) / "videos" / vid
        video_dir.mkdir(parents=True)
        (video_dir / "frames").mkdir()
        (video_dir / "masks").mkdir()
        (video_dir / "masks_raw").mkdir()
        (video_dir / "bboxes").mkdir()
        (video_dir / "corrections").mkdir()
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
            "instance_groups": {},
            "sam3_session_id": None,
            "propagated_frames": [],
            "propagation_complete": False,
            "frames_extracted": False,
            "all_frames_extracted": False,
            "preview_indices": [],
            # Tracking algorithm parameters (can be adjusted in real-time)
            "tracking_params": self.default_tracking_params(),
        }
        config["videos"][vid] = video_meta
        self._save_config(pid, config)
        return video_meta

    @staticmethod
    def default_tracking_params() -> dict:
        """Default tracking algorithm parameters."""
        return {
            "min_iou_threshold": 0.15,        # Minimum IoU with previous frame
            "max_area_ratio": 5.0,            # Max area change ratio
            "max_centroid_jump": 0.25,        # Max centroid distance (0-1)
            "consecutive_reject_limit": 5,    # After N rejects, accept anyway
            "anomaly_threshold": 2.5,         # Z-score for anomaly detection
            "confusion_threshold": 0.4,       # Score to open confusion window
            "text_similarity_threshold": 0.3, # Min text similarity for pair protection
            "propagation_mode": "temporal_tracking",  # "temporal_tracking" or "per_frame"
        }

    def get_tracking_params(self, pid: str, vid: str) -> dict:
        """Get tracking params for a video, with defaults for missing keys."""
        video = self.get_video(pid, vid)
        if video is None:
            return self.default_tracking_params()
        params = video.get("tracking_params", {})
        # Merge with defaults for any missing keys
        defaults = self.default_tracking_params()
        return {**defaults, **params}

    def update_tracking_params(self, pid: str, vid: str, updates: dict) -> dict:
        """Update tracking params. Returns the full updated params dict."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        current = config["videos"][vid].get("tracking_params", self.default_tracking_params())
        current.update(updates)
        config["videos"][vid]["tracking_params"] = current
        self._save_config(pid, config)
        return current

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
        video_dir = self._project_dir(pid) / "videos" / vid
        if video_dir.exists():
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

    def register_instance_object(self, pid: str, vid: str, base_obj_id: str, 
                                  instance_key: str, instance_num: int) -> dict:
        """
        Register an instance slot as a visible object (e.g., "1_1" from base object "1").
        
        This creates a new object entry with a color variation of the parent object,
        so that each instance is visually distinguishable and can be tracked separately.
        """
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        
        base_obj = config["videos"][vid]["objects"].get(str(base_obj_id))
        if base_obj is None:
            raise ValueError(f"Base object {base_obj_id} not found")
        
        # Don't re-register if already exists
        if instance_key in config["videos"][vid]["objects"]:
            return config["videos"][vid]["objects"][instance_key]
        
        # Create color variation
        base_color = base_obj.get("color", OBJECT_COLORS[0])
        # Shift hue/lightness for each instance
        new_color = self._vary_color(base_color, instance_num)
        
        instance_obj = {
            "id": instance_key,
            "name": f"{base_obj.get('name', base_obj_id)}_{instance_num}",
            "color": new_color,
            "description": base_obj.get("description", ""),
            "min_instances": 1,
            "max_instances": 1,
            "_parent_obj": str(base_obj_id),  # Track relationship
        }
        config["videos"][vid]["objects"][instance_key] = instance_obj
        
        # Add to instance_groups
        groups = config["videos"][vid].setdefault("instance_groups", {})
        base_group = groups.setdefault(str(base_obj_id), [int(base_obj_id)])
        if instance_key not in [str(x) for x in base_group]:
            base_group.append(instance_key)
        
        self._save_config(pid, config)
        return instance_obj

    def _vary_color(self, hex_color: str, shift: int) -> str:
        """Generate a color variation by shifting hue/lightness."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        # Rotate and add variance
        if shift == 1:
            r, g, b = g, b, r
        elif shift == 2:
            r, g, b = b, r, g
        # Add some brightness variance
        factor = 1.0 + (shift % 3) * 0.15
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def remove_object(self, pid: str, vid: str, obj_id: str):
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        config["videos"][vid]["objects"].pop(obj_id, None)
        config["videos"][vid]["point_prompts"].pop(obj_id, None)
        config["videos"][vid].get("instance_groups", {}).pop(obj_id, None)
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

    def allocate_instance_slot(self, pid: str, vid: str, ui_obj_id: str) -> Optional[int]:
        """
        Allocate and return a new SAM obj_id slot for a multi-instance object.
        Returns None if the object has already reached max_instances.
        SAM obj_ids are computed as (ui_obj_id - 1) * MAX_INST_SLOTS + slot_idx + 1
        but for backwards compat: first slot = int(ui_obj_id).
        """
        MAX_INST_SLOTS = 10
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            raise ValueError(f"Video {vid} not found")
        obj = config["videos"][vid]["objects"].get(ui_obj_id)
        if obj is None:
            raise ValueError(f"Object {ui_obj_id} not found")
        max_inst = obj.get("max_instances", 1)
        groups = config["videos"][vid].setdefault("instance_groups", {})
        current_slots = groups.get(ui_obj_id, [])
        if len(current_slots) >= max_inst:
            return None
        # Compute next slot: base = (int(ui_obj_id)-1)*MAX_INST_SLOTS
        # slot 0 = int(ui_obj_id) for backwards compat; subsequent = base + slot_idx + 1
        ui_id_int = int(ui_obj_id)
        base = (ui_id_int - 1) * MAX_INST_SLOTS
        if not current_slots:
            new_slot = ui_id_int  # first slot stays as the natural id
        else:
            # Find first unused slot index starting from 1
            used = set(current_slots)
            slot_idx = 1
            while (base + slot_idx) in used:
                slot_idx += 1
            new_slot = base + slot_idx
        current_slots.append(new_slot)
        groups[ui_obj_id] = current_slots
        self._save_config(pid, config)
        return new_slot

    def get_instance_groups(self, pid: str, vid: str) -> dict:
        """Return {ui_obj_id: [sam_obj_id, ...]} mapping."""
        config = self.get_project(pid)
        if config is None or vid not in config["videos"]:
            return {}
        return config["videos"][vid].get("instance_groups", {})

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
        return self._project_dir(pid) / "videos" / vid

    def frames_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "frames"

    def annotated_frames_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "annotated_frames"

    def masks_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "masks"

    def seed_masks_dir(self, pid: str, vid: str) -> Path:
        """
        Persistent mask storage for user-annotated keyframes.
        Files here survive "clear all masks" operations and serve as
        display fallbacks and propagation anchors.
        """
        d = self.video_dir(pid, vid) / "seed_masks"
        d.mkdir(exist_ok=True)
        return d

    def masks_raw_dir(self, pid: str, vid: str) -> Path:
        d = self.video_dir(pid, vid) / "masks_raw"
        d.mkdir(exist_ok=True)
        return d

    def bboxes_dir(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "bboxes"

    def corrections_dir(self, pid: str, vid: str) -> Path:
        d = self.video_dir(pid, vid) / "corrections"
        d.mkdir(exist_ok=True)
        return d

    def uncertainty_path(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "uncertainty.json"

    def overlaps_path(self, pid: str, vid: str) -> Path:
        return self.video_dir(pid, vid) / "overlaps.json"

    # ─── Uncertainty & Corrections ───────────────────────────────────────────

    def save_uncertainty(self, pid: str, vid: str, data: dict):
        self.uncertainty_path(pid, vid).write_text(json.dumps(data, indent=2))

    def load_uncertainty(self, pid: str, vid: str) -> dict:
        p = self.uncertainty_path(pid, vid)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def save_overlaps(self, pid: str, vid: str, data: dict):
        self.overlaps_path(pid, vid).write_text(json.dumps(data, indent=2))

    def load_overlaps(self, pid: str, vid: str) -> dict:
        p = self.overlaps_path(pid, vid)
        if not p.exists():
            return {"windows": []}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"windows": []}

    def save_correction(self, pid: str, vid: str, record: dict):
        path = self.corrections_dir(pid, vid) / f"{record['id']}.json"
        path.write_text(json.dumps(record, indent=2))

    def load_corrections(self, pid: str, vid: str) -> list[dict]:
        corrections = []
        d = self.corrections_dir(pid, vid)
        for f in sorted(d.glob("*.json")):
            try:
                corrections.append(json.loads(f.read_text()))
            except Exception:
                pass
        return sorted(corrections, key=lambda c: c.get("window_start", 0))

    def load_correction(self, pid: str, vid: str, correction_id: str) -> Optional[dict]:
        path = self.corrections_dir(pid, vid) / f"{correction_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def update_correction(self, pid: str, vid: str, correction_id: str, updates: dict):
        record = self.load_correction(pid, vid, correction_id)
        if record is None:
            raise ValueError(f"Correction {correction_id} not found")
        record.update(updates)
        self.save_correction(pid, vid, record)
        return record

    # ─── Internal ────────────────────────────────────────────────────────────

    def _save_config(self, pid: str, config: dict):
        cfg_path = self._project_dir(pid) / "config.json"
        cfg_path.write_text(json.dumps(config, indent=2))

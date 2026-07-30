from __future__ import annotations

from pathlib import Path

from project_manager import ProjectManager


def test_pose_hierarchy_and_query_annotation_persist(tmp_path: Path) -> None:
    manager = ProjectManager()
    manager.set_projects_root(str(tmp_path))
    project = manager.create_project("pose test", tracking_mode="pose_tracking")
    video = manager.add_video(
        project["id"], "mouse.mp4", "/tmp/mouse.mp4", 100, 30.0, 640, 480,
    )
    obj = manager.add_pose_object(project["id"], video["id"], "mouse", "#5B8DD9")
    part = manager.add_pose_part(
        project["id"], video["id"], obj["id"], "snout", "#E8A445",
    )
    manager.set_pose_annotation(
        project["id"], video["id"], obj["id"], part["id"], 17, 0.25, 0.75,
    )

    saved = manager.get_video(project["id"], video["id"])

    assert project["tracking_mode"] == "pose_tracking"
    assert saved["pose_objects"][obj["id"]]["parts"][part["id"]]["name"] == "snout"
    assert saved["pose_annotations"][obj["id"]][part["id"]]["17"] == {
        "x": 0.25,
        "y": 0.75,
    }

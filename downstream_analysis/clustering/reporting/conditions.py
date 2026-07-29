"""Resolve *what condition each video belongs to* for behavior quantification.

A behavior report is only meaningful once every video is labelled with its
experimental condition (housing, role, group, treatment, ...). There are two
ways to get those labels:

  1. FROM FILENAMES + REGISTRY. The pipeline already parses names of the form
     ``{camera_view}_video_{session}_{YYYYMMDD}_{HHMMSS}.mp4`` and looks the
     mouse up in ``experiment_registry.csv``. When that works, we're done.

  2. FROM THE USER. When filenames don't encode the condition (new dataset,
     ad-hoc naming, missing registry rows), the agent must ask the user for a
     description -- either free text ("R1-R4 are saline, R5-R8 are oxycodone")
     or a CSV. This module writes a template CSV keyed by the project's actual
     videos and loads a filled-in one back.

``probe_conditions`` tells the agent which case it is in, per video, so it knows
whether it needs to ask.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_config(project_dir: Path) -> dict:
    cfg = Path(project_dir) / "config.json"
    return json.loads(cfg.read_text()) if cfg.is_file() else {}


def _videos(project_dir: Path) -> list[tuple[str, str]]:
    """Return [(video_id, video_name), ...] from config.json."""
    config = _load_config(project_dir)
    out = []
    for vid, v in config.get("videos", {}).items():
        out.append((vid, str(v.get("name") or vid)))
    return sorted(out, key=lambda t: t[1])


def probe_conditions(project_dir: str | Path) -> dict:
    """Check whether each video's condition is derivable from its filename via
    the experiment registry.

    Returns a dict:
      {
        "resolvable": [ {video_id, video_name, camera_view, session}, ... ],
        "unresolved": [ {video_id, video_name, reason}, ... ],
        "all_resolved": bool,
      }

    Never raises on a missing registry -- it just reports everything as
    unresolved, which is the signal for the agent to ask the user.
    """
    project_dir = Path(project_dir)
    try:
        from downstream_analysis.clustering import experiment_registry as reg
    except Exception:
        reg = None

    resolvable, unresolved = [], []
    for vid, name in _videos(project_dir):
        parsed = None
        if reg is not None:
            try:
                parsed = reg.parse_video_name(name)
            except Exception:
                parsed = None
        if parsed:
            camera_view, session = parsed
            resolvable.append({"video_id": vid, "video_name": name,
                               "camera_view": camera_view, "session": session})
        else:
            unresolved.append({"video_id": vid, "video_name": name,
                               "reason": "filename does not match "
                                         "{camera}_video_{session}_{date}_{time}.mp4"})
    return {
        "resolvable": resolvable,
        "unresolved": unresolved,
        "all_resolved": len(unresolved) == 0 and len(resolvable) > 0,
    }


CONDITION_FIELDS = ["video_id", "video_name", "condition", "group", "role",
                    "housing", "treatment", "notes"]


def write_conditions_template(project_dir: str | Path, out_csv: str | Path) -> str:
    """Write a CSV with one row per video and blank condition columns for the
    user (or agent, from a user description) to fill in.
    """
    project_dir = Path(project_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CONDITION_FIELDS)
        w.writeheader()
        for vid, name in _videos(project_dir):
            w.writerow({"video_id": vid, "video_name": name})
    return str(out_csv)


def load_conditions_csv(csv_path: str | Path) -> dict:
    """Load a filled conditions CSV into {video_id: {field: value}} and
    {video_name: {...}} (both keys point at the same row for convenient lookup).

    Blank cells are dropped. A row needs at least one non-empty condition field
    (condition/group/role/housing/treatment) to count.
    """
    csv_path = Path(csv_path)
    by_id, by_name = {}, {}
    label_fields = {"condition", "group", "role", "housing", "treatment"}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            clean = {k: (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items() if v and str(v).strip()}
            if not (label_fields & set(clean)):
                continue
            if clean.get("video_id"):
                by_id[clean["video_id"]] = clean
            if clean.get("video_name"):
                by_name[clean["video_name"]] = clean
    return {"by_id": by_id, "by_name": by_name,
            "n_labelled": len(by_id or by_name)}


def summarize_conditions(loaded: dict) -> dict:
    """Count videos per condition/group for a quick sanity check before running
    comparisons. Returns {field: {value: count}}.
    """
    rows = list((loaded.get("by_id") or loaded.get("by_name") or {}).values())
    out: dict = {}
    for field in ("condition", "group", "role", "housing", "treatment"):
        counts: dict = {}
        for r in rows:
            if field in r:
                counts[r[field]] = counts.get(r[field], 0) + 1
        if counts:
            out[field] = counts
    return out


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Probe or template video->condition mapping for a project.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="Report which videos resolve from filenames")
    p_probe.add_argument("--project-dir", required=True)

    p_tmpl = sub.add_parser("template", help="Write a blank conditions CSV to fill in")
    p_tmpl.add_argument("--project-dir", required=True)
    p_tmpl.add_argument("--out", required=True)

    p_sum = sub.add_parser("summarize", help="Count videos per condition in a filled CSV")
    p_sum.add_argument("--csv", required=True)

    args = ap.parse_args()
    if args.cmd == "probe":
        r = probe_conditions(args.project_dir)
        print(f"resolvable from filenames: {len(r['resolvable'])}")
        print(f"unresolved:                {len(r['unresolved'])}")
        print(f"all_resolved:              {r['all_resolved']}")
        for u in r["unresolved"][:20]:
            print(f"  - {u['video_name']}  ({u['reason']})")
        if not r["all_resolved"]:
            print("\nNext: ask the user for conditions, then use "
                  "'template' to emit a CSV keyed by these videos.")
    elif args.cmd == "template":
        out = write_conditions_template(args.project_dir, args.out)
        print(f"Template written: {out}")
        print("Fill condition/group/role/housing/treatment columns (from the "
              "user's description), then load with load_conditions_csv().")
    elif args.cmd == "summarize":
        summary = summarize_conditions(load_conditions_csv(args.csv))
        import json as _json
        print(_json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()

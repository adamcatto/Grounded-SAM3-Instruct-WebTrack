"""Unit tests for agentic prompting helpers (no GPU / LLM required)."""

import os
import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import agent


class TestPlanFrames(unittest.TestCase):
    def test_every_thousandth_includes_last(self):
        frames = agent.plan_sample_frames(0, 3500, 1000, include_last=True)
        self.assertEqual(frames, [0, 1000, 2000, 3000, 3499])

    def test_interval_and_start(self):
        frames = agent.plan_sample_frames(50, 250, 100, include_last=False)
        self.assertEqual(frames, [50, 150])

    def test_empty_video(self):
        self.assertEqual(agent.plan_sample_frames(0, 0, 1000), [])

    def test_nearby_skips_current_and_clamps(self):
        near = agent.nearby_frames(5, 40, offset=20)
        self.assertNotIn(5, near)
        self.assertIn(25, near)
        self.assertIn(0, near)
        self.assertTrue(all(0 <= f < 40 for f in near))


class TestMaskQuality(unittest.TestCase):
    def test_empty_mask(self):
        q = agent.evaluate_mask_quality(np.zeros((64, 64), dtype=np.uint8))
        self.assertFalse(q["ok"])
        self.assertEqual(q["reason"], "empty_mask")
        self.assertTrue(q["retry_nearby"])

    def test_compact_blob_is_ok(self):
        m = np.zeros((200, 200), dtype=np.uint8)
        m[80:140, 70:150] = 1
        q = agent.evaluate_mask_quality(m, score=0.9)
        self.assertTrue(q["ok"])
        self.assertGreaterEqual(q["confidence"], agent.CONFIDENCE_RETRY_THRESHOLD)
        self.assertFalse(q["retry_nearby"])

    def test_full_frame_is_suspicious(self):
        m = np.ones((50, 50), dtype=np.uint8)
        q = agent.evaluate_mask_quality(m, score=0.99)
        self.assertIn("mask_covers_most_of_frame", q["reason"])


class TestProjectOverview(unittest.TestCase):
    def test_overview_lists_videos_and_current(self):
        project = {
            "id": "p1",
            "name": "Mice",
            "tracking_mode": "segmentation_tracking",
            "videos": {
                "v1": {
                    "name": "cage_a.mp4",
                    "num_frames": 2000,
                    "fps": 30,
                    "width": 640,
                    "height": 480,
                    "objects": {"1": {"name": "BackShave", "description": "shaved mouse", "color": "#5B8DD9"}},
                    "point_prompts": {"1": {"0": {"points": [[0.5, 0.5]], "labels": [1]}}},
                    "annotated_anchors": [0],
                    "propagation_complete": False,
                    "propagated_frames": [],
                },
                "v2": {
                    "name": "cage_b.mp4",
                    "num_frames": 100,
                    "objects": {},
                    "point_prompts": {},
                },
            },
        }
        ov = agent.project_overview(project, "v1", 12)
        self.assertEqual(ov["video_count"], 2)
        self.assertEqual(ov["current_video_id"], "v1")
        self.assertEqual(ov["current_frame"], 12)
        names = {row["name"] for row in ov["videos"]}
        self.assertEqual(names, {"cage_a.mp4", "cage_b.mp4"})
        current = next(r for r in ov["videos"] if r["is_current"])
        self.assertEqual(current["object_count"], 1)
        self.assertEqual(current["prompted_frame_count"], 1)
        other = next(r for r in ov["videos"] if not r["is_current"])
        self.assertNotIn("prompted_frame_count", other)
        self.assertIn("name", other)

    def test_detections_from_numpy_outputs(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        dets = agent._detections_from_sam_outputs({
            "out_obj_ids": np.array([3]),
            "out_binary_masks": np.array([mask]),
            "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4]]),
            "out_probs": np.array([0.9]),
        })
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["score"], 0.9)
        self.assertEqual(dets[0]["mask"].shape, (8, 8))

    def test_detections_empty_numpy_is_not_ambiguous(self):
        self.assertEqual(agent._detections_from_sam_outputs({
            "out_obj_ids": np.array([]),
            "out_binary_masks": np.array([]),
        }), [])


class TestToolSurface(unittest.TestCase):
    def test_required_tools_exist(self):
        names = {t["name"] for t in agent.TOOL_SCHEMAS}
        for required in (
            "think",
            "get_project_overview",
            "get_video_details",
            "select_video",
            "goto_frame",
            "inspect_frame",
            "plan_frames",
            "create_object",
            "text_segment",
            "add_point_prompt",
            "evaluate_segmentation",
            "start_propagation",
        ):
            self.assertIn(required, names)

    def test_system_prompt_covers_workflow(self):
        p = agent.SYSTEM_PROMPT.lower()
        for needle in ("every nth", "nearby", "text", "point", "propagation", "inspect"):
            self.assertIn(needle, p)

    def test_initial_messages_include_context(self):
        msgs = agent.build_initial_messages(
            "segment both mice",
            {"project_name": "Demo", "video_count": 2},
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("Demo", msgs[-1]["content"])
        self.assertIn("segment both mice", msgs[-1]["content"])


class TestLLMConfig(unittest.TestCase):
    def _clear_llm_env(self, extra=None):
        env = {
            "AGENT_LLM_PROVIDER": "",
            "AGENT_LLM_MODEL": "",
            "AGENT_LLM_BASE_URL": "",
            "AGENT_LLM_API_KEY": "",
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "OPENAI_BASE_URL": "",
            "AGENT_LLM_AUTODETECT": "1",
        }
        if extra:
            env.update(extra)
        ctx = mock.patch.dict(os.environ, env, clear=False)

        class _Cleared:
            def __enter__(self):
                self._cm = ctx
                inner = self._cm.__enter__()
                for k in ("AGENT_LLM_THINKING", "AGENT_LLM_PROFILE"):
                    if extra is None or k not in extra:
                        os.environ.pop(k, None)
                return inner

            def __exit__(self, *args):
                return self._cm.__exit__(*args)

        return _Cleared()

    def test_missing_local_and_cloud(self):
        with self._clear_llm_env({"AGENT_LLM_PROVIDER": "openai", "AGENT_LLM_MODEL": "gpt-4o"}):
            for k in ("AGENT_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AGENT_LLM_BASE_URL"):
                os.environ.pop(k, None)
            with mock.patch.object(agent, "discover_local_llm", return_value=None):
                cfg = agent.load_llm_config(probe=True)
        self.assertFalse(cfg.configured)
        self.assertIn("Ollama", cfg.missing_reason)

    def test_openai_key(self):
        with self._clear_llm_env({
            "AGENT_LLM_API_KEY": "sk-test",
            "AGENT_LLM_PROVIDER": "openai",
            "AGENT_LLM_MODEL": "gpt-4o",
        }):
            with mock.patch.object(agent, "discover_local_llm", return_value=None):
                cfg = agent.load_llm_config(probe=False)
        self.assertTrue(cfg.configured)
        self.assertEqual(cfg.provider, "openai")
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertFalse(cfg.local)

    def test_vllm_uses_dummy_key_like_sam3(self):
        found = {
            "provider": "vllm",
            "base_url": "http://127.0.0.1:8001/v1",
            "models": ["Qwen/Qwen3-VL-8B-Thinking"],
            "reachable": True,
        }
        with self._clear_llm_env({"AGENT_LLM_PROVIDER": "vllm"}):
            for k in ("AGENT_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                os.environ.pop(k, None)
            with mock.patch.object(agent, "discover_local_llm", return_value=found):
                with mock.patch.object(agent, "probe_openai_compatible", return_value=found["models"]):
                    cfg = agent.load_llm_config()
        self.assertTrue(cfg.configured)
        self.assertTrue(cfg.local)
        self.assertEqual(cfg.provider, "vllm")
        self.assertEqual(cfg.api_key, "DUMMY_API_KEY")
        self.assertEqual(cfg.base_url, "http://127.0.0.1:8001/v1")
        self.assertEqual(cfg.model, "Qwen/Qwen3-VL-8B-Thinking")

    def test_vllm_profile_a100_defaults_to_32b_instruct(self):
        with self._clear_llm_env({"AGENT_LLM_PROVIDER": "vllm", "AGENT_LLM_PROFILE": "a100"}):
            for k in ("AGENT_LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                os.environ.pop(k, None)
            with mock.patch.object(agent, "discover_local_llm", return_value=None):
                cfg = agent.load_llm_config(probe=False)
        self.assertEqual(cfg.model, "Qwen/Qwen3-VL-32B-Instruct")
        self.assertEqual(agent.default_vllm_model("a100"), "Qwen/Qwen3-VL-32B-Instruct")
        self.assertEqual(
            agent.default_vllm_model("a100", thinking=True),
            "Qwen/Qwen3-VL-32B-Thinking",
        )

    def test_vllm_profile_h100x4_defaults_to_72b(self):
        with self._clear_llm_env({"AGENT_LLM_PROVIDER": "vllm", "AGENT_LLM_PROFILE": "h100x4"}):
            with mock.patch.object(agent, "discover_local_llm", return_value=None):
                cfg = agent.load_llm_config(probe=False)
        self.assertEqual(cfg.model, "Qwen/Qwen2.5-VL-72B-Instruct")
        self.assertEqual(agent.normalize_llm_profile("cluster"), "h100x4")
        self.assertEqual(agent.normalize_llm_profile("workstation"), "a100")

    def test_vllm_profile_shared_stays_8b(self):
        self.assertEqual(agent.default_vllm_model("a100-shared"), "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(agent.default_vllm_model("demo"), "Qwen/Qwen3-VL-8B-Thinking")

    def test_ollama_autodetect(self):
        found = {
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "models": ["llama3.2:latest", "qwen2.5vl:7b"],
            "reachable": True,
        }
        with self._clear_llm_env():
            with mock.patch.object(agent, "discover_local_llm", return_value=found):
                cfg = agent.load_llm_config()
        self.assertTrue(cfg.configured)
        self.assertEqual(cfg.provider, "ollama")
        self.assertEqual(cfg.model, "qwen2.5vl:7b")

    def test_prefer_vision_model(self):
        self.assertEqual(
            agent._prefer_vision_model(["llama3.2", "qwen2.5vl:7b", "mistral"]),
            "qwen2.5vl:7b",
        )

    def test_normalize_ollama_base(self):
        self.assertEqual(
            agent._normalize_openai_base("http://127.0.0.1:11434", "ollama"),
            "http://127.0.0.1:11434/v1",
        )


class TestContextPruning(unittest.TestCase):
    def test_keeps_only_last_two_inspect_images(self):
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(4):
            msgs.append({
                "role": "user",
                "content": f"Visual frame {i}",
                "images": [{"jpeg_b64": f"fake{i}", "frame_idx": i}],
            })
        dropped = agent.prune_stale_inspect_images(msgs, keep=2)
        self.assertEqual(dropped, 2)
        with_images = [m for m in msgs if m.get("images")]
        self.assertEqual(len(with_images), 2)
        self.assertEqual(with_images[0]["images"][0]["frame_idx"], 2)
        self.assertEqual(with_images[1]["images"][0]["frame_idx"], 3)
        self.assertIn("dropped", msgs[1]["content"])

    def test_inspect_jpegs_in_messages_lists_kept_frames(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "Visual",
                "images": [{"jpeg_b64": "abc", "frame_idx": 160, "video_id": "vid1"}],
            },
        ]
        kept = agent.inspect_jpegs_in_messages(msgs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["frame_idx"], 160)
        self.assertEqual(kept[0]["video_id"], "vid1")

    def test_write_agent_context_dump_saves_markdown_and_jpeg(self):
        dump = Path(tempfile.mkdtemp(prefix="sam3wt_agent_dump_"))
        jpeg = base64.b64encode(b"\xff\xd8fakejpeg").decode()
        cfg = agent.LLMConfig(
            provider="vllm",
            api_key="DUMMY_API_KEY",
            model="Qwen/Qwen3-VL-8B-Instruct",
            base_url="http://127.0.0.1:8001/v1",
            configured=True,
            local=True,
        )
        messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Visual",
                "images": [{"jpeg_b64": jpeg, "frame_idx": 7, "video_id": "v1"}],
            },
        ]
        summary = agent.write_agent_context_dump(
            dump,
            system_prompt=agent.SYSTEM_PROMPT,
            overview={"project_name": "demo", "videos": []},
            user_text="inspect and segment two mice",
            messages=messages,
            events=[
                {"event": "tool_call", "data": {"name": "inspect_frame", "arguments": {"frame_idx": 7}, "step": 1}},
                {"event": "tool_result", "data": {"name": "inspect_frame", "ok": True, "result": {"frame_idx": 7}, "elapsed_ms": 12, "step": 1}},
            ],
            llm_turns=[{"step": 1, "inspect_jpegs_in_context": agent.inspect_jpegs_in_messages(messages)}],
            cfg=cfg,
            inspect_images=[{"jpeg_b64": jpeg, "frame_idx": 7, "video_id": "v1"}],
        )
        self.assertTrue((dump / "context.md").is_file())
        self.assertTrue((dump / "context.json").is_file())
        self.assertTrue(summary["inspect_jpegs_seen"])
        jpeg_path = dump / summary["inspect_jpegs_seen"][0]["file"]
        self.assertTrue(jpeg_path.is_file())
        md = (dump / "context.md").read_text()
        self.assertIn("System prompt", md)
        self.assertIn("Retrieved project JSON", md)
        self.assertIn("inspect_frame", md)
        self.assertIn("in final LLM context", md)
        last = agent.get_last_agent_dump()
        self.assertEqual(last["dump_dir"], str(dump))

    def test_agent_max_tokens_is_below_8k_window(self):
        self.assertLessEqual(agent.AGENT_MAX_TOKENS, 2048)
        self.assertEqual(agent.KEEP_INSPECT_IMAGES, 2)


class TestSyntheticVideoFixture(unittest.TestCase):
    """Write a tiny MP4 so import/extract can be tested against a real file."""

    def test_write_two_clip_project_dir(self):
        try:
            cv2 = __import__("cv2")
        except ImportError:
            self.skipTest("opencv-python not installed in this environment")
        root = Path(tempfile.mkdtemp(prefix="sam3wt_agent_vids_"))
        clips = []
        for name, color in (("cage_a.mp4", (40, 180, 40)), ("cage_b.mp4", (40, 40, 180))):
            path = root / name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            wri = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 120))
            self.assertTrue(wri.isOpened(), f"VideoWriter failed for {path}")
            for i in range(24):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                frame[:] = color
                cv2.circle(frame, (40 + i, 60), 12, (220, 220, 220), -1)
                cv2.circle(frame, (110, 50 + (i % 8)), 10, (20, 20, 20), -1)
                wri.write(frame)
            wri.release()
            self.assertGreater(path.stat().st_size, 500)
            clips.append(path)
        marker = root / "README.txt"
        marker.write_text(
            "Drop real behavior clips here (mp4). The agent test project imports every *.mp4 in this folder.\n"
        )
        self.assertEqual(len(clips), 2)
        print(f"SYNTHETIC_VIDEO_DIR={root}")


if __name__ == "__main__":
    unittest.main()

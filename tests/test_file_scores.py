import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import file_scores as f
import pipeline as p
import utils as u
from smoke_test import smoke_plan


def artifact(transform="reciprocal"):
    ru, text, image = u.feature_specs_from_registry(["qwen"], ["noterms"])
    text_spec = next(s for s in text if s.method_key == "radcliq").as_dict()
    text_spec["quality_transform"] = transform
    return {"selected_features": [ru[0].as_dict(), text_spec, image[0].as_dict()],
            "standardizer": {"means": [.3, .4, .5], "stds": [.1, .2, .3]},
            "weights": [.2, .3, .5],
            "calibration": {"kind": "logistic", "a": 1.7, "b": -.1, "mse": .01},
            "extraction_plan": p.resolved_plan({"translators": ["qwen"],
                "translation_kinds": ["noterms"], "methods": ["radcliq"],
                "runtime": {"gpu": 9, "device": "cpu"},
                "environments": {"default": [sys.executable]}})}


class FileScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [{"reference_text": "Эталон " + str(i), "candidate_text": "Кандидат",
                      "reference_image_path": "scan.png", "report_id": f"00{i}", "extra": {"ok": True}}
                     for i in range(2)]

    def write(self, name, data):
        path = self.root / name
        p.atomic_json(path, data)
        return path

    def export(self, fitted=None):
        return f.export_artifact(fitted or artifact(), self.root / "weights.json", self.root / "components.json")

    def test_fusion_matches_original_method_for_both_radcliq_transforms(self):
        for transform in ("reciprocal", "negate"):
            fit = artifact(transform)
            weights, _ = self.export(fit)
            self.assertEqual(set(weights), {"scaling", "fusion", "calibration"})
            values = [[.8, .6], [2., 4.] if transform == "reciprocal" else [-.2, 0.], [.7, -.1]]
            paths = [self.write(name + ".json", [{**row, "score": value} for row, value in zip(self.rows, vals)])
                     for name, vals in zip("ABC", values)]
            result = f.fuse_files(*paths, self.root / "weights.json", self.root / "D.json")
            frame = pd.DataFrame({s["feature_col"]: vals for s, vals in zip(fit["selected_features"], values)})
            np.testing.assert_allclose([r["score"] for r in result], u.score_with_config(frame, fit))
            self.assertEqual([{k: v for k, v in r.items() if k != "score"} for r in result], self.rows)
            # Fusion CLI works even with site-packages disabled: no NumPy/PyTorch/model loads.
            subprocess.run([sys.executable, "-S", str(p.ROOT / "merge_scores.py"), *map(str, paths),
                            "--weights", str(self.root / "weights.json"), "--output", str(self.root / "cli.json")],
                           check=True, capture_output=True, text=True)
            self.assertEqual(p.read_json(self.root / "cli.json"), result)

    def test_export_orders_by_component_and_removes_execution_locations(self):
        fit = artifact()
        expected, _ = self.export(fit)
        order = [2, 0, 1]
        fit["selected_features"] = [fit["selected_features"][i] for i in order]
        fit["weights"] = [fit["weights"][i] for i in order]
        fit["standardizer"] = {key: [vals[i] for i in order] for key, vals in fit["standardizer"].items()}
        weights, config = self.export(fit)
        self.assertEqual(weights, expected)
        self.assertNotIn("gpu", config["plan"]["runtime"])
        self.assertNotIn("environments", config["plan"])
        self.assertEqual(config["components"]["text"]["method_key"], "radcliq")

    def test_merge_rejects_changed_rows_counts_scores_and_weights(self):
        weights, _ = self.export()
        rows = [{**row, "score": 1.} for row in self.rows]
        a = self.write("A.json", rows)
        b = self.write("B.json", rows)
        c = self.write("C.json", rows)
        for bad in (rows[:1], list(reversed(rows)), [{**rows[0], "extra": False}, rows[1]],
                    [{**rows[0], "score": True}, rows[1]], [{**rows[0], "score": "1"}, rows[1]]):
            self.write("B.json", bad)
            with self.assertRaises(ValueError):
                f.fuse_files(a, b, c, self.root / "weights.json", self.root / "D.json")
        self.write("B.json", [{**row, "score": 0} for row in rows])
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            f.fuse_files(a, b, c, self.root / "weights.json", self.root / "D.json")
        for key, bad_value in (("stds", [1, 0, 1]), ("means", [1, 2]), ("transforms", ["bad"] * 3)):
            bad = copy.deepcopy(weights)
            bad["scaling"][key] = bad_value
            with self.assertRaises(ValueError):
                f.validate_weights(bad)
        for bad_value in ([.2, .3, .4], [-1, 1, 1], [1, True, 0]):
            bad = copy.deepcopy(weights)
            bad["fusion"]["weights"] = bad_value
            with self.assertRaises(ValueError):
                f.validate_weights(bad)
        self.assertFalse((self.root / "D.json").exists())

    def test_input_validation(self):
        for data in ([], {}, [None], [{**self.rows[0], "candidate_text": " "}],
                     [{**self.rows[0], "reference_image_path": None}]):
            with self.assertRaises(ValueError):
                f.read_records(self.write("invalid.json", data))
        path = self.root / "invalid.json"
        path.write_text(json.dumps([{**self.rows[0], "score": float("nan")}]))
        with self.assertRaises(ValueError):
            f.read_records(path, scored=True)

    def test_report_ids_must_be_nonempty_unique_strings_and_present_on_all_rows(self):
        for value in (None, "", "  ", 1, True, self.rows[1]["report_id"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                f.read_records(self.write("ids.json", [{**self.rows[0], "report_id": value}, self.rows[1]]))
        missing = {k: v for k, v in self.rows[0].items() if k != "report_id"}
        with self.assertRaises(ValueError):
            f.read_records(self.write("ids.json", [missing, self.rows[1]]))

    def test_legacy_records_without_ids_still_use_internal_row_numbers(self):
        _, config = self.export()
        source = self.write("legacy.json", [{k: v for k, v in row.items() if k != "report_id"}
                                            for row in self.rows])
        work = self.root / "legacy-work"

        def features(*args, **kwargs):
            base = p.read_csv(work / "base.csv")
            self.assertEqual(base.report_id.tolist(), ["0", "1"])
            return pd.DataFrame({config["components"]["russian"]["feature_col"]: [.8, .6]})

        with patch.object(p, "run_features", side_effect=features):
            result = f.score_component("russian", source, self.root / "components.json",
                                       output=self.root / "legacy-scores.json", work_dir=work)
        self.assertTrue(all("report_id" not in row for row in result))

    def test_separate_components_preserve_records_reuse_translation_and_match_fusion(self):
        _, config = self.export()
        config["plan"]["environments"] = {"default": [sys.executable]}
        self.write("components.json", config)
        source = self.write("reports.json", self.rows)
        (self.root / "scan.png").write_bytes(b"Fixture; neural workers are mocked")
        calls = []

        def fake_worker(command, **kwargs):
            request = p.read_json(command[-1])
            job, work = request["job"], Path(request["work_dir"])
            base = p.read_csv(work / "base.csv")
            self.assertEqual(base.report_id.tolist(), ["000", "001"])
            self.assertEqual(base.image_path.tolist(), [str(self.root / "scan.png")] * 2)
            calls.append(job["id"])
            frame = base[["report_id"]].copy()
            for col in p.expected_columns(job):
                frame[col] = "English fixture" if job["stage"] == "translate" or col.endswith(("analysis", "json")) else [.8, .6]
            output, _ = p.paths(work, job)
            frame.to_csv(output, index=False)
            p.atomic_json(output.with_suffix(".versions.json"), {"fixture": True})

        work = self.root / "work"
        with contextlib.redirect_stdout(io.StringIO()), patch.object(p.subprocess, "run", side_effect=fake_worker):
            for component, name in zip(f.COMPONENTS, "ABC"):
                result = f.score_component(component, source, self.root / "components.json",
                                           output=self.root / (name + ".json"), work_dir=work)
                self.assertEqual(len(result), 2)
                self.assertEqual(set(result[0]), set(self.rows[0]) | {"score"})
            self.assertEqual(calls, ["green", "translate.qwen.noterms", "text.radcliq.qwen.noterms", "image.qwen.noterms"])
            f.score_component("russian", source, self.root / "components.json", output=self.root / "A.json", work_dir=work)
            self.assertEqual(len(calls), 4)
        f.fuse_files(*(self.root / (name + ".json") for name in "ABC"), self.root / "weights.json", self.root / "D.json")
        self.assertTrue(all(0 <= row["score"] <= 1 for row in p.read_json(self.root / "D.json")))

    def test_default_environments_and_external_gpu_selection(self):
        plan = p.resolved_plan({})
        self.assertEqual(set(plan["environments"].values()), {"cxr-qwen", "cxr-text", "cxr-image"})
        self.assertNotIn("gpu", plan["runtime"])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            u.configure_worker_environment(device="cuda")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_smoke_covers_every_model_family_once(self):
        image = self.root / "scan.png"
        image.write_bytes(b"Fixture")
        plan, jobs = smoke_plan(image, self.root / "smoke", {"methods": ["radcliq"]})
        self.assertEqual(len(jobs), 10)
        self.assertEqual({j["translator"] for j in jobs if j["stage"] == "translate"},
                         {"qwen", "hy_mt", "translategemma"})
        self.assertEqual({m for j in jobs for m in j.get("methods", [])}, set(p.METHODS))
        self.assertTrue(any(j["stage"] == "green" for j in jobs))
        self.assertTrue(any(j["stage"] == "image" for j in jobs))


if __name__ == "__main__":
    unittest.main()

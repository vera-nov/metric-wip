import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as p
from calculate_russian_text import ManualGREENQwen
from tools.revalidate_green_hotfix_cache import migrate


def answer(insignificant="Нет клинически незначимых ошибок."):
    return ("[Клинически значимые ошибки]:\n" + "\n".join(
        f"({letter}) Категория: {1 if letter == 'а' else 0}. Описание"
        for letter in "абвгде") + "\n[Клинически незначимые ошибки]:\n" + insignificant +
        "\n[Совпадающие находки]:\n3. Находки")


class GreenHotfixTests(unittest.TestCase):
    def setUp(self):
        self.green = object.__new__(ManualGREENQwen)

    def test_zero_variants_and_formatting(self):
        for zero in ("Нет клинически незначимых ошибок.", "Клинически незначимые ошибки отсутствуют.",
                     "Незначимых ошибок не выявлено.", "Ошибок нет.", "Нет.", "0"):
            with self.subTest(zero=zero):
                parsed = self.green.parse(answer(zero))
                self.assertEqual(parsed.score, .75)
                self.assertEqual(parsed.sig_total, 1)
                self.assertEqual(parsed.insig_total, 0)
        numbered = "\n".join(f"{letter}) Категория: нет." for letter in "абвгде")
        text = answer(numbered).replace("[Клинически незначимые ошибки]:",
                                        "## **[Клинически незначимые ошибки:]**")
        self.assertEqual(self.green.parse(text).score, .75)
        direct = "\n".join(f"({letter}) 0." for letter in "абвгде")
        self.assertEqual(self.green.parse(answer(direct)).insig_total, 0)

    def test_unknown_counts_are_not_zero_and_duplicates_rejected(self):
        for bad in ("(а) Не знаю", "(а) Категория: неизвестно", "", "Нет значимых ошибок."):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.green.parse(answer(bad))
        for bad_count in ("-1", "1.5", "1,5"):
            with self.subTest(bad_count=bad_count), self.assertRaises(ValueError):
                self.green.parse(answer().replace("Категория: 1.", f"Категория: {bad_count}."))
        with self.assertRaises(ValueError):
            self.green.parse(answer() + "\n[Совпадающие находки]:\n4. Иные находки")
        with self.assertRaises(ValueError):
            self.green.parse(answer().replace("(б) Категория", "(а) Категория"))

    def test_failed_run_keeps_raw_answers_and_resume_only_generates_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "green.responses.jsonl"
            kwargs = dict(references=["r1", "r2"], candidates=["c1", "c2"],
                          row_ids=["001", "002"], response_log=journal,
                          cache_context={"model": "fixture"})
            with patch.object(self.green, "generate_batch", side_effect=[[answer()], ["bad"], ["bad retry"]]) as gen:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "002"):
                    self.green.score_pairs(**kwargs)
                self.assertEqual(gen.call_count, 3)
            entries = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
            self.assertEqual([e["report_id"] for e in entries], ["001", "002", "002"])
            self.assertEqual(entries[-1]["attempt"], 1)
            # Simulate an interrupted append; earlier complete entries survive.
            with journal.open("a") as stream:
                stream.write('{"partial":')
            with patch.object(self.green, "generate_batch", return_value=[answer()]) as gen:
                with contextlib.redirect_stdout(io.StringIO()), self.assertWarns(UserWarning):
                    result = self.green.score_pairs(**kwargs)
                self.assertEqual(gen.call_count, 1)
                self.assertIn("c2", gen.call_args.args[0][0])
                self.assertEqual(result.ru_GREEN_Qwen.tolist(), [.75, .75])
            with patch.object(self.green, "generate_batch", side_effect=AssertionError("must reuse")):
                with contextlib.redirect_stdout(io.StringIO()), self.assertWarns(UserWarning):
                    self.green.score_pairs(**kwargs)
            changed = dict(kwargs, cache_context={"model": "another"})
            with patch.object(self.green, "generate_batch", return_value=[answer()]) as gen:
                with contextlib.redirect_stdout(io.StringIO()), self.assertWarns(UserWarning):
                    self.green.score_pairs(**changed)
                self.assertEqual(gen.call_count, 2)

    def test_retry_keeps_score_and_records_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "green.responses.jsonl"
            with patch.object(self.green, "generate_batch", side_effect=[["bad"], [answer()]]):
                with contextlib.redirect_stdout(io.StringIO()):
                    out = self.green.score_pairs(["ref"], ["cand"], response_log=journal)
            self.assertEqual(out.ru_GREEN_Qwen.iloc[0], .75)
            entries = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
            self.assertEqual([e["attempt"] for e in entries], [0, 1])


class MigrationTests(unittest.TestCase):
    def test_revalidate_without_changing_csv_and_resume_skips_translations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            pd.DataFrame({"id": ["001"], "c": ["кандидат"], "r": ["эталон"],
                          "i": ["image.png"], "y": [1]}).to_csv(source, index=False)
            plan = p.resolved_plan({"input_csv": str(source), "image_root": str(root),
                "columns": {"row_id": "id", "candidate": "c", "reference": "r", "image": "i", "target": "y"},
                "translators": ["qwen"], "translation_kinds": ["terms"],
                "methods": ["cosinesim_biovilt"], "environments": {"default": [sys.executable]}})
            work = root / "work"
            base, _ = p.prepare(plan, work, True)
            old = {v.name: p.digest(v) for v in sorted(p.ROOT.glob("*.py"))}
            old["calculate_russian_text.py"] = "old-fixture-hash"
            snapshot = root / "snapshot.json"
            p.atomic_json(snapshot, old)
            job = p.stages(plan)[0]
            output, manifest = p.paths(work, job)
            pd.DataFrame({"report_id": ["001"], "generation_qwen_terms": ["candidate"],
                          "gt_qwen_terms": ["reference"]}).to_csv(output, index=False)
            with patch.object(p, "code_hash", return_value=p.fingerprint(old)):
                signature = p.job_signature(plan, job, work, base)
            p.atomic_json(manifest, {"signature": signature, "sha256": p.digest(output)})
            original = output.read_bytes()
            changed = copy.deepcopy(plan)
            changed["runtime"]["green_max_new_tokens"] += 1
            with self.assertRaisesRegex(ValueError, "Cannot verify"):
                migrate(changed, work, snapshot)
            self.assertEqual(p.read_json(manifest)["signature"], signature)
            # Corrupted CSVs fail verification before a manifest can be rewritten.
            output.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "Cannot verify"):
                migrate(plan, work, snapshot)
            output.write_bytes(original)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(migrate(plan, work, snapshot), 1)
                self.assertEqual(migrate(plan, work, snapshot), 0)
                with patch.object(p.subprocess, "run", side_effect=AssertionError("must reuse")):
                    p.run_features(plan, work, resume=True, only=[job["id"]])
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(p.read_json(manifest.with_name(manifest.name + ".pre-green-fix.bak"))["signature"], signature)
            # Changes outside GREEN cannot be blessed by this migration.
            old["pipeline.py"] = "unexpected-old-hash"
            p.atomic_json(snapshot, old)
            with self.assertRaisesRegex(ValueError, "ONLY"):
                migrate(plan, work, snapshot)


if __name__ == "__main__":
    unittest.main()

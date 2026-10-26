"""Run every model family on one short report and a real supplied CXR image."""
import argparse
from pathlib import Path

from pipeline import (METHODS, atomic_json, expected_columns, load_plan, paths, prepare,
                      read_csv, resolved_plan, run_features, stages)

REFERENCE = "Лёгкие без очаговых изменений. Плеврального выпота нет. Размеры сердца не увеличены."
CANDIDATE = "Очаговых изменений в лёгких нет. Выпот не определяется. Сердце обычных размеров."


def smoke_plan(image, work_dir, overrides=None):
    import pandas as pd

    image = Path(image).resolve()
    if not image.is_file():
        raise ValueError(f"Image does not exist: {image}")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work / "smoke.csv"
    pd.DataFrame([{"report_id": "000", "candidate": CANDIDATE,
                   "reference": REFERENCE, "image": str(image)}]).to_csv(source, index=False)
    plan = resolved_plan({**(overrides or {}), "input_csv": str(source),
                          "columns": {"row_id": "report_id", "candidate": "candidate",
                                      "reference": "reference", "image": "image"},
                          "translators": ["qwen", "hy_mt", "translategemma"],
                          "translation_kinds": ["noterms"], "methods": METHODS,
                          "runtime": {**(overrides or {}).get("runtime", {}),
                                      "green_batch_size": 1, "text_batch_size": 1,
                                      "image_batch_size": 1}})
    # Test each distinct scoring backend once; test all three translators.
    jobs = [j for j in stages(plan) if j["stage"] in {"translate", "green"}
            or j.get("translator") == "qwen"]
    return plan, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a real CXR image")
    parser.add_argument("--work-dir", default="work/smoke")
    parser.add_argument("--plan", help="Optional model paths / environment overrides")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    work = Path(args.work_dir).resolve()
    # A failed rerun must never leave an old PASS report behind.
    (work / "smoke_result.json").unlink(missing_ok=True)
    plan, jobs = smoke_plan(args.image, work, load_plan(args.plan) if args.plan else None)
    prepare(plan, work, require_target=False)
    run_features(plan, work, only=[j["id"] for j in jobs], resume=args.resume, dry_run=args.dry_run)
    if args.dry_run:
        return
    checks = []
    for job in jobs:
        frame = read_csv(paths(work, job)[0])
        checks.append({"stage": job["id"], "rows": len(frame),
                       "values": frame[expected_columns(job)].iloc[0].to_dict()})
    atomic_json(work / "smoke_result.json", {"status": "passed", "checks": checks})
    print(f"PASS: {len(checks)} stages; results in {work / 'smoke_result.json'}")


if __name__ == "__main__":
    main()

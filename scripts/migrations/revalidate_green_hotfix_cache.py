"""Revalidate completed non-GREEN stages after changing calculate_russian_text.py only.

Before applying the hotfix, save {filename: sha256} for ROOT.glob('*.py').
This script never modifies score/translation CSVs or bypasses data checks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pipeline as p


def migrate(plan, work_dir, snapshot_path):
    import numpy as np
    import pandas as pd

    work_dir = Path(work_dir).resolve()
    old_sources = p.read_json(snapshot_path)
    current_sources = {v.name: p.digest(v) for v in sorted(p.ROOT.glob("*.py"))}
    changed = {name for name in old_sources.keys() | current_sources.keys()
               if old_sources.get(name) != current_sources.get(name)}
    if changed != {"calculate_russian_text.py"}:
        raise ValueError("This migration requires ONLY calculate_russian_text.py to change; "
                         f"observed changes: {sorted(changed)}")
    old_code = p.fingerprint(old_sources)
    base = p.read_csv(work_dir / "base.csv")
    # Verify that rerunning build_method.py with this plan will prepare identical data.
    with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
        prepared, _ = p.prepare(plan, Path(temporary), require_target=True)
        prepared = p.read_csv(Path(temporary) / "base.csv")
    pd.testing.assert_frame_equal(base, prepared)

    updates = []
    for job in p.stages(plan):
        if job["stage"] == "green":
            continue
        output, manifest = p.paths(work_dir, job)
        if not output.exists() or not manifest.exists():
            continue
        images = p.image_fingerprints(base) if job["stage"] == "image" else None
        with patch.object(p, "code_hash", return_value=old_code):
            old_signature = p.job_signature(plan, job, work_dir, base, images)
        new_signature = p.job_signature(plan, job, work_dir, base, images)
        if p.valid_cache(output, manifest, new_signature):
            print(f"Already revalidated {job['id']}")
            continue
        if not p.valid_cache(output, manifest, old_signature):
            raise ValueError(f"Cannot verify original cache for {job['id']}. "
                             "Check plan, source snapshot, inputs and CSV checksum; nothing bypassed.")
        chunk = p.read_csv(output)
        p.validate_chunk(base, chunk, job)
        for col in chunk:
            if col.startswith(("text_en_", "img_en_")):
                if not np.isfinite(pd.to_numeric(chunk[col], errors="coerce")).all():
                    raise ValueError(f"Invalid scores in {job['id']}: {col}")
        meta = p.read_json(manifest)
        new_meta = dict(meta, signature=new_signature, green_hotfix_migration={
            "old_signature": old_signature, "old_code": old_code,
            "new_code": p.code_hash(), "changed_files": sorted(changed),
            "reason": "GREEN code changed; this stage does not import calculate_russian_text.py",
        })
        updates.append((job["id"], manifest, meta, new_meta))
    # Preflight all stages before changing any manifest. Score files remain untouched.
    for name, manifest, old_meta, new_meta in updates:
        backup = manifest.with_name(manifest.name + ".pre-green-fix.bak")
        if not backup.exists():
            p.atomic_json(backup, old_meta)
        p.atomic_json(manifest, new_meta)
        print(f"Revalidated {name}; CSV unchanged")
    return len(updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--source-snapshot", required=True)
    args = parser.parse_args()
    migrate(p.resolved_plan(p.load_plan(args.plan)), args.work_dir, args.source_snapshot)


if __name__ == "__main__":
    main()

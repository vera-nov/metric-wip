"""File-based training and inference. The controller never imports neural libraries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL_DEFAULTS = {
    "qwen_translator_model": "Qwen/Qwen3.5-9B",
    "hy_mt_model": "tencent/HY-MT1.5-7B",
    "translategemma_model": "google/translategemma-12b-it",
    "green_model": "Qwen/Qwen3.5-9B",
    "cxrbert_model": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "biovilt_model": "microsoft/BiomedVLP-BioViL-T",
}
RUNTIME_DEFAULTS = {
    "translation_max_new_tokens": 1024, "green_max_new_tokens": 2048,
    "green_batch_size": 1, "text_batch_size": 16, "image_batch_size": 16,
    "device": "cuda",
}
ENVIRONMENT_DEFAULTS = {
    "translate": "cxr-qwen", "green": "cxr-qwen",
    "text": "cxr-text", "image": "cxr-image",
}
METHODS = ["radgraph_partial", "radcliq", "ratescore", "cosinesim_cxrbert",
           "bertscore_cxrbert", "cosinesim_biovilt", "bertscore_biovilt"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def code_hash():
    return fingerprint({p.name: digest(p) for p in sorted(ROOT.glob("*.py"))})


def load_plan(path):
    p = Path(path).resolve()
    plan = read_json(p)
    for key in ("input_csv", "image_root"):
        if plan.get(key):
            value = Path(plan[key]).expanduser()
            plan[key] = str((p.parent/value).resolve() if not value.is_absolute() else value)
    return plan


def resolved_plan(plan):
    p = dict(plan)
    p["models"] = {**MODEL_DEFAULTS, **p.get("models", {})}
    p["runtime"] = {**RUNTIME_DEFAULTS, **p.get("runtime", {})}
    p["environments"] = p.get("environments", dict(ENVIRONMENT_DEFAULTS))
    p["translators"] = p.get("translators", ["qwen", "hy_mt", "translategemma"])
    p["translation_kinds"] = p.get("translation_kinds", ["terms", "noterms"])
    p["methods"] = p.get("methods", METHODS)
    if set(p["translators"]) - {"qwen", "hy_mt", "translategemma"}:
        raise ValueError("Unknown translator")
    if set(p["translation_kinds"]) - {"terms", "noterms"}:
        raise ValueError("Unknown translation kind")
    if not p["methods"] or set(p["methods"]) - set(METHODS):
        raise ValueError("Specify at least one known textual metric")
    return p


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, dtype={"report_id": str}, keep_default_na=False)


def merge_exact(left, right):
    """A strict 1:1 join, preserving left order, including IDs with leading zeroes."""
    for df in (left, right):
        if "report_id" not in df or df.report_id.eq("").any() or df.report_id.duplicated().any():
            raise ValueError("Missing, empty or duplicate report_id")
    if set(left.report_id) != set(right.report_id):
        raise ValueError("Cached rows do not match input report identifiers")
    overlap = (set(left.columns) & set(right.columns)) - {"report_id"}
    if overlap:
        raise ValueError(f"Conflicting cached columns: {sorted(overlap)}")
    return left.merge(right, on="report_id", how="left", validate="one_to_one", sort=False)


def prepare(plan, work_dir, require_target):
    from utils import normalize_input_dataframe
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    source = read_csv(plan["input_csv"])
    c = plan["columns"]
    if c.get("row_id") and c["row_id"] != "report_id":
        # Read identifier columns as strings, including custom IDs.
        import pandas as pd
        source = pd.read_csv(plan["input_csv"], dtype={c["row_id"]: str}, keep_default_na=False)
    normalized, mapping = normalize_input_dataframe(
        source, c["candidate"], c["reference"], c["image"], c.get("target"),
        require_target=require_target, row_id_col=c.get("row_id"))
    if normalized.empty:
        raise ValueError("Input dataset is empty")
    for col in ("candidate_report", "reference_report", "image_path"):
        if normalized[col].str.strip().eq("").any():
            raise ValueError(f"Empty required input: {col}")
    cols = ["report_id", "candidate_report", "reference_report", "image_path"]
    if require_target:
        cols.append("human_score")
    base = normalized[cols].copy()
    # Keep absolute image paths for portable jobs. Include image bytes in provenance below.
    image_root = Path(plan.get("image_root") or Path(plan["input_csv"]).parent)
    base["image_path"] = [str((image_root/Path(v)).resolve()) if not Path(v).is_absolute()
                          else str(Path(v)) for v in base.image_path]
    base.to_csv(work_dir/"base.csv", index=False)
    atomic_json(work_dir/"input_manifest.json", {
        "input_csv": plan["input_csv"], "sha256": digest(plan["input_csv"]), "columns": mapping})
    return base, mapping


def stages(plan, selected=None):
    """One translation file per setting, so both branches consume identical text."""
    jobs = []
    pairs = [(tr, kind) for tr in plan["translators"] for kind in plan["translation_kinds"]
             if (tr, kind) != ("translategemma", "terms")]
    if selected is not None:
        pairs = sorted({(s["translator"], s["kind"]) for s in selected if s.get("translator")})
    for tr, kind in pairs:
        jobs.append({"id": f"translate.{tr}.{kind}", "stage": "translate", "translator": tr,
                     "kind": kind, "deps": []})
    if selected is None or any(s["method_key"] == "ru_GREEN_Qwen" for s in selected):
        jobs.append({"id": "green", "stage": "green", "deps": []})
    for tr, kind in pairs:
        methods = plan["methods"] if selected is None else [
            s["method_key"] for s in selected if s["group"] == "english_text"
            and (s["translator"], s["kind"]) == (tr, kind)]
        for group in ("radgraph_partial", "radcliq", "ratescore", "cxrbert", "biovilt"):
            chosen = [m for m in methods if m == group or m.endswith("_"+group)]
            if chosen:
                jobs.append({"id": f"text.{group}.{tr}.{kind}", "stage": "text",
                             "group": group, "translator": tr, "kind": kind,
                             "methods": chosen, "deps": [f"translate.{tr}.{kind}"]})
        if selected is None or any(s["group"] == "image_text" and
                                  (s["translator"], s["kind"]) == (tr, kind) for s in selected):
            jobs.append({"id": f"image.{tr}.{kind}", "stage": "image", "translator": tr,
                         "kind": kind, "deps": [f"translate.{tr}.{kind}"]})
    return jobs


def environment(plan, job):
    envs = plan.get("environments", {})
    keys = [job["id"]]
    if job["stage"] == "translate":
        keys.append("translate."+job["translator"])
    if job["stage"] == "text":
        keys.append("text."+job["group"])
    keys += [job["stage"], "default"]
    value = next((envs[k] for k in keys if k in envs), None)
    if value is None:
        raise ValueError(f"No environment for {job['id']}; set environments.default or a stage override")
    if isinstance(value, str) and value:
        return ["conda", "run", "--no-capture-output", "-n", value, "python"]
    if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
        return value
    raise ValueError("Environment must be a conda name or a nonempty executable argument list")


def paths(work_dir, job):
    return Path(work_dir)/(job["id"]+".csv"), Path(work_dir)/(job["id"]+".manifest.json")


def expected_columns(job):
    if job["stage"] == "translate":
        return [f"{prefix}_{job['translator']}_{job['kind']}" for prefix in ("generation", "gt")]
    if job["stage"] == "green":
        return ["ru_GREEN_Qwen", "ru_GREEN_Qwen_analysis", "ru_GREEN_Qwen_errors_json"]
    if job["stage"] == "image":
        return [f"img_en_biovilt_{job['translator']}_{job['kind']}"]
    from utils import METHOD_BY_KEY
    return [METHOD_BY_KEY[m]["feature_template"].format(translator=job["translator"], kind=job["kind"])
            for m in job["methods"]]


def validate_chunk(base, chunk, job):
    merge_exact(base[["report_id"]], chunk)
    missing = set(expected_columns(job)) - set(chunk.columns)
    if missing:
        raise ValueError(f"Stage {job['id']} did not produce required columns: {sorted(missing)}")


def image_fingerprints(base):
    return {p: digest(p) for p in sorted(set(base.image_path))}


def job_signature(plan, job, work_dir, base, images=None):
    # Do not hash the target labels: feature extraction must not depend on expert annotations.
    data = base[["report_id", "candidate_report", "reference_report", "image_path"]].to_dict("records")
    signature = {"job": job, "data": data, "models": plan["models"],
                 "runtime": plan["runtime"], "environment": environment(plan, job), "code": code_hash(),
                 "dependencies": {dep: digest(Path(work_dir)/(dep+".csv")) for dep in job["deps"]}}
    if job["stage"] == "image":
        signature["images"] = images if images is not None else image_fingerprints(base)
    return fingerprint(signature)


def valid_cache(output, manifest, signature):
    if not output.exists() or not manifest.exists():
        return False
    try:
        meta = read_json(manifest)
        return meta["signature"] == signature and meta["sha256"] == digest(output)
    except (ValueError, KeyError):
        return False


def run_features(plan, work_dir, *, resume=False, only=None, selected=None, dry_run=False,
                 cache_only=False):
    base = read_csv(Path(work_dir)/"base.csv")
    jobs = stages(plan, selected)
    if only:
        requested = set(only)
        known = {j["id"] for j in jobs}
        if requested - known:
            raise ValueError(f"Unknown stages: {sorted(requested-known)}; available: {sorted(known)}")
        for job in reversed(jobs):
            if job["id"] in requested:
                requested.update(job["deps"])
        jobs = [j for j in jobs if j["id"] in requested]
    for job in jobs:
        output, manifest = paths(work_dir, job)
        prefix = environment(plan, job)
        if dry_run:
            print(json.dumps({"stage": job["id"], "command": prefix,
                              "output": str(output), "dependencies": job["deps"]}, ensure_ascii=False))
            continue
        cached_meta = read_json(manifest) if cache_only and manifest.exists() else {}
        images = cached_meta.get("images") if cache_only else None
        if job["stage"] == "image" and not cache_only:
            images = image_fingerprints(base)
        signature = job_signature(plan, job, work_dir, base, images)
        if (resume or cache_only) and valid_cache(output, manifest, signature):
            validate_chunk(base, read_csv(output), job)
            print(f"Reusing {job['id']}", flush=True)
            continue
        if cache_only:
            raise ValueError(f"Missing or stale stage {job['id']}; run feature extraction first")
        print(f"Computing {job['id']}", flush=True)
        # The last successful cache remains intact on worker failure, but its signature is checked.
        request = {"plan": plan, "job": job, "work_dir": str(Path(work_dir).resolve())}
        request_path = Path(work_dir)/(job["id"]+".job.json")
        atomic_json(request_path, request)
        subprocess.run(prefix + [str(ROOT/"pipeline.py"), "worker", "--job", str(request_path.resolve())],
                       cwd=ROOT, check=True)
        chunk = read_csv(output)
        validate_chunk(base, chunk, job)
        versions = read_json(output.with_suffix(".versions.json"))
        atomic_json(manifest, {"signature": signature, "sha256": digest(output),
                               "job": job, "environment": prefix, "versions": versions, "images": images})
    if dry_run or only:
        return None
    features = base[["report_id"] + (["human_score"] if "human_score" in base else [])].copy()
    for job in jobs:
        if job["stage"] != "translate":
            features = merge_exact(features, read_csv(paths(work_dir, job)[0]))
    features.to_csv(Path(work_dir)/"features.csv", index=False)
    return features


def worker(request_path):
    import pandas as pd
    from utils import configure_worker_environment
    request = read_json(request_path)
    plan, job, work_dir = request["plan"], request["job"], Path(request["work_dir"])
    r, m = plan["runtime"], plan["models"]
    configure_worker_environment(r.get("gpu"), r.get("device"))
    base = read_csv(work_dir/"base.csv")
    output, _ = paths(work_dir, job)
    fd, tmp = tempfile.mkstemp(dir=work_dir, prefix=job["id"]+".", suffix=".tmp.csv")
    os.close(fd)
    try:
        kwargs = {"input_csv": str(work_dir/"base.csv"), "output_csv": tmp,
                  "device": r["device"], "gpu": r.get("gpu")}
        if job["stage"] in {"text", "image"}:
            translated = read_csv(work_dir/(job["deps"][0]+".csv"))
            merged = merge_exact(base, translated)
            input_path = work_dir/(job["id"]+".input.csv")
            merged.to_csv(input_path, index=False)
            kwargs.update(input_csv=str(input_path), translators=[job["translator"]],
                          translation_kinds=[job["kind"]])
        if job["stage"] == "translate":
            from translate_reports import _translate_one_translator_columns
            model_key = {"qwen": "qwen_translator_model", "hy_mt": "hy_mt_model",
                         "translategemma": "translategemma_model"}[job["translator"]]
            cols = _translate_one_translator_columns(base, job["translator"], [job["kind"]],
                                                    m[model_key], r["translation_max_new_tokens"], r["device"])
            result = pd.DataFrame({"report_id": base.report_id, **cols})
            result.to_csv(tmp, index=False)
        elif job["stage"] == "green":
            from calculate_russian_text import calculate_green_for_csv
            calculate_green_for_csv(**kwargs, model_name=m["green_model"],
                                    batch_size=r["green_batch_size"], max_new_tokens=r["green_max_new_tokens"])
        elif job["stage"] == "text":
            from calculate_english_text import calculate_english_text_metrics_for_csv
            calculate_english_text_metrics_for_csv(**kwargs, methods=job["methods"],
                                                  cxrbert_model=m["cxrbert_model"], biovilt_model=m["biovilt_model"],
                                                  batch_size=r["text_batch_size"])
        elif job["stage"] == "image":
            from calculate_image_text import calculate_image_text_metrics_for_csv
            calculate_image_text_metrics_for_csv(**kwargs, biovilt_model=m["biovilt_model"],
                                                batch_size=r["image_batch_size"])
        else:
            raise ValueError(f"Unknown stage: {job['stage']}")
        result = read_csv(tmp)
        validate_chunk(base, result, job)
        import numpy as np
        for col in result:
            if col.startswith(("text_en_", "img_en_")) or col == "ru_GREEN_Qwen":
                if not np.isfinite(pd.to_numeric(result[col], errors="coerce")).all():
                    raise ValueError(f"Invalid component scores in {col}")
        os.replace(tmp, output)
        atomic_json(output.with_suffix(".versions.json"), {
            "python": sys.version, "executable": sys.executable,
            "gpu_memory": gpu_memory_usage(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}})
    finally:
        Path(tmp).unlink(missing_ok=True)


def gpu_memory_usage():
    """PyTorch allocator peaks for this worker, not total device use by other jobs."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return [{"device": i, "name": torch.cuda.get_device_name(i),
             "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
             "peak_reserved_gib": torch.cuda.max_memory_reserved(i) / 2**30}
            for i in range(torch.cuda.device_count())]


def fit(plan, work_dir, output):
    import utils as u
    features = run_features(plan, work_dir, cache_only=True)
    training = {"k_outer": 5, "k_inner": 5, "weight_step": 0.05, "random_state": 42,
                "lower_transform": "reciprocal", **plan.get("training", {})}
    allowed = {"k_outer", "k_inner", "weight_step", "random_state", "lower_transform"}
    if set(training) - allowed:
        raise ValueError(f"Unknown training parameters: {sorted(set(training)-allowed)}")
    final = u.select_and_fit_final_config(features, "human_score", plan["translators"],
                                          plan["translation_kinds"], **training)
    extraction_plan = {k: v for k, v in plan.items() if k not in {"input_csv", "training"}}
    # The inference artifact needs only the selected branch, not all candidate environments/models.
    specs = final["selected_features"]
    pairs = {(s["translator"], s["kind"]) for s in specs if s.get("translator")}
    extraction_plan["translators"] = sorted({p[0] for p in pairs})
    extraction_plan["translation_kinds"] = sorted({p[1] for p in pairs})
    extraction_plan["methods"] = sorted({s["method_key"] for s in specs if s["group"] == "english_text"})
    selected_jobs = stages(extraction_plan, specs)
    extraction_plan["environments"] = {j["id"]: environment(plan, j) for j in selected_jobs}
    provenance = {j["id"]: read_json(paths(work_dir, j)[1]) for j in stages(plan)}
    artifact = {"version": 2, **final, "extraction_plan": extraction_plan,
                "models": plan["models"], "runtime": plan["runtime"],
                "columns": read_json(Path(work_dir)/"input_manifest.json")["columns"],
                "provenance": {"code_sha256": code_hash(), "input": read_json(Path(work_dir)/"input_manifest.json"),
                               "stages": provenance}, "target_direction": "lower"}
    atomic_json(output, artifact)
    from file_scores import export_artifact
    export_artifact(artifact, Path(output).with_name("quality_weights.json"),
                    Path(output).with_name("quality_components.json"))
    print(json.dumps({"artifact": str(output), "selected_features": final["selected_features"],
                      "weights": final["weights"], "calibration": final["calibration"],
                      "cv": final["training"]}, ensure_ascii=False, indent=2))
    return artifact


def infer(artifact, input_csv, work_dir, output, runtime_override=None, resume=False, dry_run=False):
    from utils import score_with_config
    if artifact.get("version") != 2 or "extraction_plan" not in artifact:
        raise ValueError("Use a version-2 artifact trained with the staged pipeline")
    if artifact["provenance"]["code_sha256"] != code_hash():
        raise ValueError("Extraction code differs from training; use the same code checkout")
    plan = dict(artifact["extraction_plan"])
    if runtime_override:
        unknown = set(runtime_override) - {"environments", "device", "gpu", "image_root"}
        if unknown:
            raise ValueError(f"Inference may override execution locations only: {sorted(unknown)}")
        plan["environments"] = {**plan["environments"], **runtime_override.get("environments", {})}
        plan["runtime"] = {**plan["runtime"], **{k: runtime_override[k] for k in ("gpu", "device") if k in runtime_override}}
        if "image_root" in runtime_override:
            plan["image_root"] = runtime_override["image_root"]
    plan["input_csv"] = str(Path(input_csv).resolve())
    plan["columns"] = {k: v for k, v in plan["columns"].items() if k != "target"}
    normalized, _ = prepare(plan, work_dir, require_target=False)
    features = run_features(plan, work_dir, resume=resume, selected=artifact["selected_features"], dry_run=dry_run)
    if dry_run:
        return None
    scores = score_with_config(features, artifact)
    joined = merge_exact(normalized, features)
    rows = []
    for i, row in joined.iterrows():
        metrics = [{"feature_col": s["feature_col"], "method_key": s["method_key"],
                    "value": float(row[s["feature_col"]]), "weight": float(w)}
                   for s, w in zip(artifact["selected_features"], artifact["weights"])]
        rows.append({"report_id": row.report_id, "quality_score": float(scores[i]),
                     "metrics": metrics, "GREEN_analysis": row.get("ru_GREEN_Qwen_analysis"),
                     "GREEN_errors_json": row.get("ru_GREEN_Qwen_errors_json")})
    result = {"summary": {"n_rows": len(rows), "quality_score_mean": float(scores.mean())}, "entry-wise": rows}
    atomic_json(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for cmd in ("prepare", "run", "fit"):
        p = subs.add_parser(cmd)
        p.add_argument("--plan", required=True)
        p.add_argument("--work-dir", required=True)
        if cmd == "run":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--only", help="Comma-separated stage IDs; dependencies run automatically")
            p.add_argument("--dry-run", action="store_true")
        if cmd == "fit":
            p.add_argument("--output", required=True)
    p = subs.add_parser("infer")
    p.add_argument("--artifact", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--execution-config")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = subs.add_parser("worker")
    p.add_argument("--job", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.job)
    elif args.command == "infer":
        infer(read_json(args.artifact), args.input, args.work_dir, args.output,
              read_json(args.execution_config) if args.execution_config else None, args.resume, args.dry_run)
    else:
        plan = resolved_plan(load_plan(args.plan))
        prepare(plan, args.work_dir, require_target=True)
        if args.command == "run":
            run_features(plan, args.work_dir, resume=args.resume,
                         only=args.only.split(",") if args.only else None, dry_run=args.dry_run)
        elif args.command == "fit":
            fit(plan, args.work_dir, args.output)


if __name__ == "__main__":
    main()

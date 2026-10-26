"""JSON-list interface and CPU-only fusion; no neural dependencies are imported."""
from __future__ import annotations

import math
from pathlib import Path

from pipeline import atomic_json, read_json

COMPONENTS = {"russian": "ru_text", "text": "english_text", "image": "image_text"}
FIELDS = ("reference_text", "candidate_text", "reference_image_path")
DEFAULT_OUTPUTS = {"russian": "russian_scores.json", "text": "text_scores.json", "image": "image_scores.json"}


def read_records(path, *, scored=False):
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty JSON list")
    has_ids = any(isinstance(row, dict) and "report_id" in row for row in records)
    seen_ids = set()
    for i, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {i} must be an object")
        for field in FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{path}: row {i} needs a nonempty string {field}")
        if has_ids:
            report_id = row.get("report_id")
            if not isinstance(report_id, str) or not report_id.strip():
                raise ValueError(f"{path}: row {i} needs a nonempty string report_id")
            if report_id in seen_ids:
                raise ValueError(f"{path}: duplicate report_id {report_id!r}")
            seen_ids.add(report_id)
        if scored:
            finite_number(row.get("score"), f"{path}: row {i} score")
    return records


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_weights(weights):
    if set(weights) != {"scaling", "fusion", "calibration"}:
        raise ValueError("Weights must contain only scaling, fusion and calibration")
    scaling, fusion, calibration = (weights[k] for k in ("scaling", "fusion", "calibration"))
    for key in ("means", "stds", "transforms"):
        if not isinstance(scaling.get(key), list) or len(scaling[key]) != 3:
            raise ValueError(f"scaling.{key} must contain three values in A/B/C order")
    if not isinstance(fusion.get("weights"), list) or len(fusion["weights"]) != 3:
        raise ValueError("fusion.weights must contain three values")
    means = [finite_number(x, "mean") for x in scaling["means"]]
    stds = [finite_number(x, "std") for x in scaling["stds"]]
    coeffs = [finite_number(x, "weight") for x in fusion["weights"]]
    if any(x <= 0 for x in stds):
        raise ValueError("Scaling standard deviations must be positive")
    if any(x < 0 for x in coeffs) or not math.isclose(sum(coeffs), 1, abs_tol=1e-8):
        raise ValueError("Fusion weights must be nonnegative and sum to one")
    if set(scaling["transforms"]) - {"identity", "reciprocal", "negate"}:
        raise ValueError("Unknown quality transform")
    a = finite_number(calibration.get("a"), "calibration.a")
    b = finite_number(calibration.get("b"), "calibration.b")
    if calibration.get("kind") != "logistic" or a <= 0:
        raise ValueError("Expected logistic calibration with a > 0")
    return means, stds, coeffs, a, b


def export_artifact(artifact, weights_path, components_path):
    """Export learned numbers separately from model/branch selection and provenance."""
    specs = artifact["selected_features"]
    if len(specs) != 3 or {s["group"] for s in specs} != set(COMPONENTS.values()):
        raise ValueError("Expected one selected feature for each of the three components")
    order = [next(i for i, s in enumerate(specs) if s["group"] == group)
             for group in COMPONENTS.values()]
    standardizer = artifact.get("standardizer") or artifact["standardization"]
    weights = {
        "scaling": {
            "means": [standardizer["means"][i] for i in order],
            "stds": [standardizer["stds"][i] for i in order],
            "transforms": [(specs[i].get("quality_transform") or "reciprocal")
                           if not specs[i]["higher_is_better"] else "identity" for i in order],
        },
        "fusion": {"weights": [artifact["weights"][i] for i in order]},
        "calibration": {k: artifact["calibration"][k] for k in ("kind", "a", "b")},
    }
    validate_weights(weights)
    plan = {k: v for k, v in artifact["extraction_plan"].items()
            if k not in {"columns", "image_root", "input_csv", "environments"}}
    # Execution locations belong to the machine, not to the fitted method.
    plan["runtime"] = {k: v for k, v in plan.get("runtime", {}).items() if k != "gpu"}
    config = {"version": 1, "plan": plan,
              "components": {name: specs[i] for name, i in zip(COMPONENTS, order)}}
    atomic_json(weights_path, weights)
    atomic_json(components_path, config)
    return weights, config


def fuse_files(russian_path, text_path, image_path, weights_path, output="final_scores.json"):
    weights = read_json(weights_path)
    means, stds, coeffs, a, b = validate_weights(weights)
    sources = [read_records(path, scored=True) for path in (russian_path, text_path, image_path)]
    if len({len(rows) for rows in sources}) != 1:
        raise ValueError("A/B/C have different row counts")
    result = []
    for i, rows in enumerate(zip(*sources)):
        originals = [{k: v for k, v in row.items() if k != "score"} for row in rows]
        if any(row != originals[0] for row in originals[1:]):
            raise ValueError(f"A/B/C row {i} differs: use the same input and row order")
        raw = 0.0
        for j, row in enumerate(rows):
            value = finite_number(row["score"], "score")
            transform = weights["scaling"]["transforms"][j]
            if transform == "reciprocal":
                if value <= 0:
                    raise ValueError("Reciprocal transform requires strictly positive scores")
                value = 1 / value
            elif transform == "negate":
                value = -value
            raw += (value - means[j]) / stds[j] * coeffs[j]
        z = a * raw + b
        if not math.isfinite(z):
            raise ValueError("Fusion produced a non-finite value")
        score = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
        result.append({**originals[0], "score": score})
    atomic_json(output, result)
    return result


def score_component(component, input_path, config_path="quality_components.json", *,
                    output=None, work_dir="work/evaluation", resume=True, dry_run=False):
    import pandas as pd
    from pipeline import prepare, resolved_plan, run_features

    records = read_records(input_path)
    config = read_json(config_path)
    if config.get("version") != 1:
        raise ValueError("Unsupported component config; run export_weights.py")
    spec = config["components"][component]
    if spec["group"] != COMPONENTS[component]:
        raise ValueError("Component and selected feature group disagree")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    input_root = Path(input_path).resolve().parent
    frame = pd.DataFrame([{
        "report_id": row.get("report_id", str(i)), "candidate_report": row["candidate_text"],
        "reference_report": row["reference_text"],
        "image_path": str((input_root / row["reference_image_path"]).resolve()),
    } for i, row in enumerate(records)])
    # Each component can run independently and reuses the same translation cache.
    source = work / "records.csv"
    frame.to_csv(source, index=False)
    plan = resolved_plan({**config["plan"], "input_csv": str(source), "image_root": str(input_root),
                          "columns": {"row_id": "report_id", "candidate": "candidate_report",
                                      "reference": "reference_report", "image": "image_path"}})
    prepare(plan, work, require_target=False)
    features = run_features(plan, work, resume=resume, selected=[spec], dry_run=dry_run)
    if dry_run:
        return None
    values = features[spec["feature_col"]].tolist()
    result = [{**row, "score": finite_number(value, "component score")}
              for row, value in zip(records, values)]
    atomic_json(output or DEFAULT_OUTPUTS[component], result)
    return result

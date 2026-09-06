"""One-shot Formal-Evaluation-v1 over the frozen 40-path pilot generation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from repro.evaluators.formal.metrics import evaluate_prepared
from repro.evaluators.formal.protocol import PROTOCOL_VERSION, prepare_path, protocol_definition
from repro.evaluators.formal.title_resolver import TitleResolver

ROOT = Path(__file__).resolve().parents[3]
GENERATION = ROOT / "repro/results/pilot/generation_v1"
MANIFEST = ROOT / "repro/results/pilot/pilot_manifest.json"
OUT = ROOT / "repro/results/pilot/formal_evaluation_v1"
DOC = ROOT / "repro/docs/PILOT_FORMAL_EVALUATION.md"
CHECKPOINT = ROOT / "repro/results/evaluator_validation/checkpoints/SASRec-ml-1m-sas.pth"
EXPECTED_MANIFEST_SHA = "0fd235067f2fde479eaf6ab102daae225684dbc9b4f1ec9d723ea1795e49301d"
EXPECTED_CHECKPOINT_SHA = "e19b98c7b37c667e24b13b508ae5319965a23daae7a6b8dcb8dcfd2b329760bc"
METHODS = ("baseline", "mi_bridge")
LABELS = {"baseline": "Baseline", "mi_bridge": "MI-Bridge"}
MAIN = ("IoI", "IoR", "ProxyAcceptability", "Coherence")
MECHANISM = ("HISTORY_REUSE_RATE", "NEW_INTERMEDIATE_COUNT", "BRIDGE_CANDIDATE_USAGE_COUNT")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_manifest_sha(data: dict) -> str:
    copy = dict(data)
    copy.pop("manifest_sha256", None)
    raw = json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def directory_freeze(folder: Path) -> dict:
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    inventory = [{"path": p.relative_to(folder).as_posix(), "sha256": sha(p), "size_bytes": p.stat().st_size}
                 for p in files]
    digest = hashlib.sha256()
    for item in inventory:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {"algorithm": "sha256(relative_posix_path + NUL + file_sha256 + LF, sorted by path)",
            "directory": str(folder), "file_count": len(files), "files": inventory,
            "generation_v1_sha256": digest.hexdigest()}


def protected_snapshot() -> dict:
    original = read(ROOT / "repro/source_hashes.json")["sha256"]
    groups = {
        "original_14": [ROOT / p for p in original],
        "baseline_results": [p for d in (ROOT / "repro/results/minimal_baseline", ROOT / "repro/results/minimal_baseline_v2") for p in d.rglob("*") if p.is_file()],
        "mi_bridge_method": [p for p in (ROOT / "repro/methods/mi_bridge").rglob("*") if p.is_file() and "__pycache__" not in p.parts],
        "formal_protocol": [p for p in (ROOT / "repro/evaluators/formal").rglob("*") if p.is_file() and "__pycache__" not in p.parts],
        "pilot_manifest": [MANIFEST],
        "sasrec_checkpoint": [CHECKPOINT],
    }
    return {name: {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)} for name, paths in groups.items()}


def save(name: str, value) -> None:
    path = OUT / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def record_paths(manifest: dict) -> list[tuple[dict, dict]]:
    result = []
    for user in manifest["users"]:
        for method in METHODS:
            for index in (1, 2):
                path = GENERATION / "users" / str(user["user_id"]) / f"{method}_path_{index}.json"
                result.append((user, read(path)))
    return result


def path_output(record: dict, scored: dict) -> dict:
    m, v = scored["metrics"], scored["validity"]
    resolved = [x["resolved_title"] for x in scored["items"] if x["resolution_status"] == "resolved"]
    unresolved = [x["raw_title"] for x in scored["items"] if x["resolution_status"] != "resolved"]
    return {
        "user_id": record["user_id"], "method": record["method"], "path_index": record["path_index"],
        "parse_success": v["parse_success"], "formal_metric_status": scored["FORMAL_METRIC_STATUS"],
        "target_present": v["target_present"], "target_is_last": v["target_is_last"],
        "target_original_position": v["target_original_position"], "raw_path": scored["raw_path"],
        "resolved_path": resolved, "unresolved_items": unresolved,
        "P_before": m["P_before"], "P_after": m["P_after"], "IoI": m["IoI"],
        "R_before": m["R_before"], "R_after": m["R_after"], "IoR": m["IoR"],
        "ProxyAcceptability": m["ProxyAcceptability"], "Coherence": m["Coherence"],
        "HistoryReuseRate": scored["mechanism_metrics"]["HISTORY_REUSE_RATE"],
        "NewIntermediateCount": scored["mechanism_metrics"]["NEW_INTERMEDIATE_COUNT"],
        "BridgeCandidateUsageCount": scored["mechanism_metrics"]["BRIDGE_CANDIDATE_USAGE_COUNT"],
        "validity": v,
    }


def avg(values):
    values = [x for x in values if x is not None]
    return mean(values) if values else None


def user_aggregate(paths: list[dict]) -> list[dict]:
    rows = []
    for user_id in sorted({p["user_id"] for p in paths}):
        for method in METHODS:
            group = [p for p in paths if p["user_id"] == user_id and p["method"] == method]
            valid = [p for p in group if p["formal_metric_status"] == "valid"]
            parsed = [p for p in group if p["parse_success"]]
            row = {"user_id": user_id, "method": method, "valid_path_count": len(valid), "total_path_count": len(group),
                   "SR_user": avg([float(p["target_present"]) for p in parsed]),
                   "TargetLastRate_user": avg([float(p["target_is_last"]) for p in parsed])}
            for metric in MAIN:
                row[f"{metric}_user_mean"] = avg([p[metric] for p in valid])
            for metric in ("HistoryReuseRate", "NewIntermediateCount", "BridgeCandidateUsageCount"):
                row[f"{metric}_user_mean"] = avg([p[metric] for p in valid])
            rows.append(row)
    return rows


def method_aggregate(paths: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        group = [p for p in paths if p["method"] == method]
        valid = [p for p in group if p["formal_metric_status"] == "valid"]
        parsed = [p for p in group if p["parse_success"]]
        row = {"VALID_PATH_COUNT": len(valid), "TOTAL_PATH_COUNT": len(group),
               "SR": avg([float(p["target_present"]) for p in parsed]),
               "TargetLastRate": avg([float(p["target_is_last"]) for p in parsed]),
               "metric_defined_path_counts": {}}
        for metric in MAIN:
            vals = [p[metric] for p in valid if p[metric] is not None]
            row[f"{metric}_mean"] = mean(vals) if vals else None
            row[f"{metric}_median"] = median(vals) if vals else None
            row["metric_defined_path_counts"][metric] = len(vals)
        for metric in ("HistoryReuseRate", "NewIntermediateCount", "BridgeCandidateUsageCount"):
            row[f"{metric}_mean"] = avg([p[metric] for p in valid])
        result[LABELS[method]] = row
    return result


def paired(users: list[dict]) -> dict:
    rows, counts = [], {m: 0 for m in MAIN}
    by_key = {(x["user_id"], x["method"]): x for x in users}
    for user_id in sorted({x["user_id"] for x in users}):
        b, o = by_key[user_id, "baseline"], by_key[user_id, "mi_bridge"]
        row = {"user_id": user_id}
        complete = True
        for metric in MAIN:
            bv, ov = b[f"{metric}_user_mean"], o[f"{metric}_user_mean"]
            delta = ov - bv if bv is not None and ov is not None else None
            row[f"delta_{metric}"] = delta
            complete &= delta is not None
            if delta is not None and delta > 0:
                counts[metric] += 1
        for metric in ("HistoryReuseRate", "NewIntermediateCount", "BridgeCandidateUsageCount"):
            bv, ov = b[f"{metric}_user_mean"], o[f"{metric}_user_mean"]
            row[f"delta_{metric}"] = ov - bv if bv is not None and ov is not None else None
        rows.append(row)
    return {"paired_valid_user_count": sum(all(r[f"delta_{m}"] is not None for m in MAIN) for r in rows),
            "improved_user_counts": counts, "users": rows,
            "mechanism_improved_user_counts": {
                "HistoryReuseRate": sum(r["delta_HistoryReuseRate"] is not None and r["delta_HistoryReuseRate"] < 0 for r in rows),
                "NewIntermediateCount": sum(r["delta_NewIntermediateCount"] is not None and r["delta_NewIntermediateCount"] > 0 for r in rows),
            }, "significance_tests_performed": False}


def run() -> None:
    if OUT.exists():
        raise FileExistsError("formal_evaluation_v1 exists; refusing overwrite or rerun")
    manifest = read(MANIFEST)
    if canonical_manifest_sha(manifest) != EXPECTED_MANIFEST_SHA or manifest.get("manifest_sha256") != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Pilot manifest hash mismatch")
    if PROTOCOL_VERSION != "Formal-Evaluation-v1" or manifest["formal_protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError("Formal protocol mismatch")
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Checkpoint hash mismatch")
    freeze_before = directory_freeze(GENERATION)
    protected_before = protected_snapshot()
    records = record_paths(manifest)
    if len(records) != 40 or sum(r["method"] == "baseline" for _, r in records) != 20 or not all(r["parse_success"] for _, r in records):
        raise RuntimeError("Expected exactly 40 parsed records (20 per method)")

    from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
    from repro.evaluators.sasrec.data_adapter import DataAdapter
    from repro.evaluators.sasrec.evaluator import SASRecEvaluator
    model, dataset, _, checkpoint_metadata = load_checkpoint()
    evaluator, adapter = SASRecEvaluator(model), DataAdapter(dataset)
    resolver = TitleResolver.from_movies_dat(ROOT / "dataset/ml-1m/movies.dat")
    outputs = []
    for user, record in records:
        target = resolver.resolve(user["target"]["title"])
        if target["raw_item_id"] != user["target"]["movie_id"]:
            raise RuntimeError(f"Target mismatch for user {user['user_id']}")
        history_raw = [x["movie_id"] for x in user["history"]]
        history = adapter.history_raw_ids_to_internal(history_raw)
        candidates = {x["id"] for x in user["bridge"]["direct_candidates"]}
        prepared = prepare_path({"path": record["parsed_path"], "parse_success": record["parse_success"]},
                                resolver, target, set(history_raw), candidates)
        scored, _ = evaluate_prepared(prepared, target, history, adapter, evaluator, diagnostic_drop=False)
        outputs.append(path_output(record, scored))

    users = user_aggregate(outputs)
    methods = method_aggregate(outputs)
    pairs = paired(users)
    unresolved_paths = [p for p in outputs if p["unresolved_items"]]
    validity = {"total_path_count": len(outputs), "formal_valid_path_count": sum(p["formal_metric_status"] == "valid" for p in outputs),
                "formal_invalid_path_count": sum(p["formal_metric_status"] != "valid" for p in outputs),
                "unresolved_path_count": len(unresolved_paths),
                "unresolved_items_summary": [{"user_id": p["user_id"], "method": p["method"], "path_index": p["path_index"], "items": p["unresolved_items"]} for p in unresolved_paths],
                "status_counts": {s: sum(p["formal_metric_status"] == s for p in outputs) for s in sorted({p["formal_metric_status"] for p in outputs})}}
    mechanism = {k: {m: methods[k][f"{m}_mean"] for m in ("HistoryReuseRate", "NewIntermediateCount", "BridgeCandidateUsageCount")} for k in methods}
    freeze_after = directory_freeze(GENERATION)
    protected_after = protected_snapshot()
    if freeze_before != freeze_after or protected_before != protected_after:
        raise RuntimeError("Frozen/protected artifact changed during evaluation")
    summary = {"protocol_version": PROTOCOL_VERSION, **validity, "method_level_metrics": methods,
               "mechanism_metrics": mechanism, "paired_summary": pairs}
    metadata = {"timestamp": datetime.now(timezone.utc).isoformat(), "protocol_version": PROTOCOL_VERSION,
                "manifest_sha256": EXPECTED_MANIFEST_SHA, "generation_v1_sha256": freeze_before["generation_v1_sha256"],
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "mapping_status": "LIKELY_COMPATIBLE",
                "python": platform.python_version(), "torch": importlib.metadata.version("torch"),
                "recbole": importlib.metadata.version("recbole"), "device": "cpu", "checkpoint_metadata": checkpoint_metadata,
                "path_generation_performed": False, "training_performed": False, "significance_tests_performed": False,
                "aggregation": "path to user, then paired user comparison", "protected_sha256_before": protected_before,
                "protected_sha256_after": protected_after, "generation_hash_before": freeze_before["generation_v1_sha256"],
                "generation_hash_after": freeze_after["generation_v1_sha256"], "all_protection_checks_passed": True,
                "protocol": protocol_definition(CHECKPOINT, EXPECTED_CHECKPOINT_SHA)}
    OUT.mkdir(parents=True)
    save("generation_freeze.json", freeze_before)
    save("path_level_metrics.json", outputs)
    save("user_level_metrics.json", users)
    save("method_level_metrics.json", methods)
    save("mechanism_metrics.json", mechanism)
    save("validity_metrics.json", validity)
    save("paired_differences.json", pairs)
    save("evaluation_metadata.json", metadata)
    save("summary.json", summary)
    print(json.dumps({"generation_v1_sha256": freeze_before["generation_v1_sha256"], **validity,
                      "methods": methods, "paired": pairs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()

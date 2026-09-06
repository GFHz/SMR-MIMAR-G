"""Generate and formally evaluate the frozen 5-user catalog-grounded ablation."""
from __future__ import annotations

import difflib
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from repro.experiments.pilot import generate_pilot_paths as generation
from repro.experiments.pilot.evaluate_pilot_formal import (
    CHECKPOINT, EXPECTED_CHECKPOINT_SHA, LABELS, MAIN, MECHANISM,
    method_aggregate, paired, path_output, protected_snapshot, read, sha, user_aggregate,
)
from repro.experiments.pilot.freeze_pilot_manifest import payload_sha256
from repro.evaluators.formal.metrics import evaluate_prepared
from repro.evaluators.formal.protocol import PROTOCOL_VERSION, prepare_path
from repro.evaluators.formal.title_resolver import TitleResolver
from repro.methods.mi_bridge.bridge_prompt import build_bridge_prompt
from repro.utils import ROOT, read_json, write_json

SEED = 20260905
SOURCE_USERS = [5723, 2677, 829, 5651, 5021, 3113, 4671, 1032, 2249, 419]
EXPECTED_SELECTED = [419, 5021, 2677, 3113, 2249]
PILOT = ROOT / "repro/results/pilot/pilot_manifest.json"
EXPECTED_MANIFEST_SHA = "0fd235067f2fde479eaf6ab102daae225684dbc9b4f1ec9d723ea1795e49301d"
OUT = ROOT / "repro/results/pilot/catalog_grounded_ablation_v1"
GEN_OUT = OUT / "generation"
EVAL_OUT = OUT / "evaluation"
GROUNDING_INSTRUCTION = (
    "Every recommended movie must be selected from the provided MovieLens-1M catalog candidate set. "
    "Do not invent or recommend titles outside this set."
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_catalog():
    result = []
    with (ROOT / "dataset/ml-1m/movies.dat").open(encoding="latin-1") as stream:
        for line in stream:
            movie_id, title, genres = line.rstrip("\r\n").split("::")
            result.append({"movie_id": int(movie_id), "title": title, "genres": genres.split("|")})
    return result


def select_users():
    return random.Random(SEED).sample(SOURCE_USERS, 5)


def candidate_pool(user, catalog):
    ordered = []
    seen = set()
    def add(movie):
        movie_id = movie.get("movie_id", movie.get("id"))
        if movie_id not in seen:
            by_id = catalog[movie_id]
            ordered.append(by_id)
            seen.add(movie_id)
    for movie in user["history"]:
        add(movie)
    add(user["target"])
    for movie in user["bridge"]["direct_candidates"]:
        add(movie)
    remaining = [movie for movie_id, movie in catalog.items() if movie_id not in seen]
    random.Random(SEED + user["user_id"]).shuffle(remaining)
    for movie in remaining:
        if len(ordered) == 100:
            break
        add(movie)
    if len(ordered) != 100 or len({x["movie_id"] for x in ordered}) != 100:
        raise RuntimeError("Candidate pool is not exactly 100 unique catalog items")
    return ordered


def grounding_context(pool):
    return ("\n\n[CATALOG GROUNDING]\n" + GROUNDING_INSTRUCTION + "\n"
            "MovieLens-1M catalog candidate set:\n" +
            json.dumps([x["title"] for x in pool], ensure_ascii=False) +
            "\n[/CATALOG GROUNDING]\n")


def prompt_pair(user, pool):
    base_user = generation.user_prompt(user)
    grounded_user = base_user + grounding_context(pool)
    baseline = {"system_prompt": generation.SYSTEM_PROMPT, "user_prompt": grounded_user,
        "formatting_user_prompt": generation.FORMAT_PROMPT, "bridge_context": None,
        "selected_bridge": None, "bridge_type": None, "bridge_candidates": None}
    bridge = user["bridge"]
    selected = generation.selected_bridge(user)
    augmented = build_bridge_prompt(generation.SYSTEM_PROMPT, grounded_user, selected,
                                    "direct", bridge["direct_candidates"])
    ours = {**augmented, "formatting_user_prompt": generation.FORMAT_PROMPT,
            "selected_bridge": selected, "bridge_type": "direct",
            "bridge_candidates": bridge["direct_candidates"]}
    for bundle in (baseline, ours):
        bundle["prompt_hash"] = canonical_hash({k: bundle[k] for k in
            ("system_prompt", "user_prompt", "formatting_user_prompt")})
    if ours["user_prompt"] != baseline["user_prompt"] + ours["bridge_context"]:
        raise RuntimeError("Method prompt difference is not exactly bridge context")
    return baseline, ours


def preflight(check_service=False):
    manifest = read_json(PILOT)
    if manifest["manifest_sha256"] != EXPECTED_MANIFEST_SHA or payload_sha256(manifest) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Frozen pilot manifest mismatch")
    if manifest["pilot_user_ids"] != SOURCE_USERS or select_users() != EXPECTED_SELECTED:
        raise RuntimeError("Deterministic user sampling mismatch")
    # Reuse all frozen model/config/parser/source checks from generation_v1.
    generation.preflight(check_service=check_service)
    by_user = {x["user_id"]: x for x in manifest["users"]}
    catalog = {x["movie_id"]: x for x in load_catalog()}
    users, pools, prompts = [], {}, {}
    for user_id in EXPECTED_SELECTED:
        user = by_user[user_id]
        pool = candidate_pool(user, catalog)
        baseline, ours = prompt_pair(user, pool)
        users.append(user)
        pools[str(user_id)] = pool
        prompts[str(user_id)] = (baseline, ours)
    return manifest, users, pools, prompts


def slots(users):
    return [(user, method, index) for user in users
            for method in ("baseline", "mi_bridge") for index in (1, 2)]


def evaluate(users, pools, records):
    from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
    from repro.evaluators.sasrec.data_adapter import DataAdapter
    from repro.evaluators.sasrec.evaluator import SASRecEvaluator
    model, dataset, _, _ = load_checkpoint()
    evaluator, adapter = SASRecEvaluator(model), DataAdapter(dataset)
    resolver = TitleResolver.from_movies_dat(ROOT / "dataset/ml-1m/movies.dat")
    by_user = {x["user_id"]: x for x in users}
    outputs = []
    for record in records:
        user = by_user[record["user_id"]]
        target = resolver.resolve(user["target"]["title"])
        history_raw = [x["movie_id"] for x in user["history"]]
        history = adapter.history_raw_ids_to_internal(history_raw)
        bridge_ids = {x["id"] for x in user["bridge"]["direct_candidates"]}
        prepared = prepare_path({"path": record["parsed_path"], "parse_success": record["parse_success"]},
                                resolver, target, set(history_raw), bridge_ids)
        pool_ids = {x["movie_id"] for x in pools[str(user["user_id"])]}
        violations = [x["raw_title"] for x in prepared["evaluation_intermediates"]
                      if x["resolution_status"] != "resolved" or x["raw_item_id"] not in pool_ids]
        compliant = prepared["validity"]["parse_success"] and not violations
        original_status = prepared["FORMAL_METRIC_STATUS"]
        if original_status == "valid" and not compliant:
            prepared["FORMAL_METRIC_STATUS"] = "invalid_catalog_violation"
            prepared["STRICT_EVALUATION_VALID"] = False
        scored, _ = evaluate_prepared(prepared, target, history, adapter, evaluator, diagnostic_drop=False)
        row = path_output(record, scored)
        row.update(CATALOG_COMPLIANT=compliant, catalog_violations=violations,
                   resolver_status_before_catalog_check=original_status)
        outputs.append(row)
    return outputs


def run():
    if OUT.exists():
        raise FileExistsError("catalog_grounded_ablation_v1 exists; refusing overwrite or rerun")
    manifest, users, pools, prompts = preflight(check_service=True)
    protected_before = protected_snapshot()
    pilot_generation_hash = generation.canonical_hash(read_json(ROOT / "repro/results/pilot/generation_v1/generation_manifest.json"))
    OUT.mkdir(parents=True)
    write_json(OUT / "fairness_manifest.json", {"version": "Catalog-Grounded-Fairness-Ablation-v1",
        "FAIRNESS_SEED": SEED, "source_user_ids": SOURCE_USERS, "selected_user_ids": EXPECTED_SELECTED,
        "source_pilot_manifest_sha256": EXPECTED_MANIFEST_SHA, "protocol_version": PROTOCOL_VERSION,
        "planned_path_count": 20, "paths_per_method_per_user": 2,
        "candidate_pool_size": 100, "prompt_only_method_difference": "bridge context",
        "model_configuration": read_json(ROOT / "repro/local_config.json")})
    write_json(OUT / "candidate_pools.json", {uid: {"size": len(pool), "items": pool} for uid, pool in pools.items()})
    diffs = {}
    for uid, (baseline, ours) in prompts.items():
        diffs[uid] = {"baseline_prompt_hash": baseline["prompt_hash"], "ours_prompt_hash": ours["prompt_hash"],
            "common_grounding_context_sha256": canonical_hash(grounding_context(pools[uid])),
            "bridge_context": ours["bridge_context"],
            "unified_diff": list(difflib.unified_diff(baseline["user_prompt"].splitlines(),
                ours["user_prompt"].splitlines(), fromfile="grounded_baseline", tofile="grounded_mi_bridge", lineterm="")),
            "only_difference_is_bridge_context": ours["user_prompt"] == baseline["user_prompt"] + ours["bridge_context"]}
    write_json(OUT / "prompt_diff.json", diffs)

    records = []
    old_out = generation.OUT
    generation.OUT = GEN_OUT
    try:
        for ordinal, (user, method, index) in enumerate(slots(users), 1):
            bundle = prompts[str(user["user_id"])][0 if method == "baseline" else 1]
            record = generation.record_for(user, method, index, bundle)
            record["record_version"] = "Catalog-Grounded-Fairness-Ablation-v1"
            record["catalog_candidate_pool_ids"] = [x["movie_id"] for x in pools[str(user["user_id"])]]
            destination = GEN_OUT / "users" / str(user["user_id"]) / f"{method}_path_{index}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_json(destination, record)
            records.append(record)
            print(f"[{ordinal}/20] user={user['user_id']} method={method} path={index} parse={record['parse_success']}", flush=True)
    finally:
        generation.OUT = old_out
    if len(records) != 20:
        raise RuntimeError("Did not execute exactly 20 planned paths")

    outputs = evaluate(users, pools, records)
    users_out = user_aggregate(outputs)
    methods = method_aggregate(outputs)
    pairs = paired(users_out)
    compliance = {label: sum(x["CATALOG_COMPLIANT"] for x in outputs if x["method"] == method) / 10
                  for method, label in LABELS.items()}
    mechanism = {label: {m: methods[label][f"{m}_mean"] for m in
        ("HistoryReuseRate", "NewIntermediateCount", "BridgeCandidateUsageCount")} for label in methods}
    guidance_support = (methods["MI-Bridge"]["IoI_mean"] > methods["Baseline"]["IoI_mean"] and
                        methods["MI-Bridge"]["IoR_mean"] > methods["Baseline"]["IoR_mean"])
    improved = pairs["improved_user_counts"]
    fairness_result = "A_SUPPORTS_BRIDGE_EFFECT" if guidance_support or (improved["IoI"] > pairs["paired_valid_user_count"]/2 and improved["IoR"] > pairs["paired_valid_user_count"]/2) else "B_MIXED"
    EVAL_OUT.mkdir(parents=True)
    write_json(EVAL_OUT / "path_level_metrics.json", outputs)
    write_json(EVAL_OUT / "user_level_metrics.json", users_out)
    write_json(EVAL_OUT / "method_level_metrics.json", methods)
    write_json(EVAL_OUT / "paired_differences.json", pairs)
    write_json(EVAL_OUT / "catalog_compliance.json", compliance)
    if protected_snapshot() != protected_before or payload_sha256(read_json(PILOT)) != EXPECTED_MANIFEST_SHA or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Protected frozen input changed")
    summary = {"FAIRNESS_SEED": SEED, "selected_user_ids": EXPECTED_SELECTED, "path_count": len(outputs),
        "baseline_path_count": 10, "mi_bridge_path_count": 10, "catalog_compliance_rate": compliance,
        "method_level_metrics": methods, "mechanism_metrics": mechanism, "paired_analysis": pairs,
        "PROMPT_ONLY_METHOD_DIFFERENCE": "bridge context", "FAIRNESS_RESULT": fairness_result,
        "formal_protocol": PROTOCOL_VERSION, "no_significance_test": True,
        "protection": {"ORIGINAL_14_FILES_UNCHANGED": True, "BASELINE_V2_UNCHANGED": True,
            "MI_BRIDGE_V1_UNCHANGED": True, "FORMAL_EVALUATION_PROTOCOL_UNCHANGED": True,
            "PILOT_V1_UNCHANGED": True, "SASREC_CHECKPOINT_UNCHANGED": True},
        "completed_at": datetime.now(timezone.utc).isoformat(), "generation_count": 20,
        "pilot_generation_manifest_content_hash_before": pilot_generation_hash}
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight"]:
        _, users, pools, prompts = preflight(check_service=True)
        print(json.dumps({"selected_user_ids": [x["user_id"] for x in users],
            "pool_sizes": {k: len(v) for k, v in pools.items()},
            "prompt_only_method_difference": all(o["user_prompt"] == b["user_prompt"] + o["bridge_context"] for b, o in prompts.values())}, indent=2))
    elif sys.argv[1:]:
        raise SystemExit("Usage: python -m repro.experiments.pilot.run_catalog_grounded_ablation [--preflight]")
    else:
        run()

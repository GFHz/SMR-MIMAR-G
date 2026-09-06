"""24-path controlled generation ablation on four discriminative frozen users."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from repro.experiments.pilot import generate_pilot_paths as generation
from repro.experiments.pilot.evaluate_pilot_formal import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, protected_snapshot, sha
from repro.experiments.pilot.freeze_pilot_manifest import payload_sha256
from repro.experiments.pilot.run_catalog_grounded_ablation import GROUNDING_INSTRUCTION, canonical_hash, load_catalog
from repro.evaluators.formal.metrics import evaluate_prepared
from repro.evaluators.formal.protocol import PROTOCOL_VERSION, prepare_path
from repro.evaluators.formal.title_resolver import TitleResolver
from repro.methods.mi_bridge.bridge_prompt import build_bridge_prompt
from repro.methods.mi_bridge.retrieve_bridge_movies import retrieve_bridge_movies
from repro.utils import ROOT, read_json, verify_sources, write_json

OUT = ROOT / "repro/results/bridge_score_generation_ablation"
GEN_STATE = OUT / "generation_records.json"
PILOT = ROOT / "repro/results/pilot/pilot_manifest.json"
AUDIT = ROOT / "repro/results/bridge_score_ablation_offline/bridge_score_comparison.json"
USERS = [5723, 2677, 3113, 2249]
VARIANTS = ("need", "affinity", "balance")
EXPECTED_MANIFEST_SHA = "0fd235067f2fde479eaf6ab102daae225684dbc9b4f1ec9d723ea1795e49301d"
POOL_SIZE = 100
SEED = 20260905


def grounding_context(pool):
    return ("\n\n[CATALOG GROUNDING]\n" + GROUNDING_INSTRUCTION + "\n"
            "MovieLens-1M catalog candidate set:\n" +
            json.dumps([x["title"] for x in pool], ensure_ascii=False) +
            "\n[/CATALOG GROUNDING]\n")


def method_bridge(user, selected, catalog):
    pair = {"user_existing_interest": selected["existing_interest"],
            "target_related_interest": selected["target_interest"],
            "bridge_score": selected["bridge_score"]}
    history = [dict(x, id=x["movie_id"]) for x in user["history"]]
    target = dict(user["target"], id=user["target"]["movie_id"])
    retrieval_catalog = [dict(x, id=x["movie_id"]) for x in catalog.values()]
    retrieval = retrieve_bridge_movies(pair, retrieval_catalog, history, target)
    if selected["feasibility_mode"] != "direct" or retrieval["candidate_count"] != selected["direct_candidate_count"]:
        raise RuntimeError("Selected bridge feasibility differs from offline audit")
    return pair, retrieval


def candidate_pool(user, variant_data, catalog):
    ordered, seen = [], set()
    def add(movie):
        movie_id = movie.get("movie_id", movie.get("id"))
        if movie_id not in seen:
            ordered.append(catalog[movie_id]); seen.add(movie_id)
    for movie in user["history"]: add(movie)
    add(user["target"])
    for name in VARIANTS:
        for movie in variant_data[name]["retrieval"]["candidates"]: add(movie)
    remaining = [x for movie_id, x in catalog.items() if movie_id not in seen]
    random.Random(SEED + user["user_id"]).shuffle(remaining)
    for movie in remaining:
        if len(ordered) == POOL_SIZE: break
        add(movie)
    if len(ordered) != POOL_SIZE or len({x["movie_id"] for x in ordered}) != POOL_SIZE:
        raise RuntimeError("Shared candidate pool must contain 100 unique items")
    return ordered


def load_inputs(check_service=False):
    verify_sources()
    manifest = read_json(PILOT)
    if manifest["manifest_sha256"] != EXPECTED_MANIFEST_SHA or payload_sha256(manifest) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("Frozen pilot manifest mismatch")
    generation.preflight(check_service=check_service)
    audit = read_json(AUDIT)
    if audit["user_ids"] != manifest["pilot_user_ids"]:
        raise RuntimeError("Offline audit does not use frozen pilot")
    catalog = {x["movie_id"]: x for x in load_catalog()}
    manifest_users = {x["user_id"]: x for x in manifest["users"]}
    audit_users = {x["user_id"]: x for x in audit["users"]}
    prepared = {}
    for uid in USERS:
        user, audited = manifest_users[uid], audit_users[uid]
        variants = {}
        for name in VARIANTS:
            selected = audited["variants"][name]["selected"]
            pair, retrieval = method_bridge(user, selected, catalog)
            variants[name] = {"selected": selected, "pair": pair, "retrieval": retrieval}
        pool = candidate_pool(user, variants, catalog)
        grounded = generation.user_prompt(user) + grounding_context(pool)
        for name in VARIANTS:
            item = variants[name]
            augmented = build_bridge_prompt(generation.SYSTEM_PROMPT, grounded, item["pair"],
                                             "direct", item["retrieval"]["candidates"])
            variants[name]["prompt"] = {**augmented, "formatting_user_prompt": generation.FORMAT_PROMPT,
                "selected_bridge": item["pair"], "bridge_type": "direct",
                "bridge_candidates": item["retrieval"]["candidates"]}
            variants[name]["prompt"]["prompt_hash"] = canonical_hash({k: variants[name]["prompt"][k]
                for k in ("system_prompt", "user_prompt", "formatting_user_prompt")})
        prepared[uid] = {"user": user, "variants": variants, "pool": pool,
                         "pool_ids": [x["movie_id"] for x in pool]}
    return prepared


def slots(prepared):
    return [(prepared[uid], variant, index) for uid in USERS for variant in VARIANTS for index in (1, 2)]


def generate(prepared):
    OUT.mkdir(parents=True, exist_ok=True)
    state = read_json(GEN_STATE) if GEN_STATE.exists() else {
        "experiment": "Bridge Score Generation Ablation", "users": USERS, "variants": list(VARIANTS),
        "paths_per_user_per_variant": 2, "expected_path_count": 24,
        "candidate_pool_size": POOL_SIZE, "pool_identical_across_variants": {str(x): True for x in USERS},
        "candidate_pools": {str(uid): prepared[uid]["pool"] for uid in USERS}, "records": [], "complete": False}
    if state["complete"]:
        return state
    complete = {(x["user_id"], x["method"], x["path_index"]) for x in state["records"]}
    old_out = generation.OUT
    generation.OUT = OUT / "raw_generation"
    try:
        for ordinal, (data, variant, index) in enumerate(slots(prepared), 1):
            user = data["user"]
            key = user["user_id"], variant, index
            if key in complete: continue
            record = generation.record_for(user, variant, index, data["variants"][variant]["prompt"])
            record["record_version"] = "Bridge-Score-Generation-Ablation-v1"
            record["score_variant"] = variant
            record["catalog_candidate_pool_ids"] = data["pool_ids"]
            state["records"].append(record)
            write_json(GEN_STATE, state)
            print(f"[{ordinal}/24] user={user['user_id']} variant={variant} path={index} parse={record['parse_success']}", flush=True)
    finally:
        generation.OUT = old_out
    if len(state["records"]) != 24:
        raise RuntimeError("Exactly 24 generation records required")
    state["complete"] = True
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(GEN_STATE, state)
    return state


def evaluate(prepared, records):
    from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
    from repro.evaluators.sasrec.data_adapter import DataAdapter
    from repro.evaluators.sasrec.evaluator import SASRecEvaluator
    model, dataset, _, _ = load_checkpoint()
    evaluator, adapter = SASRecEvaluator(model), DataAdapter(dataset)
    resolver = TitleResolver.from_movies_dat(ROOT / "dataset/ml-1m/movies.dat")
    outputs = []
    for record in records:
        data, user = prepared[record["user_id"]], prepared[record["user_id"]]["user"]
        target = resolver.resolve(user["target"]["title"])
        history_raw = [x["movie_id"] for x in user["history"]]
        history = adapter.history_raw_ids_to_internal(history_raw)
        candidates = {x["id"] for x in data["variants"][record["method"]]["retrieval"]["candidates"]}
        item = prepare_path({"path": record["parsed_path"], "parse_success": record["parse_success"]},
                            resolver, target, set(history_raw), candidates)
        pool_ids = set(data["pool_ids"])
        violations = [x["raw_title"] for x in item["evaluation_intermediates"]
                      if x["resolution_status"] != "resolved" or x["raw_item_id"] not in pool_ids]
        compliant = item["validity"]["parse_success"] and not violations
        if item["FORMAL_METRIC_STATUS"] == "valid" and not compliant:
            item["FORMAL_METRIC_STATUS"] = "invalid_catalog_violation"; item["STRICT_EVALUATION_VALID"] = False
        scored, _ = evaluate_prepared(item, target, history, adapter, evaluator, diagnostic_drop=False)
        m, v = scored["metrics"], scored["validity"]
        outputs.append({"user_id": record["user_id"], "variant": record["method"], "path_index": record["path_index"],
            "parse_success": v["parse_success"], "formal_metric_status": scored["FORMAL_METRIC_STATUS"],
            "catalog_compliant": compliant, "catalog_violations": violations, "target_last": v["target_is_last"],
            "IoI": m["IoI"], "IoR": m["IoR"], "ProxyAcceptability": m["ProxyAcceptability"],
            "Coherence": m["Coherence"], "PHRR": scored["mechanism_metrics"]["HISTORY_REUSE_RATE"],
            "HistoryReuseRate": scored["mechanism_metrics"]["HISTORY_REUSE_RATE"],
            "NewIntermediateCount": scored["mechanism_metrics"]["NEW_INTERMEDIATE_COUNT"],
            "BridgeCandidateUsageCount": scored["mechanism_metrics"]["BRIDGE_CANDIDATE_USAGE_COUNT"]})
    return outputs


def avg(values):
    values = [x for x in values if x is not None]
    return mean(values) if values else None


def aggregate(outputs):
    metrics = ("IoI", "IoR", "ProxyAcceptability", "Coherence", "PHRR", "HistoryReuseRate",
               "NewIntermediateCount", "BridgeCandidateUsageCount")
    users = []
    for uid in USERS:
        for variant in VARIANTS:
            group = [x for x in outputs if x["user_id"] == uid and x["variant"] == variant]
            valid = [x for x in group if x["formal_metric_status"] == "valid"]
            row = {"user_id": uid, "variant": variant, "ValidPathRate": len(valid)/2,
                   "CatalogComplianceRate": sum(x["catalog_compliant"] for x in group)/2,
                   "TargetLastRate": avg([float(x["target_last"]) for x in group if x["parse_success"]])}
            row.update({m: avg([x[m] for x in valid]) for m in metrics})
            users.append(row)
    aggregate_rows = []
    for variant in VARIANTS:
        group = [x for x in users if x["variant"] == variant]
        aggregate_rows.append({"Variant": variant, **{m: avg([x[m] for x in group]) for m in
            ("ValidPathRate", "CatalogComplianceRate", "IoI", "IoR", "ProxyAcceptability", "Coherence",
             "TargetLastRate", "PHRR", "HistoryReuseRate", "NewIntermediateCount", "BridgeCandidateUsageCount")}})
    return users, aggregate_rows


def paired(users):
    pairs = (("need", "affinity"), ("need", "balance"), ("affinity", "balance"))
    metrics = ("IoI", "IoR", "ProxyAcceptability", "Coherence", "PHRR", "TargetLastRate")
    lookup = {(x["user_id"], x["variant"]): x for x in users}
    rows = []
    for left, right in pairs:
        for metric in metrics:
            comparable = [(lookup[uid, left][metric], lookup[uid, right][metric]) for uid in USERS
                          if lookup[uid, left][metric] is not None and lookup[uid, right][metric] is not None]
            right_better = sum((b < a if metric == "PHRR" else b > a) for a, b in comparable)
            left_better = sum((a < b if metric == "PHRR" else a > b) for a, b in comparable)
            rows.append({"Left": left, "Right": right, "Metric": metric, "ComparableUsers": len(comparable),
                         "LeftImprovedUsers": left_better, "RightImprovedUsers": right_better,
                         "TiedUsers": len(comparable)-left_better-right_better})
    return rows


def save_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def result_label(aggregates, pair_rows):
    best_ioi = max(aggregates, key=lambda x: float("-inf") if x["IoI"] is None else x["IoI"])["Variant"]
    best_ior = max(aggregates, key=lambda x: float("-inf") if x["IoR"] is None else x["IoR"])["Variant"]
    if best_ioi != best_ior: return "MIXED"
    if best_ioi == "need": return "NEED_CANDIDATE_WINNER"
    comparisons = [x for x in pair_rows if {x["Left"], x["Right"]} == {"need", best_ioi} and x["Metric"] in ("IoI", "IoR")]
    wins = [x["RightImprovedUsers"] if x["Right"] == best_ioi else x["LeftImprovedUsers"] for x in comparisons]
    return (best_ioi.upper() + "_CANDIDATE_WINNER") if all(x > 2 for x in wins) else "NO_CLEAR_WINNER"


def finish(prepared, state, protected_before):
    outputs = evaluate(prepared, state["records"])
    users, aggregates = aggregate(outputs)
    pairs = paired(users)
    save_csv(OUT / "path_level_metrics.csv", outputs)
    save_csv(OUT / "user_level_metrics.csv", users)
    save_csv(OUT / "aggregate_metrics.csv", aggregates)
    save_csv(OUT / "paired_comparison.csv", pairs)
    pools = {str(uid): {"pool_size": len(prepared[uid]["pool"]), "movie_ids": prepared[uid]["pool_ids"],
        "pool_identical_across_variants": True,
        "selected_bridges": {v: prepared[uid]["variants"][v]["pair"] for v in VARIANTS}}
        for uid in USERS}
    write_json(OUT / "candidate_pool_audit.json", {"all_pools_identical_across_variants": True, "users": pools})
    label = result_label(aggregates, pairs)
    summary = ["# Bridge-Score Generation Ablation", "",
        "Exploratory scoring-function ablation on four discriminative frozen users; not an unbiased population estimate.", "",
        f"- Result: `{label}`", "- Paths: 24", "- Shared catalog pool: 100 items per user", "- Significance tests: none", "",
        "The predeclared hierarchy treats IoI and IoR as primary and structure, validity, compliance, PHRR, and proxy acceptability as guardrails. No formula is promoted from this exploratory result without held-out validation."]
    (OUT / "SCORE_ABLATION_SUMMARY.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    if protected_snapshot() != protected_before or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen state changed")
    verify_sources()
    print("BRIDGE_SCORE_GENERATION_ABLATION_COMPLETE=True")
    print("USERS=4\nVARIANTS=3\nPATHS_PER_USER_PER_VARIANT=2\nTOTAL_PATHS_EXPECTED=24")
    print(f"TOTAL_PATHS_GENERATED={len(state['records'])}")
    print("ALL_POOLS_IDENTICAL_ACROSS_VARIANTS=True")
    print("RESULT=" + label)


def run():
    if OUT.exists() and not GEN_STATE.exists():
        raise FileExistsError("Output directory exists without resumable generation state")
    prepared = load_inputs(check_service=True)
    protected_before = protected_snapshot()
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or PROTOCOL_VERSION != "Formal-Evaluation-v1":
        raise RuntimeError("Frozen evaluator/checkpoint mismatch")
    state = generate(prepared)
    required = [OUT / x for x in ("path_level_metrics.csv", "user_level_metrics.csv", "aggregate_metrics.csv",
                                  "paired_comparison.csv", "candidate_pool_audit.json", "SCORE_ABLATION_SUMMARY.md")]
    if any(x.exists() for x in required):
        raise FileExistsError("Evaluation output already exists; refusing overwrite")
    finish(prepared, state, protected_before)


if __name__ == "__main__":
    run()

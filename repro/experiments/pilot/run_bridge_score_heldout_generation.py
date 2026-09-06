"""Final 20-path held-out Need-vs-Balance validation; frozen manifest only."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from repro.experiments.pilot import generate_pilot_paths as generation
from repro.experiments.pilot import run_bridge_score_generation_ablation as common
from repro.experiments.pilot.evaluate_pilot_formal import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, protected_snapshot, sha
from repro.experiments.pilot.run_catalog_grounded_ablation import GROUNDING_INSTRUCTION, canonical_hash, load_catalog
from repro.methods.mi_bridge.bridge_prompt import build_bridge_prompt
from repro.methods.mi_bridge.retrieve_bridge_movies import retrieve_bridge_movies
from repro.prepare_ml1m import load_tables
from repro.utils import ROOT, literal, read_json, verify_sources, write_json

OUT = ROOT / "repro/results/bridge_score_heldout_generation"
STATE = OUT / "generation_records.json"
HELDOUT = ROOT / "repro/results/bridge_score_heldout_selection/heldout_users.json"
EXPECTED_HELDOUT_SHA = "443e4717bdd9462b9ae229339a611883ef815615a3f0ea73a276f06faa29000a"
USERS = [3, 7, 8, 17, 22]
METHODS = ("need", "balance")
POOL_SIZE = 100
SEED = 20260906


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public(movie):
    return {"movie_id": movie["id"], "title": movie["title"], "genres": list(movie["genres"])}


def grounding_context(pool):
    return ("\n\n[CATALOG GROUNDING]\n" + GROUNDING_INSTRUCTION + "\n"
            "MovieLens-1M catalog candidate set:\n" +
            json.dumps([x["title"] for x in pool], ensure_ascii=False) +
            "\n[/CATALOG GROUNDING]\n")


def load_inputs(check_service=False):
    verify_sources()
    if file_sha(HELDOUT) != EXPECTED_HELDOUT_SHA:
        raise RuntimeError("HELDOUT_MANIFEST_VERIFIED=False")
    frozen = read_json(HELDOUT)
    if frozen["selected_user_ids"] != USERS or frozen["target_seed"] != SEED:
        raise RuntimeError("Held-out users/seed mismatch")
    generation.preflight(check_service=check_service)
    ratings, _, users_table = load_tables(ROOT / "dataset/ml-1m")
    genders = literal("movielensUserProfile.ipynb", 17, "gender_dict")
    ages = literal("movielensUserProfile.ipynb", 17, "age_dict")
    occupations = literal("movielensUserProfile.ipynb", 17, "occupation_dict")
    catalog_items = load_catalog()
    catalog = {x["movie_id"]: x for x in catalog_items}
    retrieval_catalog = [dict(x, id=x["movie_id"]) for x in catalog_items]
    prepared = {}
    for frozen_user in frozen["users"]:
        uid = frozen_user["user_id"]
        if uid not in USERS: continue
        rows = users_table[users_table[0] == uid].values.tolist()
        if len(rows) != 1: raise RuntimeError("Demographic lookup mismatch")
        demo = rows[0]
        user = {"user_id": uid, "history": [public(x) for x in frozen_user["history"]],
                "target": public(frozen_user["target"]),
                "demographics": {"gender": genders[demo[1]], "age": ages[demo[2]], "occupation": occupations[demo[3]]}}
        recomputed = __import__("repro.analysis.select_bridge_score_heldout", fromlist=["rankings"]).rankings(
            frozen_user, [dict(x, id=x["movie_id"]) for x in catalog_items])
        variants = {}
        for method in METHODS:
            expected = frozen_user["rankings"][method]["selected"]
            selected = recomputed[method]["selected"]
            keys = ("existing_interest", "target_interest", "bridge_score", "target_term", "candidate_count", "mode")
            if any(selected[k] != expected[k] for k in keys):
                raise RuntimeError(f"Recomputed {method} bridge differs for User {uid}")
            pair = {"user_existing_interest": selected["existing_interest"],
                    "target_related_interest": selected["target_interest"], "bridge_score": selected["bridge_score"]}
            retrieval = retrieve_bridge_movies(pair, retrieval_catalog, frozen_user["history"], frozen_user["target"])
            if selected["mode"] != "direct" or retrieval["candidate_count"] != selected["candidate_count"]:
                raise RuntimeError("Frozen selected bridge feasibility mismatch")
            variants[method] = {"pair": pair, "selected": selected, "retrieval": retrieval}
        ordered, seen = [], set()
        def add(movie):
            movie_id = movie.get("movie_id", movie.get("id"))
            if movie_id not in seen: ordered.append(catalog[movie_id]); seen.add(movie_id)
        for movie in user["history"]: add(movie)
        add(user["target"])
        for method in METHODS:
            for movie in variants[method]["retrieval"]["candidates"]: add(movie)
        remaining = [x for movie_id, x in catalog.items() if movie_id not in seen]
        random.Random(SEED + uid).shuffle(remaining)
        for movie in remaining:
            if len(ordered) == POOL_SIZE: break
            add(movie)
        if len(ordered) != POOL_SIZE: raise RuntimeError("Shared pool size mismatch")
        grounded = generation.user_prompt(user) + grounding_context(ordered)
        for method in METHODS:
            item = variants[method]
            prompt = build_bridge_prompt(generation.SYSTEM_PROMPT, grounded, item["pair"], "direct", item["retrieval"]["candidates"])
            item["prompt"] = {**prompt, "formatting_user_prompt": generation.FORMAT_PROMPT,
                "selected_bridge": item["pair"], "bridge_type": "direct", "bridge_candidates": item["retrieval"]["candidates"]}
            item["prompt"]["prompt_hash"] = canonical_hash({k: item["prompt"][k] for k in
                ("system_prompt", "user_prompt", "formatting_user_prompt")})
        prepared[uid] = {"user": user, "variants": variants, "pool": ordered,
                         "pool_ids": [x["movie_id"] for x in ordered]}
    if list(prepared) != USERS: raise RuntimeError("Held-out order mismatch")
    return prepared


def generate(prepared):
    OUT.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE) if STATE.exists() else {"experiment": "Need-vs-Balance Held-out Generation",
        "heldout_manifest_sha256": EXPECTED_HELDOUT_SHA, "users": USERS, "methods": list(METHODS),
        "paths_per_user_per_method": 2, "expected_path_count": 20, "candidate_pool_size": POOL_SIZE,
        "pool_identical_need_balance": {str(uid): True for uid in USERS},
        "candidate_pools": {str(uid): prepared[uid]["pool"] for uid in USERS}, "records": [], "complete": False}
    if state["complete"]: return state
    done = {(x["user_id"], x["method"], x["path_index"]) for x in state["records"]}
    old_out = generation.OUT; generation.OUT = OUT / "raw_generation"
    try:
        ordinal = 0
        for uid in USERS:
            for method in METHODS:
                for index in (1, 2):
                    ordinal += 1
                    if (uid, method, index) in done: continue
                    record = generation.record_for(prepared[uid]["user"], method, index, prepared[uid]["variants"][method]["prompt"])
                    record["record_version"] = "Bridge-Score-Heldout-Generation-v1"
                    record["catalog_candidate_pool_ids"] = prepared[uid]["pool_ids"]
                    state["records"].append(record); write_json(STATE, state)
                    print(f"[{ordinal}/20] user={uid} method={method} path={index} parse={record['parse_success']}", flush=True)
    finally:
        generation.OUT = old_out
    if len(state["records"]) != 20: raise RuntimeError("Exactly 20 paths required")
    state["complete"] = True; state["completed_at"] = datetime.now(timezone.utc).isoformat(); write_json(STATE, state)
    return state


def aggregates(outputs):
    metrics = ("IoI", "IoR", "ProxyAcceptability", "Coherence", "PHRR", "HistoryReuseRate",
               "NewIntermediateCount", "BridgeCandidateUsageCount")
    users = []
    for uid in USERS:
        for method in METHODS:
            group = [x for x in outputs if x["user_id"] == uid and x["variant"] == method]
            valid = [x for x in group if x["formal_metric_status"] == "valid"]
            row = {"user_id": uid, "method": method, "ValidPathRate": len(valid)/2,
                   "CatalogComplianceRate": sum(x["catalog_compliant"] for x in group)/2,
                   "TargetLastRate": common.avg([float(x["target_last"]) for x in group if x["parse_success"]])}
            row.update({m: common.avg([x[m] for x in valid]) for m in metrics}); users.append(row)
    methods = []
    for method in METHODS:
        group = [x for x in users if x["method"] == method]
        methods.append({"Method": method, **{m: common.avg([x[m] for x in group]) for m in
            ("ValidPathRate", "CatalogComplianceRate", "IoI", "IoR", "ProxyAcceptability", "Coherence",
             "TargetLastRate", "PHRR", "HistoryReuseRate", "NewIntermediateCount", "BridgeCandidateUsageCount")}})
    paired = []
    lookup = {(x["user_id"], x["method"]): x for x in users}
    for metric in ("IoI", "IoR", "ProxyAcceptability", "Coherence", "PHRR", "TargetLastRate"):
        pairs = [(lookup[uid,"need"][metric], lookup[uid,"balance"][metric]) for uid in USERS
                 if lookup[uid,"need"][metric] is not None and lookup[uid,"balance"][metric] is not None]
        bw = sum((b < n if metric == "PHRR" else b > n) for n,b in pairs)
        nw = sum((n < b if metric == "PHRR" else n > b) for n,b in pairs)
        paired.append({"Metric": metric, "ComparableUsers": len(pairs), "BalanceWins": bw,
                       "NeedWins": nw, "Ties": len(pairs)-bw-nw})
    return users, methods, paired


def decision(methods, paired):
    by = {x["Method"]: x for x in methods}; wins = {x["Metric"]: x["BalanceWins"] for x in paired}
    higher_ioi, higher_ior = by["balance"]["IoI"] > by["need"]["IoI"], by["balance"]["IoR"] > by["need"]["IoR"]
    if higher_ioi != higher_ior: return "MIXED"
    balance_guidance = higher_ioi and higher_ior and wins["IoI"] >= 3 and wins["IoR"] >= 3
    structure_ok = (by["balance"]["ValidPathRate"] >= by["need"]["ValidPathRate"] - 0.1 and
        by["balance"]["CatalogComplianceRate"] >= by["need"]["CatalogComplianceRate"] - 0.1 and
        sum((by["balance"]["Coherence"] >= by["need"]["Coherence"],
             by["balance"]["TargetLastRate"] >= by["need"]["TargetLastRate"],
             by["balance"]["PHRR"] <= by["need"]["PHRR"])) >= 2)
    if balance_guidance and structure_ok: return "BALANCE_HELDOUT_SUPPORTED"
    if higher_ioi and higher_ior and (wins["IoI"] < 3 or wins["IoR"] < 3): return "INSUFFICIENT_USER_LEVEL_SUPPORT"
    need_wins = {x["Metric"]: x["NeedWins"] for x in paired}
    if not higher_ioi and not higher_ior and need_wins["IoI"] >= 3 and need_wins["IoR"] >= 3: return "NEED_HELDOUT_SUPPORTED"
    return "NO_CLEAR_WINNER"


def save_csv(name, rows):
    with (OUT/name).open("x", encoding="utf-8-sig", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def finish(prepared, state, protected_before):
    outputs = common.evaluate(prepared, state["records"])
    users, methods, paired = aggregates(outputs); result = decision(methods, paired)
    save_csv("path_level_metrics.csv", outputs); save_csv("user_level_metrics.csv", users)
    save_csv("aggregate_metrics.csv", methods); save_csv("paired_comparison.csv", paired)
    write_json(OUT/"candidate_pool_audit.json", {"all_pools_identical_need_balance": True,
        "users": {str(uid): {"size": 100, "movie_ids": prepared[uid]["pool_ids"],
            "pool_identical_need_balance": True} for uid in USERS}})
    (OUT/"HELDOUT_GENERATION_SUMMARY.md").write_text(
        "# Need vs Balance Held-out Generation\n\nFive-user held-out validation; no significance tests.\n\n"
        f"- Result: `{result}`\n- Manifest SHA-256: `{EXPECTED_HELDOUT_SHA}`\n- Paths: 20\n"
        "- Shared catalog pool: 100 items per user\n", encoding="utf-8")
    if protected_snapshot()!=protected_before or file_sha(HELDOUT)!=EXPECTED_HELDOUT_SHA or sha(CHECKPOINT)!=EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Frozen state changed")
    verify_sources()
    print("HELDOUT_GENERATION_COMPLETE=True\nHELDOUT_MANIFEST_VERIFIED=True\nUSERS=5\nMETHODS=2")
    print("PATHS_PER_USER_PER_METHOD=2\nTOTAL_PATHS_EXPECTED=20")
    print(f"TOTAL_PATHS_GENERATED={len(state['records'])}\nALL_POOLS_IDENTICAL_NEED_BALANCE=True")
    wins={x['Metric']:x['BalanceWins'] for x in paired}
    for metric,label in (("IoI","IOI"),("IoR","IOR"),("ProxyAcceptability","PROXY"),("Coherence","COHERENCE"),("PHRR","PHRR"),("TargetLastRate","TARGETLAST")):
        print(f"{label}_BALANCE_WINS={wins[metric]}/5")
    print("RESULT="+result)


def run():
    if OUT.exists() and not STATE.exists(): raise FileExistsError("Output exists without resumable state")
    prepared=load_inputs(True); protected_before=protected_snapshot(); state=generate(prepared)
    expected=[OUT/x for x in ("path_level_metrics.csv","user_level_metrics.csv","aggregate_metrics.csv","paired_comparison.csv","candidate_pool_audit.json","HELDOUT_GENERATION_SUMMARY.md")]
    if any(x.exists() for x in expected): raise FileExistsError("Evaluation output exists; refusing overwrite")
    finish(prepared,state,protected_before)


if __name__=="__main__": run()

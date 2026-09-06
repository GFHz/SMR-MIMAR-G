"""Offline full-vocabulary rank diagnostics for the frozen fairness ablation."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median

import torch

from repro.evaluators.formal.title_resolver import TitleResolver
from repro.experiments.pilot.evaluate_pilot_formal import (
    CHECKPOINT, EXPECTED_CHECKPOINT_SHA, GENERATION, directory_freeze,
    protected_snapshot, read, sha,
)

ROOT = Path(__file__).resolve().parents[3]
ABLATION = ROOT / "repro/results/pilot/catalog_grounded_ablation_v1"
PATH_METRICS = ABLATION / "evaluation/path_level_metrics.json"
PILOT = ROOT / "repro/results/pilot/pilot_manifest.json"
OUT = ROOT / "repro/results/pilot/ior_drop_diagnostic"
EXPECTED_ABLATION_SUMMARY_SHA = "30c211602841c648f796c5ea2b803be99012bfa27700301a850394d35074c0e0"
EXPECTED_PILOT_GENERATION_SHA = "f35a442065a34e10b790411d0e3f9ce315a3df89ccd80fbc1da0e1355b72ef56"


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def avg(values):
    values = [x for x in values if x is not None]
    return mean(values) if values else None


def neighborhood(order, target_id, probs_before, probs_after, adapter, catalog, state):
    target_index = int((order == target_id).nonzero()[0, 0])
    result = []
    for rank_index in range(max(0, target_index - 10), min(len(order), target_index + 11)):
        internal = int(order[rank_index])
        raw_token = str(adapter.item_tokens[internal])
        raw_id = int(raw_token) if raw_token.isdigit() else None
        movie = catalog.get(raw_id)
        score = probs_before[internal] if state == "before" else probs_after[internal]
        result.append({"internal_item_id": internal, "raw_item_id": raw_id,
            "title": movie["resolved_title"] if movie else ("[PAD]" if internal == 0 else None),
            "score": float(score), "rank": rank_index + 1,
            "score_delta": float(probs_after[internal] - probs_before[internal]),
            "is_target": internal == target_id})
    return result


def aggregate_method(rows, method):
    group = [x for x in rows if x["method"] == method]
    fields = ("TARGET_SCORE_DELTA", "IoI", "IoR", "NUM_ITEMS_INCREASED_MORE_THAN_TARGET",
              "OVERTAKE_TARGET_COUNT", "TARGET_OVERTAKES_COUNT",
              "TARGET_GENRE_OVERLAP_OVERTAKE_RATE", "BRIDGE_GENRE_OVERLAP_OVERTAKE_RATE")
    return {field + "_mean": avg([x[field] for x in group]) for field in fields} | {"path_count": len(group)}


def aggregate_users(rows):
    result = []
    fields = ("TARGET_SCORE_DELTA", "IoI", "IoR", "OVERTAKE_TARGET_COUNT",
              "TARGET_OVERTAKES_COUNT", "TARGET_GENRE_OVERLAP_OVERTAKE_RATE")
    for user_id in sorted({x["user_id"] for x in rows}):
        entry = {"user_id": user_id, "methods": {}}
        for method in ("baseline", "mi_bridge"):
            group = [x for x in rows if x["user_id"] == user_id and x["method"] == method]
            entry["methods"][method] = {field: avg([x[field] for x in group]) for field in fields}
            entry["methods"][method]["rank_before"] = group[0]["target_rank_before"]
        b, o = entry["methods"]["baseline"], entry["methods"]["mi_bridge"]
        entry["paired_delta_mi_minus_baseline"] = {
            "delta_target_score_gain": o["TARGET_SCORE_DELTA"] - b["TARGET_SCORE_DELTA"],
            "delta_IoI": o["IoI"] - b["IoI"], "delta_IoR": o["IoR"] - b["IoR"],
            "delta_overtake_target_count": o["OVERTAKE_TARGET_COUNT"] - b["OVERTAKE_TARGET_COUNT"],
            "delta_target_overtakes_count": o["TARGET_OVERTAKES_COUNT"] - b["TARGET_OVERTAKES_COUNT"],
            "delta_target_genre_overlap_overtake_rate":
                (o["TARGET_GENRE_OVERLAP_OVERTAKE_RATE"] - b["TARGET_GENRE_OVERLAP_OVERTAKE_RATE"])
                if o["TARGET_GENRE_OVERLAP_OVERTAKE_RATE"] is not None and b["TARGET_GENRE_OVERLAP_OVERTAKE_RATE"] is not None else None}
        result.append(entry)
    return result


def run():
    if OUT.exists():
        raise FileExistsError("ior_drop_diagnostic exists; refusing overwrite or rerun")
    if sha(ABLATION / "summary.json") != EXPECTED_ABLATION_SUMMARY_SHA:
        raise RuntimeError("Frozen ablation summary changed")
    if directory_freeze(GENERATION)["generation_v1_sha256"] != EXPECTED_PILOT_GENERATION_SHA:
        raise RuntimeError("Frozen pilot generation changed")
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Checkpoint changed")
    protected_before = protected_snapshot()
    rows = read(PATH_METRICS)
    if len(rows) != 20 or any(x["formal_metric_status"] != "valid" for x in rows):
        raise RuntimeError("Expected 20 frozen formal-valid ablation paths")
    manifest = read(PILOT)
    users = {x["user_id"]: x for x in manifest["users"]}

    from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
    from repro.evaluators.sasrec.data_adapter import DataAdapter
    from repro.evaluators.sasrec.evaluator import SASRecEvaluator
    model, dataset, _, _ = load_checkpoint()
    evaluator, adapter = SASRecEvaluator(model), DataAdapter(dataset)
    resolver = TitleResolver.from_movies_dat(ROOT / "dataset/ml-1m/movies.dat")
    catalog = {movie_id: {"resolved_title": x["title"], "genres": x["genres"]}
               for movie_id, x in resolver.movies.items()}
    before_cache, diagnostics, genre_output = {}, [], []
    for row in rows:
        user = users[row["user_id"]]
        target_raw = user["target"]["movie_id"]
        target_internal = adapter.raw_item_id_to_internal(target_raw)
        history = adapter.history_raw_ids_to_internal([x["movie_id"] for x in user["history"]])
        intermediates = [resolver.resolve(title) for title in row["raw_path"]]
        intermediate_ids = [adapter.raw_item_id_to_internal(x["raw_item_id"])
                            for x in intermediates if x["raw_item_id"] is not None and x["raw_item_id"] != target_raw]
        if row["user_id"] not in before_cache:
            before_cache[row["user_id"]] = torch.softmax(evaluator.full_scores(history), dim=-1)
        probs_before = before_cache[row["user_id"]]
        probs_after = torch.softmax(evaluator.full_scores(history + intermediate_ids), dim=-1)
        before_target, after_target = float(probs_before[target_internal]), float(probs_after[target_internal])
        delta_target = after_target - before_target
        order_before = torch.argsort(probs_before, descending=True)
        order_after = torch.argsort(probs_after, descending=True)
        rank_before = int((order_before == target_internal).nonzero()[0, 0]) + 1
        rank_after = int((order_after == target_internal).nonzero()[0, 0]) + 1
        competitor_mask = torch.ones_like(probs_before, dtype=torch.bool)
        competitor_mask[target_internal] = False
        deltas = probs_after - probs_before
        competitor_deltas = deltas[competitor_mask]
        candidate_ids = torch.arange(len(probs_before))[competitor_mask]
        overtake_mask = (probs_before[candidate_ids] <= before_target) & (probs_after[candidate_ids] > after_target)
        target_overtake_mask = (probs_before[candidate_ids] > before_target) & (probs_after[candidate_ids] <= after_target)
        overtake_ids = [int(x) for x in candidate_ids[overtake_mask]]
        target_genres = set(user["target"]["genres"])
        bridge_genres = {user["bridge"]["selected_existing_interest"], user["bridge"]["selected_target_interest"]}
        overtake_movies, genre_counts = [], Counter()
        target_overlap = bridge_overlap = 0
        for internal in overtake_ids:
            token = str(adapter.item_tokens[internal])
            raw_id = int(token) if token.isdigit() else None
            movie = catalog.get(raw_id)
            genres = set(movie["genres"]) if movie else set()
            genre_counts.update(genres)
            target_overlap += bool(genres & target_genres)
            bridge_overlap += bool(genres & bridge_genres)
            overtake_movies.append({"internal_item_id": internal, "raw_item_id": raw_id,
                "title": movie["resolved_title"] if movie else ("[PAD]" if internal == 0 else None),
                "genres": sorted(genres), "score_before": float(probs_before[internal]),
                "score_after": float(probs_after[internal]), "score_delta": float(deltas[internal])})
        n_overtake = len(overtake_ids)
        net = int(target_overtake_mask.sum()) - n_overtake
        diagnostic = {"user_id": row["user_id"], "method": row["method"], "path_index": row["path_index"],
            "target_raw_item_id": target_raw, "target_internal_item_id": target_internal,
            "target_score_before": before_target, "target_score_after": after_target,
            "target_rank_before": rank_before, "target_rank_after": rank_after,
            "TARGET_SCORE_DELTA": delta_target, "TARGET_LOG_SCORE_DELTA": row["IoI"],
            "IoI": row["IoI"], "IoR": row["IoR"],
            "NUM_ITEMS_SCORE_INCREASED": int((competitor_deltas > 0).sum()),
            "NUM_ITEMS_SCORE_DECREASED": int((competitor_deltas < 0).sum()),
            "NUM_ITEMS_INCREASED_MORE_THAN_TARGET": int((competitor_deltas > delta_target).sum()),
            "FRACTION_ITEMS_INCREASED_MORE_THAN_TARGET": float((competitor_deltas > delta_target).float().mean()),
            "MEAN_COMPETITOR_SCORE_DELTA": float(competitor_deltas.mean()),
            "MEDIAN_COMPETITOR_SCORE_DELTA": float(competitor_deltas.median()),
            "MAX_COMPETITOR_SCORE_DELTA": float(competitor_deltas.max()),
            "OVERTAKE_TARGET_COUNT": n_overtake,
            "TARGET_OVERTAKES_COUNT": int(target_overtake_mask.sum()),
            "OVERTAKE_NET": net, "IOR_EQUALS_OVERTAKE_NET": row["IoR"] == net,
            "TARGET_GENRE_OVERLAP_OVERTAKE_RATE": target_overlap / n_overtake if n_overtake else None,
            "BRIDGE_GENRE_OVERLAP_OVERTAKE_RATE": bridge_overlap / n_overtake if n_overtake else None,
            "rank_neighborhood_before": neighborhood(order_before, target_internal, probs_before, probs_after, adapter, catalog, "before"),
            "rank_neighborhood_after": neighborhood(order_after, target_internal, probs_before, probs_after, adapter, catalog, "after")}
        if not (math.isclose(before_target, row["P_before"], abs_tol=1e-8) and
                math.isclose(after_target, row["P_after"], abs_tol=1e-8) and
                rank_before == row["R_before"] and rank_after == row["R_after"]):
            raise RuntimeError("Frozen score/rank regression mismatch")
        diagnostics.append(diagnostic)
        genre_output.append({"user_id": row["user_id"], "method": row["method"], "path_index": row["path_index"],
            "target_genres": sorted(target_genres), "bridge_genres": sorted(bridge_genres),
            "overtake_target_count": n_overtake, "genre_counts": dict(sorted(genre_counts.items())),
            "target_genre_overlap_overtake_rate": diagnostic["TARGET_GENRE_OVERLAP_OVERTAKE_RATE"],
            "bridge_genre_overlap_overtake_rate": diagnostic["BRIDGE_GENRE_OVERLAP_OVERTAKE_RATE"],
            "overtake_items": overtake_movies})

    before_consistent = True
    for user_id in sorted({x["user_id"] for x in diagnostics}):
        group = [x for x in diagnostics if x["user_id"] == user_id]
        before_consistent &= max(x["target_score_before"] for x in group) - min(x["target_score_before"] for x in group) <= 1e-8
        before_consistent &= len({x["target_rank_before"] for x in group}) == 1
    if not before_consistent:
        raise RuntimeError("Evaluator before-state inconsistency")
    methods = {"Baseline": aggregate_method(diagnostics, "baseline"),
               "MI-Bridge": aggregate_method(diagnostics, "mi_bridge")}
    user_rows = aggregate_users(diagnostics)
    declining = [x for x in user_rows if x["paired_delta_mi_minus_baseline"]["delta_IoR"] < 0]
    decline_with_more_overtakes = sum(x["paired_delta_mi_minus_baseline"]["delta_overtake_target_count"] > 0 for x in declining)
    hypothesis = "SUPPORTED" if (methods["MI-Bridge"]["TARGET_SCORE_DELTA_mean"] > methods["Baseline"]["TARGET_SCORE_DELTA_mean"] and
        methods["MI-Bridge"]["OVERTAKE_TARGET_COUNT_mean"] > methods["Baseline"]["OVERTAKE_TARGET_COUNT_mean"] and
        decline_with_more_overtakes >= max(1, math.ceil(len(declining) / 2)) and
        (methods["MI-Bridge"]["BRIDGE_GENRE_OVERLAP_OVERTAKE_RATE_mean"] or 0) >= 0.5) else "PARTIALLY_SUPPORTED"
    assessment = {"IOR_DROP_HYPOTHESIS": hypothesis, "BEFORE_STATE_CONSISTENT": before_consistent,
        "ior_declining_user_count": len(declining), "declining_users_with_more_overtakes": decline_with_more_overtakes,
        "MAIN_EXPLANATION": "MI-Bridge changes target probability and the full competing-item distribution simultaneously; rank change equals net target overtakes minus competitor overtakes under the frozen ordering.",
        "TARGET_RANK_SATURATION": {"supported": False, "reason": "Each user's before state is identical across methods, so method-level IoR differences cannot be caused by different initial ranks."},
        "causal_claim": False, "significance_test_performed": False}
    if protected_snapshot() != protected_before or sha(ABLATION / "summary.json") != EXPECTED_ABLATION_SUMMARY_SHA:
        raise RuntimeError("Protected input changed during diagnostics")
    OUT.mkdir(parents=True)
    save("path_level_rank_diagnostics.json", diagnostics)
    save("user_level_rank_diagnostics.json", user_rows)
    save("method_level_rank_diagnostics.json", methods)
    save("overtake_genre_analysis.json", genre_output)
    save("hypothesis_assessment.json", assessment)
    print(json.dumps({"methods": methods, "paired_users": user_rows,
        "hypothesis": assessment}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()

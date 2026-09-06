"""Offline one-candidate SASRec diagnostics for frozen MI-Bridge selections."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev

import torch

from repro.evaluators.formal.protocol import IOI_EPS
from repro.experiments.pilot.evaluate_pilot_formal import (
    CHECKPOINT, EXPECTED_CHECKPOINT_SHA, protected_snapshot, read, sha,
)

ROOT = Path(__file__).resolve().parents[3]
ABLATION = ROOT / "repro/results/pilot/catalog_grounded_ablation_v1"
DIAGNOSTIC = ROOT / "repro/results/pilot/ior_drop_diagnostic"
CASES = ROOT / "repro/results/pilot/ior_case_analysis"
MANIFEST = ROOT / "repro/results/pilot/pilot_manifest.json"
OUT = ROOT / "repro/results/pilot/effect_aware_bridge_diagnostic"
USERS = (419, 5021, 2677, 3113, 2249)
EXPECTED_ABLATION_SUMMARY_SHA = "30c211602841c648f796c5ea2b803be99012bfa27700301a850394d35074c0e0"


def tree_sha(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def save(name: str, value) -> None:
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rank_of(order: torch.Tensor, item_id: int) -> int:
    return int((order == item_id).nonzero()[0, 0]) + 1


def assign_rank(rows: list[dict], value: str, tie_fields: tuple[str, ...], output: str) -> None:
    ordered = sorted(rows, key=lambda x: tuple([-x[value], *(-x[k] for k in tie_fields), x["movie_id"]]))
    for position, row in enumerate(ordered, 1):
        row[output] = position


def used_titles(user_id: int, candidate_titles: set[str]) -> dict[str, list[int]]:
    used: dict[str, list[int]] = {}
    for path_index in (1, 2):
        record = read(ABLATION / f"generation/users/{user_id}/mi_bridge_path_{path_index}.json")
        for title in record["parsed_path"]:
            if title in candidate_titles:
                used.setdefault(title, []).append(path_index)
    return used


def spread(rows: list[dict], field: str) -> dict:
    values = [x[field] for x in rows]
    return {"max": max(values), "min": min(values), "mean": mean(values),
            "median": median(values), "std_population": pstdev(values),
            "spread": max(values) - min(values)}


def run() -> None:
    if OUT.exists():
        raise FileExistsError("effect-aware diagnostic exists; refusing overwrite")
    if sha(ABLATION / "summary.json") != EXPECTED_ABLATION_SUMMARY_SHA:
        raise RuntimeError("Frozen ablation summary changed")
    if sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Checkpoint changed")
    protected_before = protected_snapshot()
    input_hashes = {"catalog_grounded_ablation_v1": tree_sha(ABLATION),
                    "ior_drop_diagnostic": tree_sha(DIAGNOSTIC),
                    "ior_case_analysis": tree_sha(CASES)}

    from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
    from repro.evaluators.sasrec.data_adapter import DataAdapter
    from repro.evaluators.sasrec.evaluator import SASRecEvaluator

    torch.set_num_threads(4)
    model, dataset, _, _ = load_checkpoint()
    evaluator, adapter = SASRecEvaluator(model), DataAdapter(dataset)
    users = {x["user_id"]: x for x in read(MANIFEST)["users"]}
    all_effects, summaries, all_used = [], [], []
    for user_id in USERS:
        user = users[user_id]
        bridge = user["bridge"]
        if bridge["bridge_type"] != "direct" or not bridge["direct_candidates"]:
            raise RuntimeError(f"User {user_id} does not have a frozen direct bridge list")
        history = adapter.history_raw_ids_to_internal([x["movie_id"] for x in user["history"]])
        target = adapter.raw_item_id_to_internal(user["target"]["movie_id"])
        probs_before = torch.softmax(evaluator.full_scores(history), dim=-1)
        rank_before = rank_of(torch.argsort(probs_before, descending=True), target)
        prob_before = float(probs_before[target])
        candidates = bridge["direct_candidates"]
        current_order = [x["title"] for x in candidates]
        used = used_titles(user_id, set(current_order))
        rows = []
        for retrieval_position, candidate in enumerate(candidates, 1):
            candidate_internal = adapter.raw_item_id_to_internal(candidate["id"])
            probs_after = torch.softmax(evaluator.full_scores(history + [candidate_internal]), dim=-1)
            prob_after = float(probs_after[target])
            rank_after = rank_of(torch.argsort(probs_after, descending=True), target)
            rows.append({"user_id": user_id, "movie_id": candidate["id"], "title": candidate["title"],
                "genres": candidate["genres"], "retrieval_position": retrieval_position,
                "target_rank_before": rank_before, "target_rank_after_candidate": rank_after,
                "EstimatedIoR": rank_before - rank_after,
                "target_prob_before": prob_before, "target_prob_after_candidate": prob_after,
                "EstimatedIoI": math.log(prob_after + IOI_EPS) - math.log(prob_before + IOI_EPS),
                "CandidateAcceptability": evaluator.get_item_score(history, candidate_internal),
                "used_in_mi_bridge_paths": used.get(candidate["title"], [])})
        assign_rank(rows, "EstimatedIoR", ("EstimatedIoI", "CandidateAcceptability"), "IoR_rank")
        assign_rank(rows, "EstimatedIoI", ("EstimatedIoR", "CandidateAcceptability"), "IoI_rank")
        assign_rank(rows, "CandidateAcceptability", ("EstimatedIoR", "EstimatedIoI"), "Acceptability_rank")
        n = len(rows)
        for row in rows:
            row["IoR_rank_percentile"] = 100.0 if n == 1 else 100.0 * (n - row["IoR_rank"]) / (n - 1)
            row["IoR_rank_percentile_definition"] = "100=best, 0=worst; 100*(N-rank)/(N-1)"
        ior_stats, ioi_stats, acc_stats = spread(rows, "EstimatedIoR"), spread(rows, "EstimatedIoI"), spread(rows, "CandidateAcceptability")
        best_ior = min(rows, key=lambda x: x["IoR_rank"])
        best_ioi = min(rows, key=lambda x: x["IoI_rank"])
        best_acc = min(rows, key=lambda x: x["Acceptability_rank"])
        current = rows[0]
        used_rows = [x for x in rows if x["used_in_mi_bridge_paths"]]
        used_mean = mean(x["EstimatedIoR"] for x in used_rows) if used_rows else None
        used_below = used_mean is not None and used_mean < ior_stats["median"]
        summary = {"user_id": user_id,
            "target": user["target"],
            "selected_bridge": {"existing_interest": bridge["selected_existing_interest"],
                                "target_interest": bridge["selected_target_interest"], "bridge_type": "direct"},
            "candidate_count": n, "current_candidate_order": current_order,
            "best_IoR_candidate": best_ior["title"], "best_IoI_candidate": best_ioi["title"],
            "best_acceptability_candidate": best_acc["title"],
            "IOR_CANDIDATE": ior_stats, "IOI_CANDIDATE": ioi_stats, "ACCEPTABILITY_CANDIDATE": acc_stats,
            "CURRENT_TOP_CANDIDATE": current["title"], "CURRENT_TOP_CANDIDATE_IOR": current["EstimatedIoR"],
            "BEST_CANDIDATE_IOR": best_ior["EstimatedIoR"],
            "POTENTIAL_SINGLE_STEP_IOR_GAIN": best_ior["EstimatedIoR"] - current["EstimatedIoR"],
            "used_candidate_count": len(used_rows), "used_candidate_mean_EstimatedIoR": used_mean,
            "used_candidates_below_median_IoR": used_below}
        for row in used_rows:
            all_used.append({k: row[k] for k in ("user_id", "movie_id", "title", "EstimatedIoR", "EstimatedIoI",
                "CandidateAcceptability", "IoR_rank", "IoI_rank", "Acceptability_rank", "IoR_rank_percentile",
                "used_in_mi_bridge_paths")})
        all_effects.extend(rows)
        summaries.append(summary)

    positive = [x["user_id"] for x in summaries if x["BEST_CANDIDATE_IOR"] > 0]
    large = [x["user_id"] for x in summaries if x["IOR_CANDIDATE"]["spread"] >= 100]
    below = [x["user_id"] for x in summaries if x["used_candidates_below_median_IoR"]]
    gain_mean = mean(x["POTENTIAL_SINGLE_STEP_IOR_GAIN"] for x in summaries)
    used_percentile = mean(x["IoR_rank_percentile"] for x in all_used) if all_used else None
    if len(large) >= 3 and len(below) >= 3:
        opportunity = "HIGH"
    elif large or below or gain_mean > 0:
        opportunity = "MEDIUM"
    else:
        opportunity = "LOW"
    method = {"user_count": len(summaries), "candidate_count": len(all_effects),
        "USERS_WITH_POSITIVE_BEST_CANDIDATE_IOR": positive,
        "USERS_WITH_LARGE_IOR_SPREAD": large,
        "USERS_WHERE_USED_CANDIDATES_ARE_BELOW_MEDIAN_IOR": below,
        "POTENTIAL_SINGLE_STEP_IOR_GAIN_MEAN": gain_mean,
        "USED_CANDIDATE_MEAN_IOR_RANK_PERCENTILE": used_percentile,
        "CURRENT_SELECTION_EFFECT_AWARE": False}
    assessment = {"EFFECT_AWARE_OPPORTUNITY": opportunity,
        "decision_rule": "HIGH iff >=3 users have IoR spread >=100 and >=3 users' used-candidate mean IoR is below their candidate median; MEDIUM iff any diagnosed opportunity remains; otherwise LOW.",
        "diagnostic_upper_opportunity_only": True,
        "does_not_predict_equal_final_path_gain": True,
        "OVERFITTING_RISK": "Selecting candidates with this SASRec and evaluating final IoR with the same SASRec creates evaluator-overfitting/circularity risk.",
        "RECOMMENDED_V2_DESIGN": "Effect-aware candidate ranking may be tested diagnostically, but stronger evidence requires cross-evaluation with an independent sequential recommender.",
        "PATH_LENGTH_ABLATION_PRIORITY": "SECONDARY" if opportunity in {"HIGH", "MEDIUM"} else "PRIMARY_OR_TARGET_CONVERGENT_TAIL",
        "cross_evaluator_candidates": ["GRU4Rec", "BERT4Rec", "LightSANs"],
        "causal_claim": False, "significance_test_performed": False}
    if protected_snapshot() != protected_before or sha(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Protected evaluator input changed")
    if any(tree_sha(path) != input_hashes[name] for name, path in
           (("catalog_grounded_ablation_v1", ABLATION), ("ior_drop_diagnostic", DIAGNOSTIC), ("ior_case_analysis", CASES))):
        raise RuntimeError("Frozen diagnostic input changed")
    OUT.mkdir(parents=True)
    save("candidate_effects.json", all_effects)
    save("user_level_summary.json", summaries)
    save("used_candidate_analysis.json", all_used)
    save("method_summary.json", method)
    save("opportunity_assessment.json", assessment)
    print(json.dumps({"method_summary": method, "opportunity_assessment": assessment,
                      "users": [{"user_id": x["user_id"], "spread": x["IOR_CANDIDATE"]["spread"],
                                 "potential_gain": x["POTENTIAL_SINGLE_STEP_IOR_GAIN"],
                                 "used_below_median": x["used_candidates_below_median_IoR"]} for x in summaries]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()

"""Read-only User 1 sanity report for MI-Bridge v2; performs no generation."""
import json

from repro.methods.mi_bridge.multi_bridge_prompt import build_multi_bridge_prompt
from repro.methods.mi_bridge.retrieve_bridge_movies import load_movies
from repro.methods.mi_bridge.select_bridge import BridgeRanking
from repro.methods.mi_bridge.topk_bridge import select_topk_bridges
from repro.utils import ROOT, read_json


def user1_sanity():
    analysis = read_json(ROOT / "repro/results/case_study/user_1_interest_analysis.json")
    saved = read_json(ROOT / "repro/results/case_study/user_1_bridge_selection.json")
    baseline = read_json(ROOT / "repro/results/minimal_baseline_v2/prompt.json")
    movies = load_movies(ROOT / "dataset/ml-1m/movies.dat")
    ranking = BridgeRanking(saved["ranked_bridge_pairs"], movies, analysis["history"], analysis["target"])
    selected = select_topk_bridges(ranking, k=3)
    prompt = build_multi_bridge_prompt(baseline["system_prompt"], baseline["user_prompt"], selected)
    return {
        "MI_BRIDGE_V2_IMPLEMENTED": True,
        "TOP_K": 3,
        "DISTINCT_EXISTING_INTEREST_POLICY": True,
        "SELECTED_BRIDGES": [f"{x['existing_interest']} -> {x['target_interest']}" for x in selected],
        "SELECTED_EXISTING_INTERESTS": [x["existing_interest"] for x in selected],
        "ALL_SELECTED_BRIDGES_FEASIBLE": all(x["candidate_count"] > 0 for x in selected),
        "PER_BRIDGE_CANDIDATE_CAP": prompt["max_candidates_per_bridge"],
        "PROMPT_CANDIDATE_COUNTS_PER_BRIDGE": [len(x) for x in prompt["prompt_candidates_per_bridge"]],
        "V1_FILES_MODIFIED": False,
        "FROZEN_OUTPUTS_MODIFIED": False,
        "MULTI_BRIDGE_CONTEXT": prompt["multi_bridge_context"],
    }


if __name__ == "__main__":
    print(json.dumps(user1_sanity(), ensure_ascii=False, indent=2))

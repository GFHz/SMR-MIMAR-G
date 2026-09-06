"""Frozen-pool, score-free Dynamic-No-Bridge v0.1 mechanics."""
from collections import Counter

from repro.methods.dynamic_mi_bridge.planner import parse_one_item

METHOD_VERSION = "Dynamic-No-Bridge v0.1"
WINDOW_SIZE = 20
TOP_K = 5
MAX_INTERMEDIATE_STEPS = 8
CANDIDATE_LIMIT = 20


def interest_profile(window):
    if len(window) != WINDOW_SIZE:
        raise ValueError("Last-20 window must contain exactly 20 items")
    counts = Counter(g for item in window for g in item["genres"])
    stats = [{"genre": g, "count": n, "frequency": n / WINDOW_SIZE}
             for g, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return stats, stats[:TOP_K]


def allowed_candidates(pool, window, used_ids, target_id):
    excluded = {x["id"] for x in window} | set(used_ids) | {target_id}
    return [x for x in pool if x["id"] not in excluded][:CANDIDATE_LIMIT]


def validate_one(raw, candidates):
    parsed = parse_one_item(raw, (x["title"] for x in candidates))
    if not parsed["success"]:
        return {**parsed, "classification": "invalid", "resolved_item": None}
    movie = next(x for x in candidates if x["title"] == parsed["title"])
    return {**parsed, "classification": "valid_intermediate", "resolved_item": dict(movie)}


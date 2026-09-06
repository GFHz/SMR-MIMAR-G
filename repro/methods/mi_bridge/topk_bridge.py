"""MI-Bridge v2 Top-K selection layered over the frozen v1 ranking."""
from __future__ import annotations

from copy import deepcopy

from .find_fallback_bridge import find_fallback_bridge
from .retrieve_bridge_movies import retrieve_bridge_movies
from .select_bridge import BridgeRanking, bridge_key


METHOD_VERSION = "MI-Bridge v2 Top-K Multi-Bridge Context"


def _feasible_bridge(pair, ranking):
    """Materialize one ranked pair with unchanged v1 direct/fallback rules."""
    i, g = bridge_key(pair)
    if i == g:
        return None
    direct = retrieve_bridge_movies(pair, ranking.movies, ranking.history, ranking.target)
    common = {
        "ranking_rank": pair.get("rank"),
        "existing_interest": i,
        "target_interest": g,
        "bridge_score": pair.get("bridge_score"),
        "user_frequency": pair.get("user_frequency"),
        "target_need": pair.get("target_need"),
        "source_bridge_pair": deepcopy(dict(pair)),
    }
    if direct["candidate_count"] > 0:
        return {
            **common,
            "mode": "direct",
            "candidate_count": direct["candidate_count"],
            "candidates": deepcopy(direct["candidates"]),
            "fallback_intermediate_genre": None,
        }
    fallback = find_fallback_bridge(i, g, ranking.movies, ranking.history, ranking.target)
    selected = fallback["selected_fallback"]
    if selected is None:
        return None
    return {
        **common,
        "mode": "multi_hop",
        "candidate_count": selected["fallback_score"],
        "candidates": {
            "left": deepcopy(selected["left_candidates"]),
            "right": deepcopy(selected["right_candidates"]),
        },
        "left_candidate_count": selected["left_candidate_count"],
        "right_candidate_count": selected["right_candidate_count"],
        "fallback_intermediate_genre": selected["intermediate_interest"],
    }


def feasible_bridge_ranking(bridge_ranking):
    """Return feasible metadata in the original deterministic rank order."""
    if not isinstance(bridge_ranking, BridgeRanking):
        raise TypeError("bridge_ranking must be a BridgeRanking with v1 catalog context")
    return [item for pair in bridge_ranking if (item := _feasible_bridge(pair, bridge_ranking)) is not None]


def select_topk_bridges(bridge_ranking, k=3, prefer_distinct_existing_interests=True):
    """Select Top-K feasible bridges without mutating or re-ranking the input.

    With distinct-interest preference, the first stable pass takes the first
    feasible bridge for each unseen existing-side interest. A second stable
    pass fills any remaining slots from unselected feasible pairs. Both passes
    preserve the frozen v1 rank order.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    feasible = feasible_bridge_ranking(bridge_ranking)
    if not prefer_distinct_existing_interests:
        return deepcopy(feasible[:k])
    selected, selected_keys, seen_interests = [], set(), set()
    for bridge in feasible:
        interest = bridge["existing_interest"]
        if interest in seen_interests:
            continue
        selected.append(bridge)
        selected_keys.add((bridge["existing_interest"], bridge["target_interest"]))
        seen_interests.add(interest)
        if len(selected) == k:
            return deepcopy(selected)
    for bridge in feasible:
        key = bridge["existing_interest"], bridge["target_interest"]
        if key in selected_keys:
            continue
        selected.append(bridge)
        if len(selected) == k:
            break
    return deepcopy(selected)

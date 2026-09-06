"""Fixed Last-20 dynamic bridge replanning without feedback simulation."""
from __future__ import annotations

import ast
import json
from collections import Counter
from copy import deepcopy

from repro.evaluators.formal.title_resolver import TitleResolver
from repro.methods.mi_bridge.retrieve_bridge_movies import retrieve_bridge_movies
from repro.methods.mi_bridge.select_bridge import (
    BridgeRanking, generate_bridge_pairs, rank_bridge_pairs, select_bridge,
)

METHOD_VERSION = "Dynamic MI-Bridge v0.1"
WINDOW_SIZE = 20
TOP_K = 5
LAMBDA = 1.0
MAX_INTERMEDIATE_STEPS = 8


def interest_profile(window):
    if len(window) != WINDOW_SIZE:
        raise ValueError("Dynamic interest window must contain exactly 20 items")
    counts = Counter(genre for item in window for genre in item["genres"])
    statistics = [{"genre": genre, "count": count, "frequency": count / WINDOW_SIZE}
                  for genre, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return {"genre_counts": dict(counts), "genre_statistics": statistics,
            "top_interests": statistics[:TOP_K]}


def rank_and_select(window, target, movies):
    profile = interest_profile(window)
    analysis = {"history": window, "target": target,
                "genre_statistics": profile["genre_statistics"],
                "top_interests": profile["top_interests"]}
    pairs = generate_bridge_pairs(analysis)
    enriched = [dict(pair, **retrieve_bridge_movies(pair, movies, window, target)) for pair in pairs]
    ranked = rank_bridge_pairs(enriched)
    selected = select_bridge(BridgeRanking(ranked, movies, window, target), rejected_bridges=[])
    return profile, ranked, selected


def parse_one_item(raw, allowed_candidate_titles):
    """Accept one exact allowed title, without repair or approximate matching."""
    if not isinstance(raw, str) or not raw.strip():
        return {"success": False, "title": None, "error": "empty_or_non_string"}
    allowed = set(allowed_candidate_titles)
    if not allowed or any(not isinstance(title, str) or not title for title in allowed):
        raise ValueError("allowed_candidate_titles must contain valid title strings")
    text = raw.strip()
    # MovieLens titles normally end in a parenthesized year. Exact catalog
    # membership therefore precedes generic malformed-container diagnostics.
    if text in allowed:
        return {"success": True, "title": text, "error": None}
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, str) and value in allowed:
            return {"success": True, "title": value, "error": None}
        if (isinstance(value, list) and len(value) == 1
                and isinstance(value[0], str) and value[0] in allowed):
            return {"success": True, "title": value[0], "error": None}
        if isinstance(value, (list, tuple, dict, set)):
            return {"success": False, "title": None, "error": "not_exactly_one_item"}
    if text[:1] in "[{(" or text[-1:] in "]})":
        return {"success": False, "title": None, "error": "invalid_container_syntax"}
    return {"success": False, "title": None, "error": "not_exact_allowed_candidate"}


class DynamicMIBridgePlanner:
    def __init__(self, history, target, movies, resolver=None, max_intermediate_steps=MAX_INTERMEDIATE_STEPS):
        if len(history) != WINDOW_SIZE:
            raise ValueError("H_0 must be exactly the frozen Last-20 history")
        if max_intermediate_steps != MAX_INTERMEDIATE_STEPS:
            raise ValueError("Dynamic MI-Bridge v0.1 fixes MAX_INTERMEDIATE_STEPS=8")
        self.initial_history = deepcopy(list(history))
        self.window = deepcopy(list(history))
        self.target = deepcopy(target)
        self.movies = deepcopy(list(movies))
        self.resolver = resolver or TitleResolver(self.movies)
        self.path = []
        self.steps = []
        self.status = "ready"

    def current_plan(self):
        profile, ranking, selected = rank_and_select(self.window, self.target, self.movies)
        return {"step": len(self.path) + 1, "window": deepcopy(self.window),
                "user_frequencies": {x["genre"]: x["frequency"] for x in profile["genre_statistics"]},
                "top_interests": deepcopy(profile["top_interests"]),
                "ranked_bridge_pairs": ranking, "selected_bridge": selected}

    def _validate(self, raw, plan):
        bridge = plan["selected_bridge"]
        candidates = (bridge["direct_candidates"] if bridge["bridge_type"] == "direct"
                      else bridge["selected_fallback"]["left_candidates"] + bridge["selected_fallback"]["right_candidates"])
        parsed = parse_one_item(raw, (x["title"] for x in candidates))
        if not parsed["success"]:
            return {**parsed, "kind": "invalid", "resolution": None}
        resolved = self.resolver.resolve(parsed["title"])
        if resolved["resolution_status"] != "resolved":
            return {**parsed, "success": False, "kind": "invalid", "resolution": resolved,
                    "error": "unresolved_catalog_item"}
        if resolved["raw_item_id"] == self.target["id"]:
            return {**parsed, "kind": "target", "resolution": resolved}
        if resolved["raw_item_id"] not in {x["id"] for x in candidates}:
            return {**parsed, "success": False, "kind": "invalid", "resolution": resolved,
                    "error": "outside_current_bridge_candidates"}
        return {**parsed, "kind": "intermediate", "resolution": resolved}

    def step(self, generate_next_item):
        if self.status not in ("ready", "running"):
            raise RuntimeError("Planner has already stopped")
        if len(self.path) >= MAX_INTERMEDIATE_STEPS:
            self.status = "max_intermediate_steps"
            return {"status": self.status}
        plan = self.current_plan()
        if plan["selected_bridge"]["bridge_type"] == "unavailable":
            self.status = "no_valid_continuation"
            return {"status": self.status, "plan": plan}
        raw = generate_next_item(deepcopy(plan))
        validation = self._validate(raw, plan)
        record = {"plan": plan, "raw_next_item": raw, "validation": validation,
                  "window_before": deepcopy(self.window)}
        if validation["kind"] == "target":
            self.status = "target_reached"
            record.update(status=self.status, window_after=deepcopy(self.window))
        elif validation["kind"] == "intermediate" and validation["success"]:
            movie = deepcopy(self.resolver.movies[validation["resolution"]["raw_item_id"]])
            self.path.append(movie)
            self.window = self.window[1:] + [movie]
            if len(self.window) != WINDOW_SIZE:
                raise AssertionError("Sliding window length changed")
            self.status = "max_intermediate_steps" if len(self.path) == MAX_INTERMEDIATE_STEPS else "running"
            record.update(status=self.status, window_after=deepcopy(self.window))
        else:
            self.status = "no_valid_continuation"
            record.update(status=self.status, window_after=deepcopy(self.window))
        self.steps.append(record)
        return deepcopy(record)

    def run(self, generate_next_item):
        while self.status in ("ready", "running"):
            self.step(generate_next_item)
        return self.snapshot()

    def snapshot(self):
        return {"method_version": METHOD_VERSION, "window_size": WINDOW_SIZE,
                "max_intermediate_steps": MAX_INTERMEDIATE_STEPS,
                "offline_acceptance_assumption": True, "status": self.status,
                "initial_history": deepcopy(self.initial_history), "current_window": deepcopy(self.window),
                "accepted_intermediates": deepcopy(self.path), "steps": deepcopy(self.steps),
                "target_inserted_into_window": False}

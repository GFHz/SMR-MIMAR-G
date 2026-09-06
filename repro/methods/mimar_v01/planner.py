"""Feedback-aware MIMAR-v0.1 planner; evaluator-independent by construction."""
from copy import deepcopy

from .candidate_builder import build_candidates
from .cooccurrence import GenreCooccurrence
from .feedback import ACCEPT, apply_feedback
from .interest_profile import ACTIVATION_THRESHOLD, initial_active_interests, long_term_interests
from .prompt import build_prompt, correction_prompt, parse_one_title
from .route_scoring import rank_routes
from .route_state import FeedbackMemory

METHOD = "MIMAR-v0.1"
MAX_INTERMEDIATE_STEPS = 8
MAX_PLANNING_STEPS = 24
TARGET_COVERAGE_THRESHOLD = 0.8
MAX_LLM_RETRIES = 2


class MIMARPlanner:
    def __init__(self, full_positive_history, target, movies, demographics=None):
        self.history = deepcopy(list(full_positive_history))
        self.target = deepcopy(target)
        self.movies = deepcopy(list(movies))
        self.demographics = deepcopy(demographics or {})
        self.long_term = long_term_interests(self.history)
        self.active = initial_active_interests(self.history)
        self.cooccurrence = GenreCooccurrence(self.movies)
        self.memory = FeedbackMemory()
        self.original_history_ids = {int(x["id"]) for x in self.history}
        self.used_ids = set()
        self.accepted = []
        self.previous_route = None
        self.steps = []
        self.status = "ready"

    def plan(self):
        ranking = rank_routes(self.long_term, self.active, self.target["genres"],
                              self.cooccurrence, self.memory, self.previous_route)
        if not ranking:
            return {"ranking": [], "selected_route": None, "candidates": []}
        for row in ranking:
            route = (row["interest"], row["target_genre"])
            candidates = build_candidates(self.movies, route, self.active, self.cooccurrence,
                                          self.original_history_ids, self.used_ids,
                                          self.target["id"], self.memory)
            if candidates:
                return {"ranking": ranking, "selected_route": row, "candidates": candidates}
        return {"ranking": ranking, "selected_route": None, "candidates": []}

    def step(self, select_item, feedback_provider):
        if self.status not in ("ready", "running"):
            raise RuntimeError("Planner already stopped")
        if len(self.accepted) >= MAX_INTERMEDIATE_STEPS or len(self.steps) >= MAX_PLANNING_STEPS:
            self.status = "safety_cap"
            return {"status": self.status}
        plan = self.plan()
        if plan["selected_route"] is None:
            self.status = "no_useful_route_or_candidate"
            return {"status": self.status, "plan": plan}
        route_row = plan["selected_route"]
        route = (route_row["interest"], route_row["target_genre"])
        prompt = build_prompt(self.demographics, self.long_term, self.active,
                              self.target, route_row, plan["candidates"])
        attempts = []
        parsed = None
        raw = None
        for attempt_index in range(MAX_LLM_RETRIES + 1):
            attempt_prompt = prompt if attempt_index == 0 else correction_prompt(prompt)
            raw = select_item(deepcopy(attempt_prompt))
            parsed = parse_one_title(raw, prompt["allowed_titles"])
            attempts.append({"attempt": attempt_index + 1, "prompt": deepcopy(attempt_prompt),
                             "raw_output": raw, "parse_result": deepcopy(parsed)})
            if parsed["success"]:
                break
        record = {"plan": deepcopy(plan), "prompt": prompt, "raw": raw, "parsed": parsed,
                  "llm_attempts": attempts, "llm_retry_count": len(attempts)-1,
                  "invalid_output_reasons": [x["parse_result"]["error"] for x in attempts
                                             if not x["parse_result"]["success"]],
                  "active_before": deepcopy(self.active), "previous_route": self.previous_route}
        if not parsed["success"]:
            self.status = "invalid_llm_selection_after_retries"
            record["status"] = self.status
            self.steps.append(record)
            return deepcopy(record)
        item = next(x for x in plan["candidates"] if x["title"] == parsed["title"])
        outcome = feedback_provider(deepcopy(item), deepcopy(record))
        old_active_genres = {g for g, v in self.active.items() if v >= ACTIVATION_THRESHOLD}
        self.active = apply_feedback(self.active, self.memory, route, item, outcome)
        new_active_genres = {g for g, v in self.active.items() if v >= ACTIVATION_THRESHOLD}
        if outcome == ACCEPT:
            self.accepted.append(deepcopy(item))
            self.used_ids.add(int(item["id"]))
        switched = self.previous_route is not None and route != self.previous_route
        if self.previous_route is None:
            reason = "OTHER"
        elif outcome != ACCEPT:
            reason = "REJECTION_PENALTY"
        elif new_active_genres - old_active_genres:
            reason = "NEW_INTEREST_ACTIVATED"
        elif switched:
            reason = "ACCEPT_STATE_UPDATE"
        else:
            reason = "ROUTE_SCORE_CHANGE"
        self.previous_route = route
        covered = all(self.active.get(g, 0.0) >= TARGET_COVERAGE_THRESHOLD for g in self.target["genres"])
        self.status = "target_attribute_coverage" if covered else (
            "safety_cap" if len(self.accepted) >= MAX_INTERMEDIATE_STEPS else "running")
        record.update({"selected_item": deepcopy(item), "feedback": outcome,
                       "active_after": deepcopy(self.active), "active_genres_after": sorted(new_active_genres),
                       "newly_activated": sorted(new_active_genres-old_active_genres),
                       "deactivated": sorted(old_active_genres-new_active_genres),
                       "route_switched": switched, "switch_reason": reason, "status": self.status})
        self.steps.append(record)
        return deepcopy(record)

    def snapshot(self):
        return {"method": METHOD, "status": self.status, "long_term_interests": deepcopy(self.long_term),
                "active_interests": deepcopy(self.active), "accepted_intermediates": deepcopy(self.accepted),
                "final_path": deepcopy(self.accepted) + [deepcopy(self.target)], "steps": deepcopy(self.steps),
                "evaluator_used_during_planning": False}

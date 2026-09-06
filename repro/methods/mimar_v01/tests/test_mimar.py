import ast
import inspect
import unittest
from pathlib import Path

from repro.methods.mimar_v01.candidate_builder import build_candidates
from repro.methods.mimar_v01.cooccurrence import GenreCooccurrence
from repro.methods.mimar_v01.feedback import ACCEPT, REJECT, apply_feedback
from repro.methods.mimar_v01.interest_profile import active_genres, initial_active_interests, long_term_interests, update_active_interests
from repro.methods.mimar_v01.planner import MIMARPlanner
from repro.methods.mimar_v01.prompt import build_prompt, parse_one_title
from repro.methods.mimar_v01.route_scoring import ALPHA, BETA, DELTA_ROUTE, GAMMA, KAPPA, rank_routes
from repro.methods.mimar_v01.route_state import FeedbackMemory


def movie(i, title, genres):
    return {"id": i, "title": title, "genres": genres}


class MIMARTests(unittest.TestCase):
    def setUp(self):
        self.history = [movie(i, f"H{i}", ["Comedy"] if i <= 20 else ["Action"]) for i in range(1, 26)]
        self.target = movie(99, "Target", ["Drama", "Romance"])
        self.catalog = self.history + [self.target,
            movie(30, "Both", ["Comedy", "Drama"]), movie(31, "Left", ["Comedy", "Action"]),
            movie(32, "Right", ["Drama", "Action"]), movie(33, "New", ["Fantasy", "Drama"])]
        self.co = GenreCooccurrence(self.catalog)

    def test_long_and_short_are_separate(self):
        long = long_term_interests(self.history)
        active = initial_active_interests(self.history)
        self.assertIn("Comedy", long)
        self.assertNotEqual(long["Comedy"]["score"], active["Comedy"])

    def test_cooccurrence_formula_and_identity(self):
        d = self.co.detail("Comedy", "Drama")
        self.assertAlmostEqual(d["normalized_cooccurrence"], d["count_joint"]/(d["count_interest"]*d["count_target_genre"])**0.5)
        self.assertEqual(self.co.score("Drama", "Drama"), 1.0)

    def test_route_score(self):
        long = {"Comedy": {"score": .4}}
        active = {"Comedy": .5}
        memory = FeedbackMemory({("Comedy", "Drama"): .2})
        row = rank_routes(long, active, ["Drama"], self.co, memory, ("Comedy", "Drama"))[0]
        expected = ALPHA*.4+BETA*.5+GAMMA*self.co.score("Comedy", "Drama")-DELTA_ROUTE*.2+KAPPA
        self.assertAlmostEqual(row["route_score"], expected)

    def test_route_penalty_decay_and_recovery(self):
        memory = FeedbackMemory()
        memory.advance(("Comedy", "Drama"), 30)
        self.assertEqual(memory.penalty(("Comedy", "Drama")), .5)
        memory.advance()
        self.assertAlmostEqual(memory.penalty(("Comedy", "Drama")), .4)
        self.assertTrue(memory.cooling_down(30))
        memory.advance()
        self.assertFalse(memory.cooling_down(30))

    def test_rejection_can_switch_route(self):
        long = {"Comedy": {"score": .5}, "Action": {"score": .5}}
        active = {"Comedy": .5, "Action": .5}
        memory = FeedbackMemory()
        first = rank_routes(long, active, ["Drama"], self.co, memory)[0]
        route = (first["interest"], first["target_genre"])
        memory.advance(route, 30)
        second = rank_routes(long, active, ["Drama"], self.co, memory)[0]
        self.assertNotEqual(route, (second["interest"], second["target_genre"]))

    def test_new_interest_activation(self):
        updated = update_active_interests({"Comedy": .5}, ["Fantasy"], True)
        self.assertIn("Fantasy", active_genres(updated))

    def test_deactivation_preserves_long_term(self):
        long = {"Comedy": {"score": .6}}
        active = {"Comedy": .051}
        for _ in range(2):
            active = update_active_interests(active, ["Other"], False)
        self.assertNotIn("Comedy", active_genres(active))
        self.assertIn("Comedy", long)

    def test_permanent_history_and_path_exclusion(self):
        memory = FeedbackMemory()
        candidates = build_candidates(self.catalog, ("Comedy", "Drama"), {"Comedy": .5}, self.co,
                                      {1, 30}, {31}, 99, memory)
        ids = {x["id"] for x in candidates}
        self.assertNotIn(1, ids); self.assertNotIn(30, ids); self.assertNotIn(31, ids); self.assertNotIn(99, ids)

    def test_rejected_item_cooldown(self):
        memory = FeedbackMemory(); memory.advance(("Comedy", "Drama"), 30)
        candidates = build_candidates(self.catalog, ("Comedy", "Drama"), {"Comedy": .5}, self.co,
                                      set(range(1, 26)), set(), 99, memory)
        self.assertNotIn(30, {x["id"] for x in candidates})

    def test_multi_tag_tiers(self):
        candidates = build_candidates(self.catalog, ("Comedy", "Drama"), {"Comedy": .5, "Action": .2}, self.co,
                                      set(range(1, 26)), set(), 99, FeedbackMemory())
        tiers = {x["title"]: x["tier"] for x in candidates}
        self.assertEqual(tiers["Both"], 1)
        self.assertEqual(tiers["Left"], 2)

    def test_parser_exact_one(self):
        allowed = ["Both", "I.Q. (1994)"]
        self.assertTrue(parse_one_title("I.Q. (1994)", allowed)["success"])
        self.assertTrue(parse_one_title('["Both"]', allowed)["success"])
        self.assertFalse(parse_one_title('["Both", "I.Q. (1994)"]', allowed)["success"])
        self.assertFalse(parse_one_title("Unknown", allowed)["success"])

    def test_feedback_accept_reject(self):
        memory = FeedbackMemory(); active = {"Comedy": .5}
        accepted = apply_feedback(active, memory, ("Comedy", "Drama"), movie(30,"X",["Drama"]), ACCEPT)
        self.assertGreater(accepted["Drama"], 0)
        rejected = apply_feedback(accepted, memory, ("Comedy", "Drama"), movie(31,"Y",["Action"]), REJECT)
        self.assertGreater(memory.penalty(("Comedy", "Drama")), 0)
        self.assertNotIn(31, [])

    def test_planner_tracks_route_and_acceptance(self):
        planner = MIMARPlanner(self.history, self.target, self.catalog)
        result = planner.step(lambda p: p["allowed_titles"][0], lambda item, state: ACCEPT)
        self.assertEqual(result["feedback"], ACCEPT)
        self.assertEqual(len(planner.accepted), 1)
        self.assertFalse(planner.snapshot()["evaluator_used_during_planning"])

    def test_prompt_forbids_target_and_ends_with_allowed_constraint(self):
        planner = MIMARPlanner(self.history, self.target, self.catalog)
        plan = planner.plan()
        prompt = build_prompt({}, planner.long_term, planner.active, self.target,
                              plan["selected_route"], plan["candidates"])
        self.assertIn("TARGET ITEM MUST NEVER BE OUTPUT", prompt["system_prompt"])
        self.assertIn("DO NOT REWRITE OR ABBREVIATE", prompt["system_prompt"])
        self.assertIn("ALLOWED CANDIDATES", prompt["user_prompt"])
        self.assertTrue(prompt["user_prompt"].endswith(
            "Your entire response must be exactly one title copied verbatim from the ALLOWED CANDIDATES list."))

    def test_bounded_retry_keeps_plan_and_recovers(self):
        planner = MIMARPlanner(self.history, self.target, self.catalog)
        calls = []
        def select(prompt):
            calls.append(prompt)
            return "Target" if len(calls) == 1 else prompt["allowed_titles"][0]
        result = planner.step(select, lambda item, state: ACCEPT)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["llm_retry_count"], 1)
        self.assertTrue(result["parsed"]["success"])
        self.assertEqual(calls[0]["allowed_titles"], calls[1]["allowed_titles"])
        self.assertIn("INVALID OUTPUT", calls[1]["user_prompt"])

    def test_stops_after_three_invalid_generations(self):
        planner = MIMARPlanner(self.history, self.target, self.catalog)
        result = planner.step(lambda prompt: "Target", lambda item, state: ACCEPT)
        self.assertEqual(len(result["llm_attempts"]), 3)
        self.assertEqual(result["status"], "invalid_llm_selection_after_retries")

    def test_evaluator_independence(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = ("sasrec", "ior", "ioi", "formal.evaluator", "target_rank", "target_probability")
        for path in root.glob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            tree = ast.parse(source)
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            rendered = " ".join(ast.unparse(n).lower() for n in imports)
            self.assertFalse(any(term in rendered for term in forbidden), (path, rendered))


if __name__ == "__main__":
    unittest.main()

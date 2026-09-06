import unittest

from repro.methods.dynamic_mi_bridge.planner import (
    DynamicMIBridgePlanner, interest_profile, parse_one_item,
)


def movie(i, title, genres):
    return {"id": i, "title": title, "genres": genres}


class DynamicPlannerTests(unittest.TestCase):
    def setUp(self):
        self.history = [movie(i, f"H{i}", ["Comedy"] if i < 11 else ["Action"]) for i in range(1, 21)]
        self.target = movie(100, "Target", ["Drama"])
        self.candidates = [movie(30+i, f"C{i}", ["Comedy", "Drama"]) for i in range(10)]
        self.movies = self.history + [self.target] + self.candidates

    def test_initial_window_and_frequency(self):
        planner = DynamicMIBridgePlanner(self.history, self.target, self.movies)
        self.assertEqual(len(planner.window), 20)
        profile = interest_profile(planner.window)
        self.assertEqual(profile["genre_counts"], {"Comedy": 10, "Action": 10})
        self.assertEqual([x["genre"] for x in profile["top_interests"]], ["Action", "Comedy"])

    def test_accept_slides_exactly_one(self):
        planner = DynamicMIBridgePlanner(self.history, self.target, self.movies)
        result = planner.step(lambda _: '["C0"]')
        self.assertEqual(result["status"], "running")
        self.assertEqual(len(planner.window), 20)
        self.assertEqual(planner.window[0]["title"], "H2")
        self.assertEqual(planner.window[-1]["title"], "C0")
        self.assertEqual(interest_profile(planner.window)["genre_counts"]["Drama"], 1)

    def test_target_stops_without_window_update(self):
        planner = DynamicMIBridgePlanner(self.history, self.target, self.movies)
        before = list(planner.window)
        original_plan = planner.current_plan()
        original_plan["selected_bridge"]["direct_candidates"].append(self.target)
        planner.current_plan = lambda: original_plan
        result = planner.step(lambda _: '"Target"')
        self.assertEqual(result["status"], "target_reached")
        self.assertEqual(planner.window, before)
        self.assertNotIn(self.target, planner.window)

    def test_invalid_or_outside_candidate_stops(self):
        outside = movie(90, "Outside", ["Horror"])
        planner = DynamicMIBridgePlanner(self.history, self.target, self.movies + [outside])
        result = planner.step(lambda _: "Outside")
        self.assertEqual(result["status"], "no_valid_continuation")
        self.assertEqual(result["validation"]["error"], "not_exact_allowed_candidate")
        self.assertEqual(planner.window, self.history)

    def test_max_eight_and_replanning(self):
        planner = DynamicMIBridgePlanner(self.history, self.target, self.movies)
        calls = []
        def choose(plan):
            calls.append(plan["selected_bridge"]["selected_bridge"])
            return f"C{len(calls)-1}"
        result = planner.run(choose)
        self.assertEqual(result["status"], "max_intermediate_steps")
        self.assertEqual(len(result["accepted_intermediates"]), 8)
        self.assertEqual(len(result["current_window"]), 20)
        self.assertEqual(len(calls), 8)

    def test_strict_single_item_parser(self):
        allowed = {"C0", "C1"}
        self.assertEqual(parse_one_item('["C0"]', allowed)["title"], "C0")
        self.assertFalse(parse_one_item('["C0", "C1"]', allowed)["success"])
        self.assertFalse(parse_one_item("C0\nC1", allowed)["success"])


if __name__ == "__main__":
    unittest.main()

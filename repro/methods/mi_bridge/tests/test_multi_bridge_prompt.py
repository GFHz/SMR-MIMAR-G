import copy
import unittest

from repro.methods.mi_bridge.multi_bridge_prompt import build_multi_bridge_prompt


class MultiBridgePromptTests(unittest.TestCase):
    def setUp(self):
        self.bridges = [{
            "existing_interest": "Comedy", "target_interest": "Romance", "mode": "direct",
            "candidate_count": 1, "candidates": [{"id": 1, "title": "Bridge Movie", "genres": ["Comedy", "Romance"]}],
            "fallback_intermediate_genre": None,
        }]

    def test_prompt_is_advisory_and_baseline_is_preserved(self):
        before = copy.deepcopy(self.bridges)
        result = build_multi_bridge_prompt("SYSTEM", "USER", self.bridges)
        self.assertEqual(result["system_prompt"], "SYSTEM")
        self.assertEqual(result["user_prompt"], "USER" + result["multi_bridge_context"])
        text = result["multi_bridge_context"].lower()
        self.assertIn("do not need to use every bridge", text)
        self.assertIn("rather than constraining every individual path node", text)
        for forbidden in ["target must", "fixed path", "fixed position", "must use a bridge candidate", "history items must never"]:
            self.assertNotIn(forbidden, text)
        self.assertEqual(self.bridges, before)

    def test_context_only_once_and_nonempty(self):
        result = build_multi_bridge_prompt("", "", self.bridges)
        self.assertEqual(result["user_prompt"].count("[MULTI-INTEREST BRIDGE CONTEXT]"), 1)
        with self.assertRaises(ValueError):
            build_multi_bridge_prompt("", result["user_prompt"], self.bridges)
        with self.assertRaises(ValueError):
            build_multi_bridge_prompt("", "", [])

    def test_prompt_candidate_cap_only_changes_visible_titles(self):
        bridge = copy.deepcopy(self.bridges[0])
        bridge["candidate_count"] = 9
        bridge["candidates"] = [
            {"id": i, "title": f"Movie {i}", "genres": ["Comedy", "Romance"]} for i in range(1, 10)
        ]
        result = build_multi_bridge_prompt("", "", [bridge])
        self.assertEqual(len(result["prompt_candidates_per_bridge"][0]), 5)
        self.assertEqual(result["prompt_candidates_per_bridge"][0], [f"Movie {i}" for i in range(1, 6)])
        self.assertEqual(bridge["candidate_count"], 9)
        self.assertEqual(len(bridge["candidates"]), 9)
        with self.assertRaises(ValueError):
            build_multi_bridge_prompt("", "", [bridge], 0)

    def test_soft_novelty_is_optional_and_advisory(self):
        original = build_multi_bridge_prompt("SYSTEM", "USER", self.bridges)
        refined = build_multi_bridge_prompt(
            "SYSTEM", "USER", self.bridges, novelty_guidance=True)
        self.assertFalse(original["novelty_guidance"])
        self.assertTrue(refined["novelty_guidance"])
        self.assertNotIn("Soft novelty guidance:", original["user_prompt"])
        self.assertIn("Soft novelty guidance:", refined["user_prompt"])
        self.assertIn("not strictly forbidden", refined["user_prompt"])
        self.assertNotIn("target must be last", refined["user_prompt"].lower())
        self.assertNotIn("must use", refined["multi_bridge_context"].lower())
        module = __import__("repro.methods.mi_bridge.multi_bridge_prompt", fromlist=["SOFT_NOVELTY_GUIDANCE"])
        expected = original["user_prompt"].replace(
            "[/MULTI-INTEREST BRIDGE CONTEXT]",
            "\nSoft novelty guidance:\n" + module.SOFT_NOVELTY_GUIDANCE +
            "\n[/MULTI-INTEREST BRIDGE CONTEXT]",
        )
        self.assertEqual(refined["user_prompt"], expected)


if __name__ == "__main__":
    unittest.main()

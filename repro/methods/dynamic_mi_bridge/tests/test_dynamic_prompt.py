import unittest

from repro.methods.dynamic_mi_bridge.prompt import (
    DYNAMIC_STEP_SYSTEM_PROMPT, build_next_item_prompt,
)


class DynamicPromptTests(unittest.TestCase):
    def setUp(self):
        self.base_system = "add at least ten movies and construct a complete influence path"
        self.base_user = (
            "Gender:Male\nAge:18-24\nOccupation:programmer\n\n"
            "Historical data:\nHistory A Genre:Comedy\n"
            "Target movie: Target A Genre:Romance\n"
        )
        self.context = {"step": 1, "selected_bridge": {
            "bridge_type": "direct", "existing_interest": "Comedy",
            "target_interest": "Romance",
            "direct_candidates": [
                {"id": 1, "title": "Candidate A (1995)", "genres": ["Comedy", "Romance"]},
                {"id": 2, "title": "Candidate B (1996)", "genres": ["Comedy", "Romance"]},
            ]}}
        self.result = build_next_item_prompt(self.base_system, self.base_user, self.context)

    def test_dynamic_system_replaces_full_path_system(self):
        self.assertEqual(self.result["system_prompt"], DYNAMIC_STEP_SYSTEM_PROMPT)
        combined = self.result["system_prompt"] + "\n" + self.result["user_prompt"]
        for forbidden in ("at least ten", "complete influence path", "step-by-step reasoning",
                          "recommend movies one by one"):
            self.assertNotIn(forbidden, combined.lower())

    def test_requires_exactly_one_title_only(self):
        system = self.result["system_prompt"]
        self.assertIn("MUST choose exactly one movie title", system)
        self.assertIn("Output ONLY the exact movie title.", system)
        self.assertIn("Do not output multiple movies.", system)
        self.assertIn("Do not output a list.", system)
        self.assertIn("Do not output reasoning.", system)

    def test_requires_selection_from_exact_candidates(self):
        user = self.result["user_prompt"]
        self.assertIn('Allowed candidates:\n["Candidate A (1995)", "Candidate B (1996)"]', user)
        self.assertIn("Select exactly one movie from the allowed candidates.", user)
        self.assertIn("Return only the exact title.", user)

    def test_preserves_user_state_fields(self):
        user = self.result["user_prompt"]
        for value in ("Gender:Male", "Age:18-24", "Occupation:programmer",
                      "Historical data:", "History A Genre:Comedy",
                      "Target movie: Target A Genre:Romance",
                      "Current existing interest: Comedy",
                      "Current target-side interest: Romance",
                      "Current bridge type: direct"):
            self.assertIn(value, user)


if __name__ == "__main__":
    unittest.main()

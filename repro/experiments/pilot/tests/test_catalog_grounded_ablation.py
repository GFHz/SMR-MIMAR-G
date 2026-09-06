import unittest

from repro.experiments.pilot.run_catalog_grounded_ablation import (
    EXPECTED_SELECTED, GROUNDING_INSTRUCTION, candidate_pool, load_catalog,
    preflight, select_users,
)


class CatalogGroundedAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.users, cls.pools, cls.prompts = preflight(check_service=False)

    def test_deterministic_selection(self):
        self.assertEqual(select_users(), EXPECTED_SELECTED)

    def test_pools_are_100_unique_catalog_items(self):
        catalog_ids = {x["movie_id"] for x in load_catalog()}
        for pool in self.pools.values():
            ids = [x["movie_id"] for x in pool]
            self.assertEqual(len(ids), len(set(ids)), 100)
            self.assertTrue(set(ids) <= catalog_ids)

    def test_history_target_and_bridge_candidates_in_pool(self):
        for user in self.users:
            ids = {x["movie_id"] for x in self.pools[str(user["user_id"])]}
            self.assertTrue({x["movie_id"] for x in user["history"]} <= ids)
            self.assertIn(user["target"]["movie_id"], ids)
            self.assertTrue({x["id"] for x in user["bridge"]["direct_candidates"]} <= ids)

    def test_same_grounding_and_only_bridge_difference(self):
        for baseline, ours in self.prompts.values():
            self.assertEqual(baseline["system_prompt"], ours["system_prompt"])
            self.assertEqual(baseline["formatting_user_prompt"], ours["formatting_user_prompt"])
            self.assertEqual(baseline["user_prompt"].count(GROUNDING_INSTRUCTION), 1)
            self.assertEqual(ours["user_prompt"].count(GROUNDING_INSTRUCTION), 1)
            self.assertEqual(ours["user_prompt"], baseline["user_prompt"] + ours["bridge_context"])

    def test_pool_determinism(self):
        catalog = {x["movie_id"]: x for x in load_catalog()}
        for user in self.users:
            self.assertEqual(candidate_pool(user, catalog), candidate_pool(user, catalog))


if __name__ == "__main__":
    unittest.main()

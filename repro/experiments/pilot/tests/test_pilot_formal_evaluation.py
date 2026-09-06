import math
import unittest

from repro.experiments.pilot.evaluate_pilot_formal import (
    EXPECTED_MANIFEST_SHA, GENERATION, MANIFEST, MAIN, canonical_manifest_sha,
    directory_freeze, method_aggregate, paired, read, user_aggregate,
)


class PilotFormalEvaluationTests(unittest.TestCase):
    def test_manifest_and_generation_inventory(self):
        manifest = read(MANIFEST)
        self.assertEqual(canonical_manifest_sha(manifest), EXPECTED_MANIFEST_SHA)
        self.assertEqual(manifest["formal_protocol_version"], "Formal-Evaluation-v1")
        self.assertTrue(all("movie_id" in u["target"] for u in manifest["users"]))
        self.assertTrue(all(len(u["history"]) == 20 and all("movie_id" in x for x in u["history"])
                            for u in manifest["users"]))
        freeze = directory_freeze(GENERATION)
        self.assertEqual(sum(x["path"].startswith("users/") for x in freeze["files"]), 40)
        self.assertEqual(sum(x["path"].startswith("raw/") for x in freeze["files"]), 84)

    def test_all_frozen_records_are_present_and_parsed(self):
        manifest = read(MANIFEST)
        records = []
        for user in manifest["users"]:
            for method in ("baseline", "mi_bridge"):
                for index in (1, 2):
                    records.append(read(GENERATION / "users" / str(user["user_id"]) / f"{method}_path_{index}.json"))
        self.assertEqual(len(records), 40)
        self.assertTrue(all(r["parse_success"] and r["parsed_path"] for r in records))

    def test_path_to_user_then_pair_aggregation(self):
        def path(uid, method, value):
            return {"user_id": uid, "method": method, "formal_metric_status": "valid", "parse_success": True,
                    "target_present": True, "target_is_last": True, **{m: value for m in MAIN},
                    "HistoryReuseRate": value, "NewIntermediateCount": value, "BridgeCandidateUsageCount": value}
        paths = [path(1, "baseline", 1), path(1, "baseline", 3), path(1, "mi_bridge", 3), path(1, "mi_bridge", 5)]
        users = user_aggregate(paths)
        result = paired(users)
        self.assertEqual(result["paired_valid_user_count"], 1)
        self.assertEqual(result["users"][0]["delta_IoI"], 2)

    def test_invalid_excluded_from_formal_mean_but_retained_in_total(self):
        base = {"user_id": 1, "method": "baseline", "parse_success": True, "target_present": True,
                "target_is_last": True, "HistoryReuseRate": 0.0, "NewIntermediateCount": 1,
                "BridgeCandidateUsageCount": 0}
        valid = {**base, "formal_metric_status": "valid", **{m: 2.0 for m in MAIN}}
        invalid = {**base, "formal_metric_status": "invalid_unresolved", **{m: None for m in MAIN}}
        ours = [{**valid, "method": "mi_bridge"}, {**valid, "method": "mi_bridge"}]
        result = method_aggregate([valid, invalid, *ours])["Baseline"]
        self.assertEqual(result["TOTAL_PATH_COUNT"], 2)
        self.assertEqual(result["VALID_PATH_COUNT"], 1)
        self.assertEqual(result["IoI_mean"], 2.0)

    def test_user1_regression_formulas_and_saved_values(self):
        root = MANIFEST.parents[3]
        saved = read(root / "repro/results/case_study/mi_bridge_v1/formal_evaluation/path_level_metrics.json")
        known = saved["baseline_path_1"]["metrics"]
        self.assertTrue(math.isclose(known["IoI"], math.log(known["P_after"] + 1e-12) - math.log(known["P_before"] + 1e-12)))
        self.assertEqual(known["IoR"], known["R_before"] - known["R_after"])
        self.assertTrue(math.isclose(known["ProxyAcceptability"], sum(known["path_item_scores"]) / len(known["path_item_scores"])))
        self.assertEqual(known["Coherence"], sum(x["pair_coherence"] for x in known["pairs"]) / len(known["pairs"]))


if __name__ == "__main__":
    unittest.main()

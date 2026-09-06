"""Real checkpoint CPU inference on User 1; no model/LLM training/generation."""
import math
import unittest
import torch
from repro.evaluators.sasrec.evaluator import validation_context


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev, cls.adapter, _, cls.history, cls.target, cls.paths, _ = validation_context()

    def test_score_finite(self):
        score = self.ev.get_target_score(self.history, self.target)
        self.assertTrue(math.isfinite(score))
        self.assertTrue(0 <= score <= 1)

    def test_rank_legal(self):
        rank = self.ev.get_target_rank(self.history, self.target)
        self.assertIsInstance(rank, int)
        self.assertTrue(1 <= rank <= self.ev.item_count)

    def test_semantic_id_bypass(self):
        self.assertFalse(hasattr(self.ev, 'tokenizer'))
        self.assertEqual(len(self.history), 20)
        self.assertTrue(all(isinstance(i, int) for i in self.history))
        self.assertEqual(self.ev.full_scores(self.history).numel(), self.ev.item_count)

    def test_strict_path_not_scored(self):
        mapping = self.adapter.evaluation_intermediates(self.paths['ours_path_1'], 2620)
        result = self.ev.evaluate_path(self.history, self.target, mapping)
        self.assertIsNone(result['target_score_after'])
        self.assertIsNone(result['SASREC_PROXY_ACCEPTABILITY'])

    def test_drop_path_pipeline(self):
        mapping = self.adapter.evaluation_intermediates(self.paths['ours_path_1'], 2620,
                                                       'DROP_UNRESOLVED_DIAGNOSTIC_MODE')
        result = self.ev.evaluate_path(self.history, self.target, mapping)
        self.assertTrue(math.isfinite(result['target_score_after']))
        self.assertEqual(len(result['path_item_scores']), 8)
        self.assertEqual(result['history_length_after'], 28)
        self.assertTrue(result['diagnostic_only'])

    def test_target_first_no_leakage(self):
        mapping = self.adapter.evaluation_intermediates(self.paths['baseline_path_2'], 2620)
        result = self.ev.evaluate_path(self.history, self.target, mapping)
        self.assertFalse(result['target_in_extension'])
        self.assertEqual(result['history_length_after'], 29)

    def test_proxy_scores_before_append(self):
        items = self.history[:2]
        expected = [self.ev.get_item_score(self.history, items[0]),
                    self.ev.get_item_score(self.history + [items[0]], items[1])]
        actual = self.ev.get_path_item_scores(self.history, items)
        self.assertEqual(actual['path_item_scores'], expected)
        self.assertEqual(actual['SASREC_PROXY_ACCEPTABILITY'], sum(expected) / 2)

    def test_no_identical_full_scores(self):
        scores = self.ev.full_scores(self.history)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertFalse(torch.all(scores == scores[0]))

    def test_score_and_rank_not_constant_across_frozen_paths(self):
        scores = [self.ev.get_target_score(self.history, self.target)]
        ranks = [self.ev.get_target_rank(self.history, self.target)]
        for path in self.paths.values():
            mapped = self.adapter.evaluation_intermediates(path, 2620, 'DROP_UNRESOLVED_DIAGNOSTIC_MODE')
            extended = self.history + [i['internal_id'] for i in mapped['evaluation_intermediates']]
            scores.append(self.ev.get_target_score(extended, self.target))
            ranks.append(self.ev.get_target_rank(extended, self.target))
        self.assertGreater(len(set(scores)), 1)
        self.assertGreater(len(set(ranks)), 1)

    def test_reject_invalid_id(self):
        for item in (0, -1, self.ev.item_count):
            with self.assertRaises(ValueError):
                self.ev.get_target_score(self.history, item)

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            self.ev.get_target_score(self.history * 3, self.target)


if __name__ == '__main__':
    unittest.main()

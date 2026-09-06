"""Exact mapping and adapted target-exclusion rules on the frozen single case."""
import unittest
from repro.evaluators.sasrec.evaluator import validation_context


class DataAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev, cls.adapter, cls.sample, cls.history, cls.target, cls.paths, _ = validation_context()

    def test_raw_item_roundtrip(self):
        index = self.adapter.raw_item_id_to_internal(2620)
        self.assertEqual(str(self.adapter.item_tokens[index]), '2620')

    def test_title_maps_raw(self):
        self.assertEqual(self.adapter.movie_title_to_raw_item_id('This Is My Father (1998)'), 2620)

    def test_unknown_remains_unresolved(self):
        self.assertIsNone(self.adapter.movie_title_to_internal_id('The Secret Garden, The (1993)'))

    def test_user_one_exists(self):
        self.assertGreater(self.adapter.raw_user_id_to_internal(1), 0)

    def test_target_in_vocabulary(self):
        self.assertTrue(0 < self.target < self.ev.item_count)

    def test_twenty_history_items(self):
        expected = [item['id'] for item in self.sample['history']]
        self.assertEqual(len(self.history), 20)
        self.assertEqual(self.history, self.adapter.history_raw_ids_to_internal(expected))

    def test_target_excluded(self):
        for path in self.paths.values():
            result = self.adapter.evaluation_intermediates(path, 2620, 'DROP_UNRESOLVED_DIAGNOSTIC_MODE')
            self.assertNotIn(self.target, [i['internal_id'] for i in result['evaluation_intermediates']])

    def test_target_first_preserves_non_target_order(self):
        path = self.paths['baseline_path_2']
        result = self.adapter.evaluation_intermediates(path, 2620)
        self.assertEqual(result['target_original_positions'], [1])
        self.assertEqual([i['title'] for i in result['evaluation_intermediates']], path[1:])

    def test_strict_fail(self):
        result = self.adapter.evaluation_intermediates(self.paths['ours_path_1'], 2620)
        self.assertEqual(result['status'], 'unresolved_strict_failure')
        self.assertIsNone(result['evaluation_intermediates'])

    def test_drop_diagnostic_only(self):
        result = self.adapter.evaluation_intermediates(self.paths['ours_path_1'], 2620,
                                                      'DROP_UNRESOLVED_DIAGNOSTIC_MODE')
        self.assertEqual(result['status'], 'ready')
        self.assertTrue(result['diagnostic_only'])
        self.assertEqual(len(result['evaluation_intermediates']), 8)
        self.assertEqual(len(result['unresolved_items']), 1)

    def test_all_resolved_items_map(self):
        for path in self.paths.values():
            mapping = self.adapter.path_titles_to_internal(path)
            for item in mapping['items']:
                if item['raw_id'] is not None:
                    self.assertEqual(str(self.adapter.item_tokens[item['internal_id']]), str(item['raw_id']))


if __name__ == '__main__':
    unittest.main()

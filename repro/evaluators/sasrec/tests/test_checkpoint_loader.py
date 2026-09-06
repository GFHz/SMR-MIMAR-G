"""Strict checkpoint compatibility, with no training and no external writes."""
import unittest
from repro.evaluators.sasrec.evaluator import validation_context


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev, _, _, _, _, _, cls.metadata = validation_context()

    def test_checkpoint_strict_load(self):
        self.assertTrue(self.metadata['strict_load_success'])
        self.assertEqual(self.metadata['missing_keys'], [])
        self.assertEqual(self.metadata['unexpected_keys'], [])
        self.assertEqual(self.metadata['shape_mismatches'], {})

    def test_architecture_matches(self):
        for value in self.metadata['architecture_comparison'].values():
            self.assertEqual(value['checkpoint'], value['effective'])

    def test_embedding_vocab_matches(self):
        self.assertEqual(self.metadata['item_embedding_shape'], [self.ev.item_count, 64])

    def test_history_not_truncated(self):
        self.assertEqual(self.ev.max_length, 50)
        self.assertGreaterEqual(self.ev.max_length, 20)

    def test_no_training_or_split(self):
        self.assertFalse(self.metadata['training_performed'])
        self.assertFalse(self.metadata['split_performed'])
        self.assertFalse(self.ev.model.training)


if __name__ == '__main__':
    unittest.main()

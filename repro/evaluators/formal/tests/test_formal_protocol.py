"""Synthetic unit fixtures test rules, not simulated users or experimental paths."""
import copy
import math
import unittest
from pathlib import Path

from repro.evaluators.formal.title_resolver import TitleResolver
from repro.evaluators.formal.protocol import prepare_path, IOI_EPS
from repro.evaluators.formal.coherence import genre_coherence
from repro.evaluators.formal.metrics import interest_increase, rank_increase, evaluate_prepared, aggregate
from repro.evaluators.sasrec.evaluator import SASRecEvaluator

ROOT = Path(__file__).resolve().parents[4]


class FakeAdapter:
    def raw_item_id_to_internal(self, item):
        return item


class PrimitiveSpy:
    """Use the real existing sequential-update method with observable toy scores."""
    max_length = 50
    get_path_item_scores = SASRecEvaluator.get_path_item_scores

    def __init__(self):
        self.calls = []

    def get_target_score(self, history, target):
        return 0.1 if len(history) == 1 else 0.2

    def get_target_rank(self, history, target):
        self.calls.append(('rank', tuple(history), target))
        return 100 if len(history) == 1 else 40

    def get_item_score(self, history, item):
        self.calls.append(('proxy', tuple(history), item))
        return len(history) / 10


class FormalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            {'id': 1, 'title': 'History (1990)', 'genres': ['Comedy']},
            {'id': 2, 'title': 'Secret Garden, The (1993)', 'genres': ["Children's", 'Drama']},
            {'id': 3, 'title': 'Bridge, A (1991)', 'genres': ['Drama', 'Romance']},
            {'id': 4, 'title': 'Target (1998)', 'genres': ['Romance']},
            {'id': 5, 'title': 'Other (1995)', 'genres': ['Action']},
        ]
        self.resolver = TitleResolver(self.catalog)
        self.target = self.resolver.resolve('Target (1998)')
        self.spy = PrimitiveSpy()

    def prepare(self, titles):
        return prepare_path({'path': titles, 'parse_success': True}, self.resolver, self.target, {1}, {3})

    def evaluate(self, titles, diagnostic=False):
        return evaluate_prepared(self.prepare(titles), self.target, [1], FakeAdapter(), self.spy, diagnostic)

    def test_target_excluded(self):
        result = self.prepare(['Bridge, A (1991)', 'Target (1998)'])
        self.assertEqual([x['raw_item_id'] for x in result['evaluation_intermediates']], [3])

    def test_target_first(self):
        result, _ = self.evaluate(['Target (1998)', 'Bridge, A (1991)'])
        self.assertEqual(result['validity']['target_original_position'], 1)
        self.assertFalse(result['validity']['target_is_last'])
        self.assertEqual(result['metrics']['mapped_evaluation_intermediates'], [3])

    def test_target_middle(self):
        result = self.prepare(['History (1990)', 'Target (1998)', 'Bridge, A (1991)'])
        self.assertEqual(result['validity']['target_original_position'], 2)
        self.assertEqual([x['raw_item_id'] for x in result['evaluation_intermediates']], [1, 3])

    def test_all_target_occurrences_excluded(self):
        result = self.prepare(['Target (1998)', 'Bridge, A (1991)', 'Target (1998)'])
        self.assertEqual(result['validity']['target_original_positions'], [1, 3])
        self.assertEqual(len(result['evaluation_intermediates']), 1)

    def test_exact_match(self):
        self.assertEqual(self.resolver.resolve('Secret Garden, The (1993)')['resolution_type'], 'exact')

    def test_unique_article_resolution(self):
        for title in ('The Secret Garden (1993)', 'The Secret Garden, The (1993)'):
            result = self.resolver.resolve(title)
            self.assertEqual(result['raw_item_id'], 2)
            self.assertEqual(result['raw_title'], title)
            self.assertEqual(result['resolution_type'], 'article_normalization')

    def test_an_article(self):
        resolver = TitleResolver([{'id': 8, 'title': 'Example, An (1990)', 'genres': []}])
        self.assertEqual(resolver.resolve('An Example (1990)')['raw_item_id'], 8)

    def test_ambiguous_invalid(self):
        resolver = TitleResolver(self.catalog + [{'id': 6, 'title': 'The Secret Garden (1993)', 'genres': ['Drama']}])
        title = 'The Secret Garden, The (1993)'
        self.assertEqual(resolver.resolve(title)['resolution_status'], 'ambiguous')
        result = prepare_path({'path': [title], 'parse_success': True}, resolver, self.target, {1}, {3})
        self.assertEqual(result['FORMAL_METRIC_STATUS'], 'invalid_unresolved')

    def test_exact_has_priority(self):
        resolver = TitleResolver(self.catalog + [{'id': 6, 'title': 'The Secret Garden (1993)', 'genres': ['Drama']}])
        self.assertEqual(resolver.resolve('The Secret Garden (1993)')['raw_item_id'], 6)

    def test_no_year_word_or_fuzzy_correction(self):
        for title in ('The Secret Garden (1994)', 'Secret Garden (1993)', 'The Secrt Garden (1993)',
                      'The The Secret Garden (1993)', 'A Secret Garden, The (1993)'):
            self.assertEqual(self.resolver.resolve(title)['resolution_status'], 'unresolved')

    def test_unresolved_formal_invalid(self):
        result, diagnostic = self.evaluate(['Unknown (1990)', 'Target (1998)'])
        self.assertFalse(result['STRICT_EVALUATION_VALID'])
        self.assertIsNone(result['metrics']['IoI'])
        self.assertIsNone(result['metrics']['Coherence'])
        self.assertIsNone(diagnostic)
        self.assertEqual(self.spy.calls, [])

    def test_diagnostic_not_in_formal_mean(self):
        invalid, diagnostic = self.evaluate(['Unknown (1990)', 'Bridge, A (1991)'], True)
        self.assertTrue(diagnostic['DIAGNOSTIC_ONLY'])
        main, _ = aggregate([invalid, diagnostic])
        self.assertEqual(main['valid_path_count'], 0)
        self.assertIsNone(main['IoI_mean'])

    def test_ioi_formula_exact(self):
        self.assertEqual(interest_increase(0.1, 0.2), math.log(0.2 + 1e-12) - math.log(0.1 + 1e-12))
        self.assertEqual(IOI_EPS, 1e-12)
        self.assertTrue(math.isfinite(interest_increase(0, 0.1)))

    def test_ior_formula_exact(self):
        self.assertEqual(rank_increase(1000, 300), 700)

    def test_rank_direction_and_delegation(self):
        self.assertLess(rank_increase(30, 100), 0)
        result, _ = self.evaluate(['Bridge, A (1991)'])
        self.assertEqual(result['metrics']['IoR'], 60)
        self.assertEqual([c for c in self.spy.calls if c[0] == 'rank'], [('rank', (1,), 4), ('rank', (1, 3), 4)])

    def test_proxy_sequential_update(self):
        result, _ = self.evaluate(['Secret Garden, The (1993)', 'Bridge, A (1991)', 'Target (1998)'])
        self.assertEqual([c for c in self.spy.calls if c[0] == 'proxy'], [('proxy', (1,), 2), ('proxy', (1, 2), 3)])
        self.assertEqual(result['metrics']['path_item_scores'], [0.1, 0.2])
        self.assertAlmostEqual(result['metrics']['ProxyAcceptability'], 0.15)

    def test_empty_intermediates(self):
        result, _ = self.evaluate(['Target (1998)'])
        self.assertIsNone(result['metrics']['ProxyAcceptability'])
        self.assertFalse(result['metrics']['PROXY_ACCEPTABILITY_DEFINED'])
        self.assertIsNone(result['metrics']['Coherence'])
        self.assertFalse(result['metrics']['COHERENCE_DEFINED'])
        self.assertEqual(result['metrics']['IoI'], 0)
        self.assertEqual(result['metrics']['IoR'], 0)

    def test_coherence_overlap(self):
        result = genre_coherence([self.resolver.resolve('Secret Garden, The (1993)'), self.resolver.resolve('Bridge, A (1991)')], self.target)
        self.assertEqual(result['Coherence'], 1)
        self.assertEqual(result['adjacent_pair_count'], 2)

    def test_coherence_excludes_history_edge(self):
        result, _ = self.evaluate(['Bridge, A (1991)', 'Target (1998)'])
        # History Comedy and first bridge Drama/Romance do not overlap; not counted.
        self.assertEqual(result['metrics']['Coherence'], 1)
        self.assertEqual(result['metrics']['adjacent_pair_count'], 1)

    def test_coherence_includes_last_to_target(self):
        result = genre_coherence([self.resolver.resolve('Other (1995)')], self.target)
        self.assertEqual(result['Coherence'], 0)
        self.assertEqual(result['pairs'][0]['right_raw_id'], 4)

    def test_unknown_not_new(self):
        result = self.prepare(['Unknown (1990)', 'History (1990)', 'Target (1998)'])
        self.assertEqual(result['mechanism_metrics']['NEW_INTERMEDIATE_COUNT'], 0)
        self.assertEqual(result['mechanism_metrics']['HISTORY_REUSE_RATE'], 0.5)

    def test_candidate_canonical_identity_and_occurrences(self):
        result = self.prepare(['A Bridge (1991)', 'Bridge, A (1991)', 'Target (1998)'])
        self.assertEqual(result['mechanism_metrics']['BRIDGE_CANDIDATE_USAGE_COUNT'], 2)
        self.assertEqual(result['validity']['duplicate_item_count'], 1)

    def test_valid_only_mean(self):
        valid, _ = self.evaluate(['Bridge, A (1991)', 'Target (1998)'])
        invalid, _ = self.evaluate(['Unknown (1990)', 'Target (1998)'])
        main, mechanism = aggregate([valid, invalid])
        self.assertEqual(main['valid_path_count'], 1)
        self.assertEqual(main['IoI_mean'], valid['metrics']['IoI'])
        self.assertEqual(main['SR'], 1)
        self.assertEqual(mechanism['NEW_INTERMEDIATE_COUNT_mean'], 1)

    def test_raw_not_mutated(self):
        parsed = {'path': ['The Secret Garden, The (1993)', 'Target (1998)'], 'parse_success': True}
        original = copy.deepcopy(parsed)
        prepare_path(parsed, self.resolver, self.target, {1}, {3})
        self.assertEqual(parsed, original)

    def test_null_mean_denominator(self):
        result, _ = self.evaluate(['Target (1998)'])
        main, _ = aggregate([result])
        self.assertEqual(main['valid_path_count'], 1)
        self.assertIsNone(main['ProxyAcceptability_mean'])
        self.assertEqual(main['metric_defined_path_counts']['ProxyAcceptability'], 0)

    def test_missing_target_sr(self):
        result, _ = self.evaluate(['Bridge, A (1991)'])
        main, _ = aggregate([result])
        self.assertEqual(main['SR'], 0)
        self.assertEqual(result['FORMAL_METRIC_STATUS'], 'valid')

    def test_parse_failure_excluded_sr(self):
        prepared = prepare_path({'path': [], 'parse_success': False}, self.resolver, self.target, {1}, {3})
        result, _ = evaluate_prepared(prepared, self.target, [1], FakeAdapter(), self.spy)
        main, _ = aggregate([result])
        self.assertIsNone(main['SR'])
        self.assertEqual(main['parsed_path_count'], 0)

    def test_real_secret_garden_unique(self):
        resolver = TitleResolver.from_movies_dat(ROOT / 'dataset/ml-1m/movies.dat')
        resolved = resolver.resolve('The Secret Garden, The (1993)')
        self.assertEqual(resolved['resolution_status'], 'resolved')
        self.assertEqual(resolved['resolved_title'], 'Secret Garden, The (1993)')
        self.assertEqual(resolved['raw_item_id'], 531)


if __name__ == '__main__':
    unittest.main()

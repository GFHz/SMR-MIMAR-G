"""Pilot freeze tests; synthetic catalog fixtures are not extra pilot users."""
import copy
import random
import unittest
from collections import Counter
from unittest.mock import patch

from repro.experiments.pilot.freeze_pilot_manifest import (
    PILOT_SEED, ROOT, MANIFEST, load_inputs, build_manifest, serialized,
    payload_sha256, interest_profile, build_bridge, read)
from repro.methods.mi_bridge.retrieve_bridge_movies import retrieve_bridge_movies
from repro.methods.mi_bridge.find_fallback_bridge import find_fallback_bridge


class PilotManifestTests(unittest.TestCase):
    prebuilt = None

    @classmethod
    def setUpClass(cls):
        if cls.prebuilt is None:
            cls.inputs = load_inputs()
            cls.manifest = build_manifest(cls.inputs)
        else:
            cls.inputs, cls.manifest = cls.prebuilt
        cls.ratings, _, cls.movies = cls.inputs
        cls.catalog = {m['id']: m for m in cls.movies}

    def test_exactly_ten_unique_users(self):
        ids = self.manifest['pilot_user_ids']
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        self.assertEqual(ids, [u['user_id'] for u in self.manifest['users']])

    def test_full_eligible_pool_and_single_seed_draw(self):
        counts = self.ratings[self.ratings[2] >= 4][0].value_counts()
        eligible = sorted(int(uid) for uid, count in counts.items() if 1 <= uid <= 6040 and count > 20)
        self.assertEqual(self.manifest['eligible_user_ids'], eligible)
        self.assertEqual(self.manifest['pilot_user_ids'], random.Random(PILOT_SEED).sample(eligible, 10))

    def test_original_last_twenty_positive_items(self):
        for user in self.manifest['users']:
            rows = self.ratings[(self.ratings[2] >= 4) & (self.ratings[0] == user['user_id'])].sort_values(3)
            self.assertGreater(len(rows), 20)
            self.assertEqual(user['positive_interaction_count'], len(rows))
            self.assertEqual(user['positive_movie_ids'], rows[1].tolist())
            self.assertEqual(user['history_length'], 20)
            self.assertEqual([m['movie_id'] for m in user['history']], rows[1].tolist()[-20:])
            self.assertEqual([m['timestamp'] for m in user['history_rating_timestamps']], rows[3].tolist()[-20:])

    def test_target_outside_all_positive(self):
        for user in self.manifest['users']:
            self.assertNotIn(user['target']['movie_id'], user['positive_movie_ids'])

    def test_target_catalog_metadata(self):
        for user in self.manifest['users']:
            target = user['target']
            actual = self.catalog[target['movie_id']]
            self.assertEqual(target['title'], actual['title'])
            self.assertEqual(target['genres'], actual['genres'])

    def test_independent_target_seed_first_legal_draw(self):
        for user in self.manifest['users']:
            rng, count = random.Random(PILOT_SEED + user['user_id']), 0
            while True:
                item = rng.randint(1, 3952)
                count += 1
                if item in self.catalog and item not in user['positive_movie_ids']:
                    break
            self.assertEqual(user['target']['movie_id'], item)
            self.assertEqual(user['target_draw_count'], count)

    def test_top_five_history_only_deterministic(self):
        for user in self.manifest['users']:
            counts = Counter(g for movie in user['history'] for g in movie['genres'])
            stats = [{'genre': g, 'count': n, 'frequency': n / 20}
                     for g, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
            self.assertEqual(user['interest_profile']['genre_statistics'], stats)
            self.assertEqual(user['interest_profile']['top5_interests'], stats[:5])

    def test_pair_scores_and_ranking_unchanged(self):
        for user in self.manifest['users']:
            counts = user['interest_profile']['genre_counts']
            pairs = user['bridge']['ranked_pairs']
            self.assertEqual(pairs, sorted(pairs, key=lambda p: (-p['bridge_score'], -p['user_frequency'],
                p['user_existing_interest'], p['target_related_interest'])))
            for p in pairs:
                i, g = p['user_existing_interest'], p['target_related_interest']
                self.assertNotEqual(i, g)
                self.assertAlmostEqual(p['bridge_score'], counts[i] / 20 + 1 - counts.get(g, 0) / 20)

    def test_selected_first_ranked_feasible(self):
        for user in self.manifest['users']:
            history = [self.catalog[m['movie_id']] for m in user['history']]
            target = self.catalog[user['target']['movie_id']]
            expected_pair, expected_type = None, 'unavailable'
            for pair in user['bridge']['ranked_pairs']:
                direct = retrieve_bridge_movies(pair, self.movies, history, target)
                if direct['candidate_count']:
                    expected_pair, expected_type = pair, 'direct'
                    break
                fallback = find_fallback_bridge(pair['user_existing_interest'], pair['target_related_interest'], self.movies, history, target)
                if fallback['selected_fallback']:
                    expected_pair, expected_type = pair, 'fallback'
                    break
            self.assertEqual(user['bridge_status'], expected_type)
            self.assertEqual(user['bridge']['selected_rank'], expected_pair['rank'] if expected_pair else None)
            if expected_pair:
                self.assertEqual(user['bridge']['selected_existing_interest'], expected_pair['user_existing_interest'])
                self.assertEqual(user['bridge']['selected_target_interest'], expected_pair['target_related_interest'])

    def test_direct_candidates_excluded_and_ordered(self):
        for user in self.manifest['users']:
            excluded = {m['movie_id'] for m in user['history']} | {user['target']['movie_id']}
            bridge = user['bridge']
            candidates = bridge['direct_candidates']
            self.assertEqual([m['id'] for m in candidates], sorted(m['id'] for m in candidates))
            self.assertLessEqual(len(candidates), 20)
            for m in candidates:
                self.assertNotIn(m['id'], excluded)
                self.assertTrue({bridge['selected_existing_interest'], bridge['selected_target_interest']} <= set(m['genres']))
            if user['bridge_status'] == 'direct':
                self.assertGreater(bridge['direct_candidate_count'], 0)
                self.assertEqual(len(candidates), min(20, bridge['direct_candidate_count']))

    def test_fallback_only_without_direct(self):
        for user in self.manifest['users']:
            bridge = user['bridge']
            if user['bridge_status'] == 'fallback':
                self.assertEqual(bridge['direct_candidate_count'], 0)
                self.assertGreater(bridge['fallback_left_candidate_count'], 0)
                self.assertGreater(bridge['fallback_right_candidate_count'], 0)
                self.assertNotIn(bridge['fallback_middle_genre'], (bridge['selected_existing_interest'], bridge['selected_target_interest']))
            if user['bridge_status'] == 'direct':
                self.assertIsNone(bridge['fallback_middle_genre'])

    def test_same_seed_full_manifest_bytes(self):
        repeated = build_manifest(self.inputs)
        self.assertEqual(serialized(self.manifest), serialized(repeated))

    def test_hash_stable_and_detects_changes(self):
        self.assertEqual(payload_sha256(self.manifest), self.manifest['manifest_sha256'])
        reordered = dict(reversed(list(self.manifest.items())))
        self.assertEqual(payload_sha256(reordered), payload_sha256(self.manifest))
        changed = copy.deepcopy(self.manifest)
        changed['users'][0]['target']['title'] += ' altered'
        self.assertNotEqual(payload_sha256(changed), payload_sha256(self.manifest))

    def test_unavailable_users_not_replaced(self):
        # Interface fixture only: force unavailability without creating a second cohort.
        with patch('repro.experiments.pilot.freeze_pilot_manifest.select_bridge', return_value={
                'bridge_type': 'unavailable', 'selected_bridge': None, 'fallback_supported': True}):
            unavailable = build_manifest(self.inputs)
        self.assertEqual(unavailable['pilot_user_ids'], self.manifest['pilot_user_ids'])
        self.assertEqual([u['target'] for u in unavailable['users']], [u['target'] for u in self.manifest['users']])
        self.assertEqual(len(unavailable['users']), 10)
        self.assertTrue(all(u['bridge_status'] == 'unavailable' for u in unavailable['users']))

    def fixture(self, direct=False, fallback=False):
        history = [{'id': 1, 'title': 'History', 'genres': ['Comedy']}]
        target = {'id': 9, 'title': 'Target', 'genres': ['Drama']}
        movies = history + [target]
        if direct:
            movies += [{'id': 2, 'title': 'Direct', 'genres': ['Comedy', 'Drama']}]
        if fallback:
            movies += [{'id': 3, 'title': 'Left', 'genres': ['Comedy', 'Adventure']},
                       {'id': 4, 'title': 'Right', 'genres': ['Adventure', 'Drama']}]
        return build_bridge(history, target, interest_profile(history), movies)

    def test_direct_priority_synthetic(self):
        bridge = self.fixture(direct=True, fallback=True)
        self.assertEqual(bridge['bridge_type'], 'direct')
        self.assertIsNone(bridge['fallback_middle_genre'])

    def test_real_method_fallback_serialization_synthetic(self):
        bridge = self.fixture(fallback=True)
        self.assertEqual(bridge['bridge_type'], 'fallback')
        self.assertEqual(bridge['method_bridge_type'], 'multi_hop')
        self.assertEqual(bridge['fallback_middle_genre'], 'Adventure')
        self.assertEqual(bridge['fallback_score'], 1)
        self.assertEqual([m['id'] for m in bridge['fallback_left_candidates']], [3])
        self.assertEqual([m['id'] for m in bridge['fallback_right_candidates']], [4])

    def test_unavailable_synthetic(self):
        bridge = self.fixture()
        self.assertEqual(bridge['bridge_type'], 'unavailable')
        self.assertIsNone(bridge['selected_existing_interest'])

    def test_saved_manifest_if_present(self):
        if MANIFEST.exists():
            self.assertEqual(MANIFEST.read_bytes(), serialized(self.manifest))
            self.assertEqual(read(MANIFEST)['manifest_sha256'], payload_sha256(self.manifest))


if __name__ == '__main__':
    unittest.main()

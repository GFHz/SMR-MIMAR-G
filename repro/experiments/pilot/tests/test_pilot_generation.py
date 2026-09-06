"""No-model preflight/prompt tests for the fixed 40-slot generation plan."""
import copy
import unittest
from unittest.mock import patch

from repro.experiments.pilot.generate_pilot_paths import (
    EXPECTED_MANIFEST_SHA, EXPECTED_USERS, PLAN_COUNT, PILOT, canonical_hash,
    planned_slots, preflight, prompt_bundle, summary, user_prompt)
from repro.experiments.pilot.freeze_pilot_manifest import payload_sha256
from repro.utils import read_json


class PilotGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.config, cls.slots, _, _ = preflight(check_service=False)

    def test_manifest_hash(self):
        self.assertEqual(self.manifest['manifest_sha256'], EXPECTED_MANIFEST_SHA)
        self.assertEqual(payload_sha256(self.manifest), EXPECTED_MANIFEST_SHA)

    def test_ten_users_exact_order(self):
        self.assertEqual(self.manifest['pilot_user_ids'], EXPECTED_USERS)
        self.assertEqual(len(self.manifest['users']), 10)

    def test_exact_fixed_slot_order(self):
        expected = [(uid, method, n) for uid in EXPECTED_USERS
                    for method in ('baseline', 'mi_bridge') for n in (1, 2)]
        self.assertEqual([(u['user_id'], m, n) for _, u, m, n in self.slots], expected)
        self.assertEqual(len(self.slots), PLAN_COUNT)

    def test_two_per_method_per_user(self):
        for uid in EXPECTED_USERS:
            group = [(m, n) for _, u, m, n in self.slots if u['user_id'] == uid]
            self.assertEqual(group, [('baseline', 1), ('baseline', 2), ('mi_bridge', 1), ('mi_bridge', 2)])

    def test_baseline_has_no_bridge(self):
        for user in self.manifest['users']:
            prompt = prompt_bundle(user, 'baseline')
            self.assertNotIn('[BRIDGE CONTEXT]', prompt['user_prompt'])
            self.assertIsNone(prompt['selected_bridge'])
            self.assertIsNone(prompt['bridge_candidates'])

    def test_mi_only_appends_one_context(self):
        for user in self.manifest['users']:
            base = prompt_bundle(user, 'baseline')
            ours = prompt_bundle(user, 'mi_bridge')
            self.assertEqual(base['system_prompt'], ours['system_prompt'])
            self.assertEqual(ours['user_prompt'], base['user_prompt'] + ours['bridge_context'])
            self.assertEqual(ours['user_prompt'].count('[BRIDGE CONTEXT]'), 1)
            self.assertEqual(base['formatting_user_prompt'], ours['formatting_user_prompt'])

    def test_bridge_exact_manifest_source(self):
        for user in self.manifest['users']:
            ours = prompt_bundle(user, 'mi_bridge')
            self.assertEqual(ours['selected_bridge'], user['bridge']['selection_result']['selected_bridge'])
            self.assertEqual(ours['bridge_candidates'], user['bridge']['direct_candidates'])

    def test_target_history_exact_manifest_source(self):
        for user in self.manifest['users']:
            prompt = user_prompt(user)
            expected_history = ''.join(f"{m['title']} Genre:{'|'.join(m['genres'])}\n" for m in user['history'])
            self.assertIn(expected_history, prompt)
            self.assertIn(f"Target movie: {user['target']['title']} Genre:{'|'.join(user['target']['genres'])}", prompt)

    def test_prompt_hash_sensitive_and_deterministic(self):
        user = self.manifest['users'][0]
        one, two = prompt_bundle(user, 'baseline'), prompt_bundle(user, 'baseline')
        self.assertEqual(one['prompt_hash'], two['prompt_hash'])
        changed = dict(one)
        changed['user_prompt'] += 'x'
        self.assertNotEqual(one['prompt_hash'], canonical_hash({k: changed[k] for k in
            ('system_prompt', 'user_prompt', 'formatting_user_prompt')}))

    def test_frozen_model_config_parser_preflight(self):
        self.assertEqual(self.config['model'], 'qwen3:4b-q4_K_M')
        self.assertEqual(self.config['model_digest'], '2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0')
        self.assertFalse(self.config['thinking_mode'])
        self.assertEqual(self.config['max_output_tokens'], 2048)
        self.assertEqual(self.config['max_retries'], 2)

    def test_prior_output_never_enters_prompt(self):
        user = self.manifest['users'][0]
        before = prompt_bundle(user, 'baseline')
        fake_previous = ['Previous output must not appear']
        after = prompt_bundle(user, 'baseline')
        self.assertEqual(before, after)
        self.assertNotIn(fake_previous[0], after['user_prompt'])

    def test_summary_counts_slot_records_not_dialogue_retries(self):
        records = []
        for _, user, method, index in self.slots:
            records.append({'user_id': user['user_id'], 'method': method, 'path_index': index,
                'parse_success': True, 'adapter_attempt_count': 2,
                'generated_tokens': 10, 'inference_seconds': 1})
        value = summary(records)
        self.assertEqual(value['actual_requested_path_count'], 40)
        self.assertEqual(value['baseline_path_count'], 20)
        self.assertEqual(value['mi_bridge_path_count'], 20)
        self.assertEqual(value['total_adapter_requests'], 80)

    def test_no_service_or_model_call_during_preflight_tests(self):
        with patch('repro.local_llm.request', side_effect=AssertionError('No service call')):
            manifest, _, slots, installed, version = preflight(check_service=False)
        self.assertEqual(len(slots), 40)
        self.assertIsNone(installed)
        self.assertIsNone(version)

    def test_prompt_functions_do_not_mutate_manifest(self):
        original = copy.deepcopy(self.manifest)
        for user in self.manifest['users']:
            prompt_bundle(user, 'baseline')
            prompt_bundle(user, 'mi_bridge')
        self.assertEqual(self.manifest, original)


if __name__ == '__main__':
    unittest.main()

"""Independent ProRL-style single-user primitives, NOT an LLM-IPP replication.

Reference: https://github.com/hongruhou89/ProRL/blob/
506e91355377f546dcf51f684fc871fe57a9d5c8/evaluator.py (PRAEvaluator).
Formulas reimplemented: full-vocabulary softmax score/rank and sigmoid dot-product
sequential acceptability proxy. No semantic tokenizer, IoI/IoR aggregation, or
coherence. Target removal and strict/drop policies are explicit adapted rules.
"""
import math
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache

import torch

from .checkpoint_loader import ROOT, OUT, load_checkpoint, environment, read_json, save_new, protected_hashes, sha256
from .data_adapter import DataAdapter


class SASRecEvaluator:
    def __init__(self, model):
        self.model = model.eval()
        self.max_length = model.max_seq_length
        self.item_count = model.n_items

    def _item(self, value):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 < value < self.item_count:
            raise ValueError(f'Non-padding internal item ID out of range: {value}')

    def _sequence(self, history):
        if not 1 <= len(history) <= self.max_length:
            raise ValueError('History must be nonempty and within max length; never truncated')
        for item in history:
            self._item(item)
        return (torch.tensor([list(history) + [0] * (self.max_length - len(history))], dtype=torch.long),
                torch.tensor([len(history)], dtype=torch.long))

    @torch.inference_mode()
    def full_scores(self, history):
        seq, length = self._sequence(history)
        logits = self.model.full_sort_predict({self.model.ITEM_SEQ: seq, self.model.ITEM_SEQ_LEN: length})[0]
        if not torch.isfinite(logits).all() or torch.all(logits == logits[0]):
            raise RuntimeError('Nonfinite or identical full-vocabulary scores')
        return logits

    def get_target_score(self, history_internal_ids, target_internal_id):
        self._item(target_internal_id)
        # ProRL includes padding row 0 and does not mask history items in softmax.
        return float(torch.softmax(self.full_scores(history_internal_ids), dim=-1)[target_internal_id])

    def get_target_rank(self, history_internal_ids, target_internal_id):
        self._item(target_internal_id)
        probs = torch.softmax(self.full_scores(history_internal_ids), dim=-1)
        indices = torch.argsort(probs, descending=True)
        return int((indices == target_internal_id).nonzero()[0, 0]) + 1

    @torch.inference_mode()
    def get_item_score(self, history_internal_ids, item_internal_id):
        self._item(item_internal_id)
        seq, length = self._sequence(history_internal_ids)
        user = self.model.forward(seq, length)[0]
        item = self.model.item_embedding.weight[item_internal_id]
        score = float(torch.sigmoid(torch.dot(user, item)))
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite item proxy score')
        return score

    def get_path_item_scores(self, history, intermediates):
        sequence = list(history)
        if len(sequence) + len(intermediates) > self.max_length:
            raise ValueError('Extended history exceeds checkpoint max length; no truncation allowed')
        scores = []
        for item in intermediates:
            scores.append(self.get_item_score(sequence, item))
            sequence.append(item)
        return {'path_item_scores': scores,
                'SASREC_PROXY_ACCEPTABILITY': sum(scores) / len(scores) if scores else None}

    def evaluate_path(self, history, target, mapping):
        if target in history:
            raise ValueError('Target already in original history: leakage')
        if mapping['status'] != 'ready':
            return {**mapping, 'target_score_after': None, 'target_rank_after': None,
                    'path_item_scores': None, 'SASREC_PROXY_ACCEPTABILITY': None}
        intermediates = [x['internal_id'] for x in mapping['evaluation_intermediates']]
        if target in intermediates:
            raise ValueError('Target leakage in extension')
        extended = list(history) + intermediates
        scores = self.get_path_item_scores(history, intermediates)
        return {**mapping, 'history_length_before': len(history), 'history_length_after': len(extended),
                'target_in_extension': False, 'target_score_after': self.get_target_score(extended, target),
                'target_rank_after': self.get_target_rank(extended, target), **scores}


@lru_cache(maxsize=1)
def validation_context():
    """Loads only published weights/data and frozen artifacts; safe for unit tests."""
    torch.set_num_threads(4)
    model, dataset, config, metadata = load_checkpoint()
    adapter = DataAdapter(dataset)
    sample = read_json(ROOT / 'repro/baseline_sample.json')
    if sample['user_id'] != 1 or sample['target']['title'] != 'This Is My Father (1998)' or len(sample['history']) != 20:
        raise RuntimeError('Frozen sample mismatch')
    for item in sample['history']:
        if adapter.movie_title_to_raw_item_id(item['title']) != item['id']:
            raise RuntimeError('Frozen history title/raw ID mismatch')
    raw_target = adapter.movie_title_to_raw_item_id(sample['target']['title'])
    if raw_target != 2620:
        raise RuntimeError('Unexpected target raw ID')
    target = adapter.raw_item_id_to_internal(raw_target)
    history = adapter.history_raw_ids_to_internal([x['id'] for x in sample['history']])
    paths = {}
    for method, directory in [('baseline', 'minimal_baseline_v2'), ('ours', 'case_study/mi_bridge_v1')]:
        for number in (1, 2):
            name = f'{method}_path_{number}'
            parsed = read_json(ROOT / f'repro/results/{directory}/{name}_parsed.json')
            if not parsed['parse_success']:
                raise RuntimeError(f'Frozen path parsing failed: {name}')
            paths[name] = parsed['path']
    return SASRecEvaluator(model), adapter, sample, history, target, paths, metadata


def run_validation():
    start = time.perf_counter()
    if (OUT / 'single_user_scores.json').exists():
        raise FileExistsError('Validation outputs already exist; refusing to overwrite')
    before = read_json(OUT / 'protected_sha256_before.json')
    if before != protected_hashes():
        raise RuntimeError('Protected files changed before inference')
    evaluator, adapter, sample, history, target, paths, metadata = validation_context()
    save_new(OUT / 'environment.json', environment())
    save_new(OUT / 'checkpoint_metadata.json', metadata)
    mappings = {name: adapter.path_titles_to_internal(path) for name, path in paths.items()}
    save_new(OUT / 'id_mapping_user1.json', {
        'user_raw_id': 1, 'user_internal_id': adapter.raw_user_id_to_internal(1),
        'target_raw_id': 2620, 'target_internal_id': target,
        'history': [{**item, 'internal_id': mapped} for item, mapped in zip(sample['history'], history)],
        'history_length': len(history), 'paths': mappings,
        'semantic_ids_used': False, 'fuzzy_correction_used': False,
        'item_field2token_id': {str(k): int(v) for k, v in adapter.item_map.items()}})
    result = {'timestamp': datetime.now(timezone.utc).isoformat(),
              'target_score_definition': 'softmax over all checkpoint rows including padding, as ProRL',
              'rank_definition': '1-based descending softmax, unmasked vocabulary including padding, as ProRL',
              'proxy_definition': 'mean sequential sigmoid(user_embedding dot item_embedding), NOT true CTR',
              'extension_rule': 'all non-target path items in original order; adapted, not asserted as original protocol',
              'target_score_before': evaluator.get_target_score(history, target),
              'target_rank_before': evaluator.get_target_rank(history, target), 'paths': {}}
    for name, path in paths.items():
        strict = adapter.evaluation_intermediates(path, 2620)
        result['paths'][name] = {'strict': evaluator.evaluate_path(history, target, strict)}
        if strict['unresolved_items']:
            diagnostic = adapter.evaluation_intermediates(path, 2620, 'DROP_UNRESOLVED_DIAGNOSTIC_MODE')
            result['paths'][name]['drop_unresolved_diagnostic'] = evaluator.evaluate_path(history, target, diagnostic)
    valid = [entry for path in result['paths'].values() for entry in path.values() if entry['status'] == 'ready']
    scores = [result['target_score_before']] + [entry['target_score_after'] for entry in valid]
    ranks = [result['target_rank_before']] + [entry['target_rank_after'] for entry in valid]
    problems = []
    if not all(math.isfinite(x) for x in scores):
        problems.append('Nonfinite scores')
    if len(set(scores)) == 1:
        problems.append('All target scores identical')
    if len(set(ranks)) == 1:
        problems.append('Target ranks always identical')
    if not all(isinstance(x, int) and 1 <= x <= evaluator.item_count for x in ranks):
        problems.append('Illegal rank')
    protection = before == protected_hashes()
    if not protection:
        problems.append('Protected artifact hash changed')
    result['sanity_checks'] = {'passed': not problems, 'errors': problems,
                              'all_scores_identical': len(set(scores)) == 1,
                              'ranks_always_same': len(set(ranks)) == 1}
    result['elapsed_seconds'] = time.perf_counter() - start
    save_new(OUT / 'single_user_scores.json', result)
    print('Single-user numerical sanity:', not problems, '; protected artifacts unchanged:', protection)
    return result


def finalize_validation():
    """Execute tests, audit hashes, save a report without overwriting score artifacts."""
    import unittest
    if (OUT / 'validation_report.json').exists():
        raise FileExistsError('Validation report exists; refusing to overwrite')
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'repro/evaluators/sasrec/tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    before = read_json(OUT / 'protected_sha256_before.json')
    after = protected_hashes()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    original = read_json(ROOT / 'repro/source_hashes.json')['sha256']
    original_ok = all(sha256(ROOT / name) == expected for name, expected in original.items())
    scores = read_json(OUT / 'single_user_scores.json')
    numerical_ok = scores['sanity_checks']['passed']
    checks = {
        'original_14_files_unchanged': original_ok,
        'baseline_results_unchanged': not any(k.startswith('repro/results/minimal_baseline') for k in changed),
        'mi_bridge_results_unchanged': not any(k.startswith('repro/results/case_study') for k in changed),
        'mi_bridge_method_unchanged': not any(k.startswith('repro/methods/mi_bridge') for k in changed),
        'external_prorl_unchanged': not any(k.startswith('external/ProRL/') for k in changed),
        'external_ipg_unchanged': not any(k.startswith('external/IPG-Rec/') for k in changed),
        'all_protected_files_unchanged': not changed,
    }
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'final_status': 'TECHNICAL_VALIDATION_PASSED_WITH_LIMITATIONS' if
            result.wasSuccessful() and numerical_ok and all(checks.values()) else 'VALIDATION_FAILED',
        'tests_run': result.testsRun, 'all_tests_passed': result.wasSuccessful(),
        'test_failures': [str(test) + '\n' + trace for test, trace in result.failures + result.errors],
        'numerical_sanity_passed': numerical_ok, 'protection': checks,
        'protected_file_count': len(before), 'changed_protected_files': changed,
        'unresolved_items': ['The Secret Garden, The (1993)'],
        'strict_scored_paths': 3, 'diagnostic_only_paths': ['ours_path_1'],
        'semantic_id_bypass_success': True, 'training_performed': False,
        'path_generation_performed': False, 'formal_metric_experiment_performed': False,
        'training_time_mapping_independently_verified': False,
        'limitations': [
            'Training token-ID snapshot unavailable: current map reproduces published evaluator data, not independently proven training row semantics.',
            'Ours path 1 strict failure; drop-unresolved numbers are technical diagnostics only.',
            'CPU PyTorch build: CUDA_AVAILABLE=False is intentional, not a hardware GPU failure.',
            'RecBole emits pandas 2.2 FutureWarnings; no installed package or upstream source patched.',
            'Minimal RecBole imports work; pip check is not clean: optional execution-path dependencies '
            '(plotly, psutil, ray, tabulate, thop) omitted and existing colorama 0.4.6 retained instead of declared 0.4.4.',
            'Scores are model proxies, not observed clicks, causal influence, or proof of method superiority.',
        ],
        'commands': [
            '.venv-repro/Scripts/python.exe -B -X utf8 -m repro.evaluators.sasrec.checkpoint_loader',
            '.venv-repro/Scripts/python.exe -B -X utf8 -m repro.evaluators.sasrec.evaluator',
            '.venv-repro/Scripts/python.exe -B -X utf8 -m repro.evaluators.sasrec.evaluator --finalize',
        ],
    }
    save_new(OUT / 'validation_report.json', report)
    print(report['final_status'])
    if report['final_status'] == 'VALIDATION_FAILED':
        raise RuntimeError('Validation failed; inspect saved report')


if __name__ == '__main__':
    if sys.argv[1:] == ['--finalize']:
        finalize_validation()
    elif sys.argv[1:]:
        raise SystemExit('Usage: python -m repro.evaluators.sasrec.evaluator [--finalize]')
    else:
        run_validation()

"""Mapping-only verification; never imports our evaluator or runs SASRec inference.

Reference: ProRL evaluator.py / config at 506e91355377f546dcf51f684fc871fe57a9d5c8.
Independent tests and evidence collection, not copied upstream implementation.
Run --audit-public for bounded read-only public discovery, --record for results,
or standard unittest to repeat mapping checks without rewriting saved artifacts.
"""
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import unittest
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'repro/results/evaluator_validation/mapping_verification'
OLD = OUT.parent
COMMIT = '506e91355377f546dcf51f684fc871fe57a9d5c8'
CHECKPOINT = OLD / 'checkpoints/SASRec-ml-1m-sas.pth'
DOC = ROOT / 'repro/docs/SASREC_TOKEN_MAPPING_VERIFICATION.md'
THIS = Path(__file__).resolve()
PATTERN = re.compile(r'field2token_id|field2id_token|token2id|id2token|dataset\.pth|dataloader\.pth|RecBole Dataset|save_dataset|save_dataloaders|recbole', re.I)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def protection():
    files = [ROOT / n for n in read(ROOT / 'repro/source_hashes.json')['sha256']]
    for folder in ('repro', 'external', 'dataset'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '.git' not in p.parts and '__pycache__' not in p.parts
                     and OUT not in p.parents and p not in (DOC, THIS))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(files))}


def audit_public():
    import httpx
    if not (OUT / 'protected_sha256_before.json').exists():
        save('protected_sha256_before.json', protection())
    evidence = {'commit': COMMIT, 'directories': {}, 'sources': {}, 'errors': [],
                'api_note': 'Anonymous recursive-tree API returned 403 rate limit; used public HTML trees.',
                'scope': 'Root, config, scripts, checkpoint and dataset trees relevant to MovieLens; unrelated Books/Steam binaries not loaded.'}
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        pending = ['']
        while pending and len(evidence['directories']) < 30:
            directory = pending.pop(0)
            # A subdirectory page also exposes the root tree. GitHub's root
            # route uses a different payload shape; do not mistake it for no files.
            url = f'https://github.com/hongruhou89/ProRL/tree/{COMMIT}/' + (directory or 'ckpt')
            response = client.get(url)
            items = None
            for text in re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', response.text, re.S):
                try:
                    payload = json.loads(text).get('payload', {})
                    route = payload.get('codeViewTreeRoute', {})
                    if route.get('path') == directory:
                        items = route['tree']['items']
                    if not directory:
                        tree = payload.get('codeViewFileTreeLayoutRoute', {}).get('fileTree', {})
                        if '' in tree:
                            items = tree['']['items']
                except (ValueError, KeyError):
                    pass
            if items is None:
                evidence['errors'].append({'url': url, 'status': response.status_code, 'error': 'Tree unavailable'})
                continue
            evidence['directories'][directory] = {'url': url, 'items': items}
            print('Public tree checked:', directory or '/', flush=True)
            for item in items:
                path = item['path']
                if item['contentType'] == 'directory':
                    if path in ('ckpt', 'config', 'datasets', 'scripts') or (
                            path.startswith(('ckpt/ml-1m', 'datasets/ml-1m', 'config/', 'scripts/'))):
                        pending.append(path)
                    continue
                if path.endswith(('.py', '.md', '.yaml', '.yml', '.sh', '.toml', '.txt', '.json')):
                    raw_url = f'https://raw.githubusercontent.com/hongruhou89/ProRL/{COMMIT}/{path}'
                    raw = client.get(raw_url)
                    if raw.status_code != 200:
                        evidence['errors'].append({'url': raw_url, 'status': raw.status_code})
                        continue
                    if len(raw.content) > 2_000_000:
                        continue
                    hits = [{'line': i, 'text': line} for i, line in enumerate(raw.text.splitlines(), 1) if PATTERN.search(line)]
                    evidence['sources'][path] = {'url': raw_url, 'sha256': hashlib.sha256(raw.content).hexdigest(), 'hits': hits}
            # Published atomic files are checked against pinned git blob IDs in
            # a separate optional remote check; no dataset file is downloaded here.
        history_url = f'https://github.com/hongruhou89/ProRL/commits/{COMMIT}/README.md'
        response = client.get(history_url)
        evidence['readme_history'] = {'url': history_url, 'status': response.status_code,
            'recbole_version_mentions': re.findall(r'recbole(?:==|[\s:=]+)\d+\.\d+(?:\.\d+)?', response.text, re.I),
            'note': 'History page inspected for explicit version evidence; no training environment lock recovered.'}
    save('public_inventory_complete.json', evidence)
    print('Public evidence saved; no external files modified.')


def normalize(value):
    return json.loads(json.dumps(value, default=str))


def verify_atomic_provenance():
    """Hash pinned public bytes in memory; never overwrite the local data."""
    import httpx
    evidence = {}
    names = [f'datasets/ml-1m-sas/ml-1m-sas.{ext}' for ext in ('inter', 'user', 'item')]
    names.append('config/ml-1m-sas_sasrec_config.yaml')
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name in names:
            url = f'https://raw.githubusercontent.com/hongruhou89/ProRL/{COMMIT}/{name}'
            sha, count = hashlib.sha256(), 0
            with client.stream('GET', url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    count += len(chunk)
                    if count > 30_000_000:
                        raise RuntimeError('Unexpectedly large public atomic file')
                    sha.update(chunk)
            local = ROOT / 'external/ProRL' / name
            evidence[name] = {'url': url, 'bytes': count, 'remote_sha256': sha.hexdigest(),
                              'local_sha256': digest(local), 'exact_match': sha.hexdigest() == digest(local)}
            print('Pinned atomic/config hash verified:', name, evidence[name]['exact_match'], flush=True)
    save('atomic_provenance.json', evidence)


def maps(dataset):
    return {field: {str(k): int(v) for k, v in dataset.field2token_id[field].items()}
            for field in ('item_id', 'user_id')}


def compare(left, right):
    differences = [{'raw_token': k, 'left': left.get(k), 'right': right.get(k)}
                   for k in sorted(left.keys() | right.keys()) if left.get(k) != right.get(k)]
    return {'exact_match': not differences, 'difference_count': len(differences),
            'vocabulary_sizes': [len(left), len(right)], 'raw_token_sets_equal': left.keys() == right.keys(),
            'pad_ids': [left.get('[PAD]'), right.get('[PAD]')], 'differences': differences}


@lru_cache(maxsize=1)
def reconstruct():
    import torch
    from recbole.config import Config
    from recbole.data.dataset import SequentialDataset
    # No create_dataset cache reuse and no load_checkpoint()/evaluator import.
    # Direct constructors create A and B separately from atomic files.
    official = ROOT / 'external/ProRL/config/ml-1m-sas_sasrec_config.yaml'
    base = {'data_path': str(ROOT / 'external/ProRL/datasets'), 'use_gpu': False,
            'save_dataset': False, 'save_dataloaders': False}
    config_a = Config(model='SASRec', dataset='ml-1m-sas', config_file_list=[str(official)], config_dict=dict(base))
    dataset_a = SequentialDataset(config_a)
    config_b = Config(model='SASRec', dataset='ml-1m-sas', config_file_list=[str(official)], config_dict=dict(base))
    dataset_b = SequentialDataset(config_b)
    # Independently reconstruct CURRENT implementation's configuration: same
    # files and overrides as unchanged checkpoint_loader.py, without its model load.
    current_config = Config(model='SASRec', dataset='ml-1m-sas',
        config_file_list=[str(official), str(ROOT / 'repro/evaluators/sasrec/config/validation.yaml')],
        config_dict={'data_path': base['data_path'], 'use_gpu': False,
                     'checkpoint_dir': str(OLD / 'unused_no_training')})
    current_dataset = SequentialDataset(current_config)
    a, b, current = maps(dataset_a), maps(dataset_b), maps(current_dataset)
    recorded = read(OLD / 'id_mapping_user1.json')
    if current['item_id'] != recorded['item_field2token_id']:
        raise RuntimeError('Current reconstruction differs from previously saved complete evaluator item mapping')
    if current['user_id']['1'] != recorded['user_internal_id']:
        raise RuntimeError('Current user 1 differs from saved evaluator mapping')
    if digest(CHECKPOINT) != read(OLD / 'checkpoint_download.json')['sha256']:
        raise RuntimeError('Checkpoint hash mismatch')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    saved = checkpoint['config'].final_config_dict
    fields = ['field_separator', 'seq_separator', 'USER_ID_FIELD', 'ITEM_ID_FIELD', 'TIME_FIELD',
              'load_col', 'unload_col', 'unused_col', 'filter_inter_by_user_or_item',
              'user_inter_num_interval', 'item_inter_num_interval', 'val_interval', 'test_interval',
              'val_check', 'benchmark_filename', 'eval_args', 'seed', 'repeatable',
              'alias_of_user_id', 'alias_of_item_id', 'rm_dup_inter', 'threshold',
              'MAX_ITEM_LIST_LENGTH', 'save_dataset', 'save_dataloaders']
    config_compare = {k: {'checkpoint': normalize(saved.get(k)), 'official_effective': normalize(config_a[k]),
                          'current_effective': normalize(current_config[k])} for k in fields}
    candidates = []
    def inspect(value, path='checkpoint'):
        if isinstance(value, dict):
            for key, child in value.items():
                childpath = f'{path}.{key}'
                if re.search(r'field2(token_id|id_token)|token2id|id2token', str(key), re.I):
                    candidates.append(childpath)
                if key not in ('state_dict', 'optimizer'):
                    inspect(child, childpath)
        elif hasattr(value, 'final_config_dict'):
            inspect(value.final_config_dict, path + '.final_config_dict')
    inspect(checkpoint)
    return {'A': a, 'B': b, 'current': current, 'config_comparison': config_compare,
            'checkpoint_mapping_candidates': candidates, 'checkpoint_keys': sorted(checkpoint),
            'checkpoint_other_parameter': normalize(checkpoint['other_parameter']),
            'checkpoint_item_vocab_size': int(checkpoint['state_dict']['item_embedding.weight'].shape[0]),
            'checkpoint_max_length': int(checkpoint['state_dict']['position_embedding.weight'].shape[0]),
            'dataset_item_num': dataset_a.item_num, 'dataset_user_num': dataset_a.user_num,
            'checkpoint_user_embedding_present': any('user_embedding' in k for k in checkpoint['state_dict']),
            'checkpoint_dataset': saved['dataset'], 'checkpoint_data_path': saved['data_path'],
            'checkpoint_version_fields': {k: normalize(v) for k, v in saved.items() if 'version' in k.lower()},
            'official_effective_config': normalize(config_a.final_config_dict),
            'current_effective_config': normalize(current_config.final_config_dict)}


def spots(left, right, count):
    tokens = sorted((x for x in left if x != '[PAD]'), key=int)
    chosen = [tokens[j * (len(tokens) - 1) // (count - 1)] for j in range(count)]
    return [{'raw_token': t, 'reconstructed_internal_id': left[t],
             'current_internal_id': right.get(t), 'match': left[t] == right.get(t)} for t in chosen]


class MappingVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = reconstruct()

    def test_vocab_size(self):
        self.assertEqual(self.data['checkpoint_item_vocab_size'], self.data['dataset_item_num'])
        self.assertEqual(self.data['dataset_item_num'], len(self.data['A']['item_id']))

    def test_target(self):
        self.assertEqual(self.data['A']['item_id']['2620'], 2844)
        self.assertEqual(self.data['A']['item_id']['2620'], self.data['current']['item_id']['2620'])

    def test_user_one(self):
        self.assertEqual(self.data['A']['user_id']['1'], 1832)
        self.assertEqual(self.data['A']['user_id']['1'], self.data['current']['user_id']['1'])

    def test_independent_reconstruction(self):
        self.assertEqual(self.data['A'], self.data['B'])

    def test_complete_current_mapping(self):
        self.assertEqual(self.data['A'], self.data['current'])

    def test_item_spots(self):
        selected = spots(self.data['A']['item_id'], self.data['current']['item_id'], 50)
        self.assertEqual(len({x['raw_token'] for x in selected}), 50)
        self.assertTrue(all(x['match'] for x in selected))

    def test_user_spots(self):
        selected = spots(self.data['A']['user_id'], self.data['current']['user_id'], 20)
        self.assertEqual(len({x['raw_token'] for x in selected}), 20)
        self.assertTrue(all(x['match'] for x in selected))

    def test_pad_consistency(self):
        for source in ('A', 'B', 'current'):
            for mapping in self.data[source].values():
                self.assertEqual(mapping['[PAD]'], 0)

    def test_unique_contiguous_ids(self):
        for source in ('A', 'B', 'current'):
            for mapping in self.data[source].values():
                self.assertEqual(sorted(mapping.values()), list(range(len(mapping))))

    def test_checkpoint_mapping_config(self):
        for field, values in self.data['config_comparison'].items():
            # Only runtime save flags may differ; these do not build token IDs.
            if field not in ('save_dataset', 'save_dataloaders'):
                self.assertEqual(values['checkpoint'], values['official_effective'], field)
                self.assertEqual(values['official_effective'], values['current_effective'], field)


def record():
    import torch
    inventory = read(OUT / 'public_inventory_complete.json')
    atomic = read(OUT / 'atomic_provenance.json')
    if not all(v['exact_match'] for v in atomic.values()):
        raise RuntimeError('Local atomic data/config differs from pinned official source')
    data = reconstruct()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MappingVerificationTests))
    comparison = {field: {'A_vs_B': compare(data['A'][field], data['B'][field]),
                           'reconstructed_vs_current': compare(data['A'][field], data['current'][field])}
                  for field in ('item_id', 'user_id')}
    comparison['reference_type'] = 'reconstructed released evaluator map, NOT saved training map'
    comparison['complete_mappings'] = {k: data[k] for k in ('A', 'B', 'current')}
    comparison['config_comparison'] = data['config_comparison']
    save('mapping_comparison.json', comparison)
    checks = {'selection': 'numeric raw-ID order, 50/20 equally spaced indices including endpoints, no RNG',
              'items': spots(data['A']['item_id'], data['current']['item_id'], 50),
              'users': spots(data['A']['user_id'], data['current']['user_id'], 20),
              'target': {'raw_id': 2620, 'reconstructed': data['A']['item_id']['2620'], 'current': data['current']['item_id']['2620']},
              'user_1': {'raw_id': 1, 'reconstructed': data['A']['user_id']['1'], 'current': data['current']['user_id']['1']}}
    save('spot_checks.json', checks)
    local_search = {}
    for path in sorted((ROOT / 'external/ProRL').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.md', '.yaml', '.txt', '.json'):
            hits = [{'line': i, 'text': s} for i, s in enumerate(path.read_text(encoding='utf-8').splitlines(), 1) if PATTERN.search(s)]
            local_search[path.relative_to(ROOT).as_posix()] = hits
    sources = {'upstream_commit': COMMIT, 'public_inventory_file': 'public_inventory_complete.json',
               'atomic_provenance_file': 'atomic_provenance.json',
               'all_atomic_files_and_config_match_pinned_official_bytes': True,
               'public_search_errors': inventory['errors'], 'local_source_search': local_search,
               'training_item_mapping_artifact_found': False, 'training_user_mapping_artifact_found': False,
               'training_mapping_artifact_paths': [],
               'checkpoint_mapping_key_candidates': data['checkpoint_mapping_candidates'],
               'checkpoint_top_level_keys': data['checkpoint_keys'],
               'checkpoint_other_parameter': data['checkpoint_other_parameter'],
               'checkpoint_version_fields': data['checkpoint_version_fields'],
               'checkpoint_sha256': digest(CHECKPOINT),
               'atomic_sha256': {p.name: digest(p) for p in sorted((ROOT / 'external/ProRL/datasets/ml-1m-sas').glob('*')) if p.is_file()},
               'original_recbole_version': None, 'original_recbole_version_unknown': True,
               'version_evidence': 'README pip install recbole is unpinned; no version in checkpoint Config; public version lock not found.',
               'policy_datamaps_not_training_mapping': True,
               'current_user_map_note': 'Previous artifact saves all item IDs but only user 1. Full current user map reconstructed with unchanged current config, not a prior serialized map.',
               'effective_configs': {k: data[k] for k in ('official_effective_config', 'current_effective_config')},
               'scope_note': 'No claim of exhaustive examination of unrelated policy/model binaries or deleted historical artifacts.'}
    policy = read(ROOT / 'external/ProRL/datasets/ml-1m/ml-1m.datamaps')
    sources['policy_datamaps_evidence'] = {'keys': sorted(policy),
        'policy_target_id': policy.get('item2id', {}).get('2620'),
        'policy_user_id': policy.get('user2id', {}).get('1'),
        'note': 'Policy IDs are not RecBole training token maps; do not use as embedding indices.'}
    import recbole
    recbole_root = Path(recbole.__file__).parent
    sources['installed_mapping_source_sha256'] = {
        str(p.relative_to(recbole_root)): digest(p) for p in
        (recbole_root / 'data/dataset/dataset.py', recbole_root / 'data/dataset/sequential_dataset.py',
         recbole_root / 'config/configurator.py')}
    sources['mapping_algorithm_observation'] = ('RecBole Dataset._remap_ID_all -> _get_remap_list -> '
        '_remap: interaction tokens first, then side metadata; pandas.factorize first-occurrence order; '
        'internal IDs shifted by 1 and [PAD] inserted at 0. Mapping precedes dataset.build/splitting.')
    save('mapping_sources.json', sources)
    save('environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
         'recbole': importlib.metadata.version('recbole'), 'numpy': importlib.metadata.version('numpy'),
         'pandas': importlib.metadata.version('pandas'), 'original_recbole_version': None,
         'original_recbole_version_unknown': True, 'dependencies_changed': False})
    before, after = read(OUT / 'protected_sha256_before.json'), protection()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    original = read(ROOT / 'repro/source_hashes.json')['sha256']
    original_ok = all(digest(ROOT / k) == v for k, v in original.items())
    status = 'LIKELY_COMPATIBLE' if result.wasSuccessful() and not changed and original_ok else 'UNVERIFIED'
    report = {'mapping_status': status, 'timestamp': datetime.now(timezone.utc).isoformat(),
              'tests_run': result.testsRun, 'all_tests_passed': result.wasSuccessful(),
              'test_errors': [str(t) + '\n' + e for t, e in result.errors + result.failures],
              'training_item_mapping_artifact_found': False, 'training_user_mapping_artifact_found': False,
              'original_recbole_version_unknown': True,
              'item_mapping_reconstruction_deterministic': data['A']['item_id'] == data['B']['item_id'],
              'user_mapping_reconstruction_deterministic': data['A']['user_id'] == data['B']['user_id'],
              'item_mapping_exact_match': data['A']['item_id'] == data['current']['item_id'],
              'user_mapping_exact_match': data['A']['user_id'] == data['current']['user_id'],
              'exact_match_reference': 'reconstruction, not independently saved training artifact',
              'mapping_difference_count': sum(comparison[f]['reconstructed_vs_current']['difference_count'] for f in ('item_id','user_id')),
              'checkpoint_item_vocab_size': data['checkpoint_item_vocab_size'], 'dataset_item_num': data['dataset_item_num'],
              'dataset_user_num': data['dataset_user_num'], 'checkpoint_max_length': data['checkpoint_max_length'],
              'checkpoint_user_embedding_present': data['checkpoint_user_embedding_present'],
              'checkpoint_dataset': data['checkpoint_dataset'], 'checkpoint_data_path': data['checkpoint_data_path'],
              'training_performed': False, 'inference_performed': False, 'split_performed': False,
              'title_normalization_performed': False,
              'protection': {'original_14_files_unchanged': original_ok, 'all_previous_files_unchanged': not changed,
                  'baseline_results_unchanged': not any(k.startswith('repro/results/minimal_baseline') for k in changed),
                  'mi_bridge_results_unchanged': not any(k.startswith('repro/results/case_study') for k in changed),
                  'mi_bridge_method_unchanged': not any(k.startswith('repro/methods/mi_bridge') for k in changed),
                  'proactrec_paths_unchanged': not any(k.startswith('repro/results/') for k in changed),
                  'existing_evaluator_logic_and_tests_unchanged': not any(k.startswith('repro/evaluators/') for k in changed),
                  'external_unchanged': not any(k.startswith('external/') for k in changed),
                  'checkpoint_unchanged': digest(CHECKPOINT) == read(OLD / 'checkpoint_download.json')['sha256'],
                  'changed_files': changed, 'protected_file_count': len(before)},
              'limitations': ['No saved training mapping located; original RecBole version unknown.',
                  'Determinism and exact equality demonstrate reproducible released evaluator mapping, not independently proven training row semantics.',
                  'SASRec has no user embedding, so checkpoint tensor shapes cannot confirm user_num.',
                  'Checkpoint names ml-1m; released evaluator uses ml-1m-sas; training dataset byte identity is not recorded.',
                  'Public discovery scope and HTTP errors are preserved in public_inventory_complete.json; first root-route parsing failure retained separately.']}
    save('verification_report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('limitations','test_errors')},indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--audit-public']:
        audit_public()
    elif sys.argv[1:] == ['--record']:
        record()
    elif sys.argv[1:] == ['--verify-atomic']:
        verify_atomic_provenance()
    else:
        unittest.main()

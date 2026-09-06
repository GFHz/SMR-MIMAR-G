"""Reference: https://github.com/hongruhou89/ProRL, evaluator.py and SASRec config.

Observed upstream snapshot: 506e91355377f546dcf51f684fc871fe57a9d5c8.
Rewritten checkpoint/data setup; no upstream implementation copied. No training.
Only the explicitly downloaded, pinned official checkpoint is trusted for pickle.
"""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'repro/results/evaluator_validation'
COMMIT = '506e91355377f546dcf51f684fc871fe57a9d5c8'
URL = f'https://raw.githubusercontent.com/hongruhou89/ProRL/{COMMIT}/ckpt/SASRec-ml-1m-sas.pth'
CHECKPOINT = OUT / 'checkpoints/SASRec-ml-1m-sas.pth'
DATA = ROOT / 'external/ProRL/datasets'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_hashes():
    files = set()
    for directory in ('external', 'dataset', 'repro'):
        for path in (ROOT / directory).rglob('*'):
            if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:
                continue
            if OUT in path.parents or ROOT / 'repro/evaluators' in path.parents:
                continue
            if path.name == 'SASREC_EVALUATOR_VALIDATION.md':
                continue
            files.add(path)
    files.update(ROOT / name for name in read_json(ROOT / 'repro/source_hashes.json')['sha256'])
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def prepare():
    """No model imports. Preserve existing files; download original binary once."""
    guard = OUT / 'protected_sha256_before.json'
    if not guard.exists():
        save_new(guard, protected_hashes())
        save_new(OUT / 'packages_before.json', {
            d.metadata['Name']: d.version for d in importlib.metadata.distributions()})
    if CHECKPOINT.exists():
        meta = read_json(OUT / 'checkpoint_download.json')
        if sha256(CHECKPOINT) != meta['sha256']:
            raise RuntimeError('Checkpoint differs from downloaded official bytes')
        return
    import httpx
    response = httpx.get(URL, follow_redirects=True, timeout=90)
    response.raise_for_status()
    if not response.content.startswith(b'PK'):
        raise RuntimeError('Expected a PyTorch ZIP checkpoint, not HTML/LFS pointer')
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open('xb') as stream:
        stream.write(response.content)
    save_new(OUT / 'checkpoint_download.json', {
        'url': URL, 'commit': COMMIT, 'bytes': len(response.content),
        'sha256': sha256(CHECKPOINT), 'external_modified': False})


def load_checkpoint():
    import torch
    from recbole.config import Config
    from recbole.data.utils import create_dataset
    from recbole.model.sequential_recommender.sasrec import SASRec

    for suffix in ('inter', 'user', 'item'):
        required = DATA / f'ml-1m-sas/ml-1m-sas.{suffix}'
        if not required.is_file():
            raise FileNotFoundError(f'Missing published evaluator dataset: {required}')
    download = read_json(OUT / 'checkpoint_download.json')
    if download['url'] != URL or sha256(CHECKPOINT) != download['sha256']:
        raise RuntimeError('Unverified checkpoint provenance')
    # The user authorized this official checkpoint. It includes a RecBole Config
    # pickle; weights_only=False must never be used here on arbitrary user files.
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    saved = checkpoint['config'].final_config_dict
    source_config = ROOT / 'external/ProRL/config/ml-1m-sas_sasrec_config.yaml'
    config = Config(model=SASRec, dataset='ml-1m-sas',
                    config_file_list=[str(source_config), str(Path(__file__).parent / 'config/validation.yaml')],
                    config_dict={'data_path': str(DATA), 'use_gpu': False,
                                 'checkpoint_dir': str(OUT / 'unused_no_training')})
    architecture_keys = ('n_layers', 'n_heads', 'hidden_size', 'inner_size',
                         'hidden_dropout_prob', 'attn_dropout_prob', 'hidden_act',
                         'layer_norm_eps', 'loss_type', 'MAX_ITEM_LIST_LENGTH')
    comparison = {k: {'checkpoint': saved[k], 'effective': config[k]} for k in architecture_keys}
    mismatch = {k: v for k, v in comparison.items() if v['checkpoint'] != v['effective']}
    if mismatch:
        raise RuntimeError(f'Checkpoint/config architecture mismatch: {mismatch}')
    if config['MAX_ITEM_LIST_LENGTH'] < 20:
        raise RuntimeError('Checkpoint sequence length does not support 20 history items')
    # Dataset construction creates exactly the released evaluator vocabulary.
    # No data_preparation/build/split/training is invoked: token maps precede split.
    dataset = create_dataset(config)
    model = SASRec(config, dataset).cpu()
    state = checkpoint['state_dict']
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = {k: {'expected': list(expected[k].shape), 'checkpoint': list(state[k].shape)}
              for k in set(expected) & set(state) if expected[k].shape != state[k].shape}
    preflight = {'missing_keys': missing, 'unexpected_keys': unexpected, 'shape_mismatches': shapes}
    if missing or unexpected or shapes:
        raise RuntimeError(f'Strict state_dict preflight failed: {preflight}')
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {
        'checkpoint_path': str(CHECKPOINT), 'checkpoint_sha256': sha256(CHECKPOINT),
        'source_url': URL, 'strict_load_success': True, **preflight,
        'checkpoint_keys': sorted(checkpoint), 'state_dict_keys': sorted(state),
        'item_embedding_shape': list(state['item_embedding.weight'].shape),
        'position_embedding_shape': list(state['position_embedding.weight'].shape),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'epoch': checkpoint.get('epoch'), 'architecture_comparison': comparison,
        'checkpoint_dataset_name': saved['dataset'], 'checkpoint_data_path': saved['data_path'],
        'effective_dataset': config['dataset'], 'effective_data_path': config['data_path'],
        'dataset_item_num': dataset.item_num, 'dataset_user_num': dataset.user_num,
        'dataset_interaction_count': len(dataset.inter_feat),
        'mapping_source': 'released evaluator dataset field2token_id, before split',
        'training_time_mapping_independently_verified': False,
        'mapping_limitation': 'Checkpoint has no independently verified training token map. '
            'Reconstructing the released evaluator map and matching tensor shapes does not '
            'prove historical training row semantics; checkpoint names ml-1m, evaluator ml-1m-sas.',
        'data_sha256': {p.name: sha256(p) for p in sorted((DATA / 'ml-1m-sas').glob('*')) if p.is_file()},
        'training_performed': False, 'split_performed': False, 'device': 'cpu',
    }
    return model, dataset, config, metadata


def environment():
    import torch
    return {'python_version': platform.python_version(), 'python_executable': sys.executable,
            'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(),
            'recbole_version': importlib.metadata.version('recbole'), 'inference_device': 'cpu',
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}


if __name__ == '__main__':
    prepare()
    print('Official checkpoint acquired; existing files protected.')

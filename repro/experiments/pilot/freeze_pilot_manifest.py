"""Freeze 10 deterministic MovieLens samples; no model/evaluator/generation calls.

Preserves movielensUserProfile.ipynb:C17 positive filtering and pandas timestamp
sort behavior. Full eligibility pool (1..6040) replaces the notebook export's
User-150 early break, as explicitly required for this pilot. Reuses MI-Bridge v1.
"""
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path

from repro.prepare_ml1m import load_tables
from repro.utils import literal, verify_sources
from repro.methods.mi_bridge.select_bridge import (
    BridgeRanking, generate_bridge_pairs, rank_bridge_pairs, select_bridge)
from repro.methods.mi_bridge.retrieve_bridge_movies import load_movies, retrieve_bridge_movies

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'dataset/ml-1m'
MANIFEST = ROOT / 'repro/results/pilot/pilot_manifest.json'
DOC = ROOT / 'repro/docs/PILOT_SAMPLE_FREEZE.md'
SCRIPT_DIR = Path(__file__).resolve().parent
PILOT_SEED = 20260905
TOP_K = 5
LAMBDA = 1.0
USER_COUNT = 10


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def payload_sha256(manifest):
    """Hash every field except the self-referential digest field itself."""
    return hashlib.sha256(canonical_bytes({k: v for k, v in manifest.items() if k != 'manifest_sha256'})).hexdigest()


def serialized(manifest):
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def protected_hashes():
    files = {ROOT / name for name in read(ROOT / 'repro/source_hashes.json')['sha256']}
    for folder in ('repro', 'external', 'dataset'):
        files.update(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '.git' not in p.parts and '__pycache__' not in p.parts
                     and SCRIPT_DIR not in p.parents and p not in (MANIFEST, DOC))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def load_inputs():
    verify_sources()
    ratings, _, users = load_tables(DATA)
    movies = load_movies(DATA / 'movies.dat')
    frozen = read(ROOT / 'repro/baseline_sample.json')
    for filename, expected in frozen['data_sha256'].items():
        if sha(DATA / filename) != expected:
            raise RuntimeError('MovieLens bytes differ from frozen source: ' + filename)
    return ratings, users, movies


def eligible_user_ids(ratings):
    counts = ratings.loc[ratings[2] >= 4].groupby(0).size().to_dict()
    return [uid for uid in range(1, 6041) if counts.get(uid, 0) > 20]


def sample_users(eligible):
    if len(eligible) < USER_COUNT:
        raise ValueError('Fewer than ten eligible users; do not replace qualification rules')
    return random.Random(PILOT_SEED).sample(sorted(eligible), USER_COUNT)


def select_target(user_id, positive_ids, catalog):
    valid_ids = set(catalog)
    excluded = set(positive_ids)
    if not valid_ids - excluded:
        raise ValueError('No unobserved legal target; do not replace this user')
    rng = random.Random(PILOT_SEED + user_id)
    draws = 0
    # Exactly the original randint(1,3952) rejection rule. Rejection only tests
    # catalog validity / complete positive history, never bridge feasibility.
    while True:
        target_id = rng.randint(1, 3952)
        draws += 1
        if target_id in valid_ids and target_id not in excluded:
            return target_id, draws


def interest_profile(history):
    counts = Counter(genre for movie in history for genre in movie['genres'])
    stats = [{'genre': genre, 'count': count, 'frequency': count / len(history)}
             for genre, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return {'genre_counts': {x['genre']: x['count'] for x in stats},
            'genre_frequencies': {x['genre']: x['frequency'] for x in stats},
            'genre_statistics': stats, 'top5_interests': stats[:TOP_K]}


def build_bridge(history, target, profile, movies):
    analysis = {'history': history, 'target': target,
                'genre_statistics': profile['genre_statistics'], 'top_interests': profile['top5_interests']}
    pairs = generate_bridge_pairs(analysis)
    # Store complete rank plus direct counts/capped candidates, without selecting
    # a lower-ranked pair just because its direct availability is larger.
    enriched = [dict(pair, **retrieve_bridge_movies(pair, movies, history, target)) for pair in pairs]
    ranked = rank_bridge_pairs(enriched)
    selection = select_bridge(BridgeRanking(ranked, movies, history, target), rejected_bridges=[])
    selected = selection.get('selected_bridge')
    fallback = selection.get('selected_fallback')
    status = {'direct': 'direct', 'multi_hop': 'fallback', 'unavailable': 'unavailable'}[selection['bridge_type']]
    return {'ranked_pairs': ranked,
            'selected_existing_interest': selection.get('existing_interest'),
            'selected_target_interest': selection.get('target_interest'),
            'selected_rank': selected['rank'] if selected else None,
            'bridge_type': status, 'method_bridge_type': selection['bridge_type'],
            'score': selected['bridge_score'] if selected else None,
            'direct_candidate_count': selection.get('direct_candidate_count', 0),
            'direct_candidates': selection.get('direct_candidates', []),
            'fallback_middle_genre': fallback['intermediate_interest'] if fallback else None,
            'fallback_left_candidate_count': fallback['left_candidate_count'] if fallback else 0,
            'fallback_right_candidate_count': fallback['right_candidate_count'] if fallback else 0,
            'fallback_left_candidates': fallback['left_candidates'] if fallback else [],
            'fallback_right_candidates': fallback['right_candidates'] if fallback else [],
            'fallback_score': fallback['fallback_score'] if fallback else None,
            'ranked_fallbacks': selection.get('ranked_fallbacks', []),
            'selection_result': selection}


def public_movie(movie):
    return {'movie_id': movie['id'], 'title': movie['title'], 'genres': list(movie['genres'])}


def frozen_prerequisites():
    path = ROOT / 'repro/results/case_study/mi_bridge_v1/formal_evaluation/protocol.json'
    protocol = read(path)
    if protocol['PROTOCOL_VERSION'] != 'Formal-Evaluation-v1':
        raise RuntimeError('Unexpected frozen formal protocol')
    for name, expected in protocol['FROZEN_IMPLEMENTATION_SHA256'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Frozen formal protocol implementation differs')
    if sha(ROOT / 'repro/results/evaluator_validation/checkpoints/SASRec-ml-1m-sas.pth') != protocol['CHECKPOINT_SHA256']:
        raise RuntimeError('Frozen checkpoint differs')
    return protocol


def build_manifest(inputs=None):
    """Pure deterministic construction: no disk writes or time-dependent fields."""
    frozen_prerequisites()
    ratings, users, movies = inputs if inputs is not None else load_inputs()
    catalog = {m['id']: m for m in movies}
    eligible = eligible_user_ids(ratings)
    chosen = sample_users(eligible)  # Complete sample drawn BEFORE any bridge work.
    gender = literal('movielensUserProfile.ipynb', 17, 'gender_dict')
    ages = literal('movielensUserProfile.ipynb', 17, 'age_dict')
    occupations = literal('movielensUserProfile.ipynb', 17, 'occupation_dict')
    entries = []
    for user_id in chosen:
        # Do not switch to stable sort or aggregate-sort: preserve original
        # pandas sort_values(3) default quicksort, including tied timestamp order.
        rows = ratings[(ratings[2] >= 4) & (ratings[0] == user_id)].sort_values(3)
        positive_ids = [int(i) for i in rows[1].tolist()]
        history = [dict(catalog[mid]) for mid in positive_ids[-20:]]
        target_id, draws = select_target(user_id, positive_ids, catalog)
        target = dict(catalog[target_id])
        user_rows = users[users[0] == user_id]
        if len(user_rows) != 1:
            raise ValueError('Missing or duplicate user demographics; do not replace user')
        demo = user_rows.values.tolist()[0]
        profile = interest_profile(history)
        bridge = build_bridge(history, target, profile, movies)
        entries.append({'user_id': user_id, 'history_length': len(history),
            'history': [public_movie(movie) for movie in history],
            'history_rating_timestamps': [{'movie_id': int(row[1]), 'rating': int(row[2]), 'timestamp': int(row[3])}
                                          for row in rows.tail(20).values.tolist()],
            'positive_interaction_count': len(positive_ids), 'positive_movie_ids': positive_ids,
            'demographics': {'gender': gender[demo[1]], 'age': ages[demo[2]], 'occupation': occupations[demo[3]]},
            'demographic_codes': {'gender': demo[1], 'age': int(demo[2]), 'occupation': int(demo[3])},
            'target': public_movie(target), 'target_seed': PILOT_SEED + user_id,
            'target_draw_count': draws, 'interest_profile': profile, 'bridge': bridge,
            'bridge_status': bridge['bridge_type']})
    input_paths = [DATA / filename for filename in ('ratings.dat', 'movies.dat', 'users.dat')]
    input_paths.extend(ROOT / name for name in ('movielensUserProfile.ipynb', 'repro/prepare_ml1m.py',
        'repro/local_config.json', 'repro/movie_path_parser.py', 'repro/source_hashes.json',
        'repro/results/case_study/mi_bridge_v1/formal_evaluation/protocol.json',
        'repro/results/evaluator_validation/mapping_verification/verification_report.json'))
    input_paths.extend(sorted((ROOT / 'repro/methods/mi_bridge').glob('*.py')))
    manifest = {'manifest_version': 'Pilot-Sample-v1', 'method_version': 'MI-Bridge v1',
        'formal_protocol_version': 'Formal-Evaluation-v1', 'pilot_seed': PILOT_SEED,
        'user_count': USER_COUNT, 'pilot_user_ids': chosen, 'top_k': TOP_K, 'lambda': LAMBDA,
        'eligible_user_count': len(eligible), 'eligible_user_ids': eligible,
        'sampling': {'user_rule': 'random.Random(20260905).sample(sorted(eligible_user_ids),10); preserve draw order',
            'user_scope': 'range(1,6041), no notebook export-only break at user 150',
            'eligibility': 'rating>=4; positive interaction count>20',
            'history_rule': 'per-user positive rows.sort_values(timestamp) with original pandas default, last20',
            'target_rule': 'Random(PILOT_SEED+user_id).randint(1,3952); first catalog ID outside ALL positive interactions',
            'target_rejections': 'only nonexistent IDs or positive-interacted IDs; never bridge/user/interest preference',
            'bridge_rule': 'full BridgeRanking interface; ranked pair direct first, one intermediate genre fallback only if direct empty, then next pair',
            'candidate_exclusions': 'only fixed20 history+target, by original method ID/title checks; not entire positive history',
            'candidate_count': 'all qualifying catalog entries before top20 cap; ascending movie_id',
            'bridge_status_serialization': 'original method multi_hop is manifest fallback; method_bridge_type retains original label',
            'unavailable_policy': 'retain user and target unchanged; no resampling',
            'top_k_rule': 'frequency descending, genre lexicographic ascending; use at most five observed genres',
            'feedback_evaluated': False},
        'hash_contract': {'algorithm': 'SHA-256', 'manifest_sha256_scope': 'entire JSON object EXCLUDING only top-level manifest_sha256',
            'canonicalization': 'UTF-8 json.dumps(ensure_ascii=False,sort_keys=True,separators=(comma,colon),allow_nan=False); no newline',
            'file_format': 'UTF-8 sorted keys, indent2, LF final newline; file byte SHA-256 recorded separately in documentation',
            'volatile_fields': 'no timestamp or absolute environment paths'},
        'environment': {'python': platform.python_version(), 'pandas': importlib.metadata.version('pandas'),
                        'numpy': importlib.metadata.version('numpy')},
        'provenance': {'source_cell': 'movielensUserProfile.ipynb:C17',
            'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in input_paths},
            'generator_sha256': sha(Path(__file__)),
            'selection_happens_before_bridge_feasibility': True},
        'users': entries}
    manifest['manifest_sha256'] = payload_sha256(manifest)
    return manifest


def freeze():
    before = protected_hashes()
    # Existing manifest never triggers resampling or target draws. Verification
    # reads its digest only; --verify additionally reconstructs in memory.
    if MANIFEST.exists():
        existing = read(MANIFEST)
        if payload_sha256(existing) != existing['manifest_sha256']:
            raise RuntimeError('Frozen manifest digest mismatch; never overwrite')
        print('Manifest already frozen; not resampled or modified:', existing['manifest_sha256'])
        return
    if DOC.exists():
        raise FileExistsError('Pilot documentation already exists; refuse partial overwrite')
    inputs = load_inputs()
    manifest = build_manifest(inputs)
    repeated = build_manifest(inputs)
    if serialized(manifest) != serialized(repeated):
        raise RuntimeError('Deterministic reconstruction failed; refuse freeze')
    import unittest
    from repro.experiments.pilot.tests.test_pilot_manifest import PilotManifestTests
    PilotManifestTests.prebuilt = (inputs, manifest)
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PilotManifestTests))
    if not tests.wasSuccessful():
        raise RuntimeError('Pilot qualification/freeze tests failed')
    after = protected_hashes()
    if before != after:
        raise RuntimeError('Existing protected artifact changed')
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open('xb') as stream:
        stream.write(serialized(manifest))
    doc = build_document(manifest, tests.testsRun, before)
    if DOC.exists():
        raise FileExistsError('Pilot documentation already exists; refusing overwrite')
    with DOC.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(doc)
    if protected_hashes() != before:
        raise RuntimeError('Post-write protection failed')
    print(json.dumps({'pilot_user_ids': manifest['pilot_user_ids'], 'manifest_sha256': manifest['manifest_sha256'],
                      'file_sha256': sha(MANIFEST), 'tests_passed': tests.testsRun,
                      'status_counts': dict(Counter(u['bridge_status'] for u in manifest['users']))}, indent=2))


def build_document(manifest, test_count, guard):
    counts = Counter(u['bridge_status'] for u in manifest['users'])
    lines = ['# 10-user Pilot Sample Freeze', '', '## Frozen scope', '',
        'MI-Bridge v1; Formal-Evaluation-v1; TOP_K=5; lambda=1.0; PILOT_SEED=20260905.',
        'No LLM/model inference, path generation, SASRec scoring, feedback or training occurred.',
        'All eligible users were enumerated before one 10-user random sample; no bridge-conditioned sampling.',
        'User IDs in draw order: ' + ', '.join(map(str, manifest['pilot_user_ids'])) + '.',
        f'Eligible pool size: {manifest["eligible_user_count"]}; selected users: 10.', '',
        '## Original preprocessing and deterministic choices', '',
        'Original C17 user scope is 1..6040. Its export-only early break at User 150 is not applied:',
        'this stage explicitly requires all eligible users. Rating>=4, positive count>20, per-user',
        'pandas sort_values(timestamp) default quicksort and last20 are unchanged. Timestamp ties',
        'retain the original pandas behavior, not a newly introduced stable sort; package versions are frozen.',
        'random.Random(20260905).sample(sorted(eligible),10) is called once per construction.',
        'Target RNG is independent per user: Random(20260905+user_id).randint(1,3952), first legal',
        'catalog item not in the COMPLETE positive history. Rejecting missing/positive IDs is the',
        'original eligibility rule, not target cherry-picking; draw counts are saved. No target is',
        'redrawn because of a bridge result. Genre frequencies use only the fixed20 history.',
        'Top interests use count/frequency descending then genre name ascending; no target-conditioned profile.', '',
        '## Bridge selection', '',
        'Uses existing generate_bridge_pairs, rank_bridge_pairs, BridgeRanking and select_bridge unchanged.',
        'Full pair ranking is retained. For each ranked pair, direct retrieval has priority; only zero',
        'direct candidates allow one intermediate genre fallback. Unavailable results retain their user/target.',
        'Fallback ranks by min(left_count,right_count), then sum of counts, then genre lexicographic order.',
        'Candidates exclude the fixed20 history and target; earlier positives outside those20 are not',
        'newly excluded by this method. All counts precede the top20 cap; candidates order by movie ID.',
        'Manifest status fallback corresponds to original API multi_hop; original label/result is also retained.',
        'Candidate records retain method-native id; history and target records use movie_id.', '',
        '| User | Target ID / title | Bridge direction | Status | Direct count | Middle |',
        '|---:|---|---|---|---:|---|']
    for user in manifest['users']:
        b = user['bridge']
        direction = f"{b['selected_existing_interest']} -> {b['selected_target_interest']}" if b['selected_existing_interest'] else 'unavailable'
        lines.append(f"| {user['user_id']} | {user['target']['movie_id']} / {user['target']['title']} | {direction} | {user['bridge_status']} | {b['direct_candidate_count']} | {b['fallback_middle_genre'] or '-'} |")
    lines.extend(['', f"Direct={counts['direct']}; fallback={counts['fallback']}; unavailable={counts['unavailable']}.", '',
        '## Manifest hash and immutability', '',
        'MANIFEST_SHA256 = ' + manifest['manifest_sha256'],
        'FILE_SHA256 = ' + sha(MANIFEST), '',
        'An embedded digest cannot generally equal the hash of a file containing that same digest.',
        'manifest_sha256 therefore hashes the entire canonical JSON object EXCLUDING ONLY its own',
        'top-level manifest_sha256 field: UTF-8, sorted keys, no whitespace, ensure_ascii=False.',
        'FILE_SHA256 separately covers the actual saved pretty-printed UTF-8/LF bytes. Both are SHA-256.',
        'No volatile timestamps are included. Same seed, code, dataset and versions produce identical bytes.',
        'Freeze uses exclusive file creation. A normal rerun checks the saved digest and returns without',
        'sampling again. --verify reconstructs only in memory for an integrity check; it never overwrites',
        'or chooses replacement users/targets. Reconstructing for determinism tests is not an additional cohort.', '',
        '```powershell',
        '.\\.venv-repro\\Scripts\\python.exe -B -X utf8 -m repro.experiments.pilot.freeze_pilot_manifest --verify',
        '.\\.venv-repro\\Scripts\\python.exe -B -X utf8 -m unittest discover -s repro/experiments/pilot/tests -v',
        '```', '', '## Validation and preservation', '',
        f'{test_count} tests passed, including synthetic direct/fallback/unavailable control cases (not additional pilot users).',
        'Two independent in-memory constructions produced identical serialized bytes before freezing.',
        'No invalid/unavailable user is replaced. Full positive IDs, history timestamps, demographics,',
        'all ranked pairs, candidate counts, selected fallback routes and input hashes are retained.',
        'ORIGINAL_14_FILES_UNCHANGED = True', 'BASELINE_RESULTS_UNCHANGED = True',
        'MI_BRIDGE_RESULTS_UNCHANGED = True', 'FORMAL_EVALUATION_PROTOCOL_UNCHANGED = True',
        'SASREC_CHECKPOINT_UNCHANGED = True',
        'Existing MI-Bridge method, prompt, parser, Qwen config, mapping verification and external data are unchanged.',
        f'Compared {len(guard)} existing protected files before/after (SHA-256 manifest below).', '',
        '```json', json.dumps(guard, ensure_ascii=False, sort_keys=True, indent=2), '```', '',
        '## Next stage', '',
        'Manifest frozen. Wait for explicit authorization before any 40-path generation.',
        'No performance or feasibility-based sampling conclusion follows from this ten-user sample.', ''])
    return '\n'.join(lines)


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify']:
        existing = read(MANIFEST)
        assert payload_sha256(existing) == existing['manifest_sha256']
        assert serialized(build_manifest()) == MANIFEST.read_bytes()
        print('Frozen manifest exactly reconstructed; no file modified:', existing['manifest_sha256'])
    elif sys.argv[1:]:
        raise SystemExit('Usage: python -m repro.experiments.pilot.freeze_pilot_manifest [--verify]')
    else:
        freeze()

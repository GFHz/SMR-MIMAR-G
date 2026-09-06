"""Generate exactly 40 frozen pilot slots in fixed order; no evaluation.

Each slot is an independent source-behavior dialogue: plan, then the unchanged
formatting turn. A parse failure may retry that same slot at most twice; it never
creates/cherry-picks a replacement slot. Transport retries remain local_llm's
unchanged max_retries=2 behavior. No title repair or path mutation occurs.
"""
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from repro import local_llm
from repro.movie_path_parser import parse_movie_path, truncation_diagnostic
from repro.methods.mi_bridge.bridge_prompt import build_bridge_prompt
from repro.utils import ROOT, literal, source_string, verify_sources, read_json, write_json, sha256
from repro.experiments.pilot.freeze_pilot_manifest import payload_sha256

OUT = ROOT / 'repro/results/pilot/generation_v1'
PILOT = ROOT / 'repro/results/pilot/pilot_manifest.json'
EXPECTED_MANIFEST_SHA = '0fd235067f2fde479eaf6ab102daae225684dbc9b4f1ec9d723ea1795e49301d'
EXPECTED_USERS = [5723, 2677, 829, 5651, 5021, 3113, 4671, 1032, 2249, 419]
EXPECTED_MODEL = 'qwen3:4b-q4_K_M'
EXPECTED_DIGEST = '2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0'
EXPECTED_CONFIG_SHA = '2921e9161fafe272e716e75d66a94f3916dd95070fc912161cea7a23c405d378'
EXPECTED_PARSER_SHA = '157f5fe9156a646bf21834783a747f07ba86f2c98d8fca22f5f20e9e6f215401'
PLAN_COUNT = 40
FORMAT_PROMPT = source_string('chatGPT.ipynb', 3, 'Output the influence path in the format of python list object.')
SYSTEM_PROMPT = literal('chatGPT.ipynb', 3, 'cot_prompt')


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def protected_hashes():
    paths = {ROOT / name for name in read_json(ROOT / 'repro/source_hashes.json')['sha256']}
    for folder in ('repro', 'external', 'dataset'):
        paths.update(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '.git' not in p.parts and '__pycache__' not in p.parts
                     and OUT not in p.parents
                     and p != Path(__file__).resolve()
                     and ROOT / 'repro/experiments/pilot/tests' not in p.parents)
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def user_prompt(user):
    profile = user['demographics']
    text = (f"Gender:{profile['gender']}\nAge:{profile['age']}\n"
            f"Occupation:{profile['occupation']}\n\nHistorical data:\n")
    text += ''.join(f"{movie['title']} Genre:{'|'.join(movie['genres'])}\n" for movie in user['history'])
    target = user['target']
    return text + f"\nTarget movie: {target['title']} Genre:{'|'.join(target['genres'])}\n"


def selected_bridge(user):
    bridge = user['bridge']
    selected = bridge['selection_result']['selected_bridge']
    if selected is None:
        raise RuntimeError('Frozen bridge is unavailable; existing bridge_prompt cannot be silently changed')
    return selected


def prompt_bundle(user, method):
    baseline_user = user_prompt(user)
    baseline = {'system_prompt': SYSTEM_PROMPT, 'user_prompt': baseline_user,
                'formatting_user_prompt': FORMAT_PROMPT}
    if method == 'baseline':
        result = dict(baseline, bridge_context=None, selected_bridge=None,
                      bridge_type=None, bridge_candidates=None)
    elif method == 'mi_bridge':
        bridge = user['bridge']
        if bridge['bridge_type'] != 'direct' or bridge['method_bridge_type'] != 'direct':
            raise RuntimeError('Frozen bridge is not direct; do not alter direct-only frozen prompt implementation')
        augmented = build_bridge_prompt(SYSTEM_PROMPT, baseline_user, selected_bridge(user),
                                        'direct', bridge['direct_candidates'])
        if augmented['system_prompt'] != SYSTEM_PROMPT:
            raise RuntimeError('MI-Bridge changed baseline system prompt')
        if augmented['user_prompt'] != baseline_user + augmented['bridge_context']:
            raise RuntimeError('MI-Bridge prompt difference is not one appended context')
        if augmented['user_prompt'].count('[BRIDGE CONTEXT]') != 1:
            raise RuntimeError('Expected exactly one bridge context')
        result = {**baseline, **augmented, 'selected_bridge': selected_bridge(user),
                  'bridge_type': bridge['bridge_type'], 'bridge_candidates': bridge['direct_candidates']}
    else:
        raise ValueError('Unknown method')
    result['prompt_hash'] = canonical_hash({k: result[k] for k in
        ('system_prompt', 'user_prompt', 'formatting_user_prompt')})
    return result


def planned_slots(manifest):
    users = {user['user_id']: user for user in manifest['users']}
    return [(position, users[user_id], method, path_index)
            for position, user_id in enumerate(EXPECTED_USERS, 1)
            for method in ('baseline', 'mi_bridge') for path_index in (1, 2)]


def preflight(check_service=False):
    verify_sources()
    manifest = read_json(PILOT)
    if manifest.get('manifest_sha256') != EXPECTED_MANIFEST_SHA or payload_sha256(manifest) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError('Frozen pilot manifest hash mismatch; stop before inference')
    if manifest['pilot_user_ids'] != EXPECTED_USERS or len(manifest['users']) != 10:
        raise RuntimeError('Frozen pilot user order/count mismatch')
    cfg = read_json(local_llm.CONFIG)
    baseline_meta = read_json(ROOT / 'repro/results/minimal_baseline_v2/run_metadata.json')
    ours_meta = read_json(ROOT / 'repro/results/case_study/mi_bridge_v1/run_metadata.json')
    required = {'model': EXPECTED_MODEL, 'model_digest': EXPECTED_DIGEST,
                'thinking_mode': False, 'max_output_tokens': 2048,
                'temperature': 0.1, 'context_length': 4096, 'seed': 42,
                'max_retries': 2, 'request_timeout_seconds': 180}
    if any(cfg.get(k) != v for k, v in required.items()):
        raise RuntimeError('Frozen local configuration mismatch')
    if sha256(local_llm.CONFIG) != EXPECTED_CONFIG_SHA or cfg != baseline_meta['configuration'] or cfg != ours_meta['configuration']:
        raise RuntimeError('Configuration differs from frozen baseline v2/User 1 case')
    if sha256(ROOT / 'repro/movie_path_parser.py') != EXPECTED_PARSER_SHA or baseline_meta['parser_sha256'] != EXPECTED_PARSER_SHA:
        raise RuntimeError('Frozen parser mismatch')
    if SYSTEM_PROMPT != read_json(ROOT / 'repro/results/minimal_baseline_v2/prompt.json')['system_prompt']:
        raise RuntimeError('Baseline system prompt source changed')
    if FORMAT_PROMPT != read_json(ROOT / 'repro/results/minimal_baseline_v2/prompt.json')['formatting_user_prompt']:
        raise RuntimeError('Formatting prompt source changed')
    # Reproduce frozen User 1 user-prompt construction even though User 1 is not
    # selected; this is a source-behavior regression check, not a generation.
    sample = read_json(ROOT / 'repro/baseline_sample.json')
    synthetic = {'demographics': sample['profile'],
        'history': [{'movie_id': x['id'], 'title': x['title'], 'genres': x['genres']} for x in sample['history']],
        'target': {'movie_id': sample['target']['id'], 'title': sample['target']['title'], 'genres': sample['target']['genres']}}
    if user_prompt(synthetic) != sample['original_prompt']:
        raise RuntimeError('Pilot baseline user prompt does not reproduce frozen source behavior')
    slots = planned_slots(manifest)
    if len(slots) != PLAN_COUNT:
        raise RuntimeError('Planned slot count is not exactly 40')
    for _, user, method, _ in slots:
        bundle = prompt_bundle(user, method)
        if method == 'baseline' and ('[BRIDGE CONTEXT]' in bundle['user_prompt'] or bundle['selected_bridge'] is not None):
            raise RuntimeError('Baseline prompt contaminated by bridge data')
        if method == 'mi_bridge':
            if bundle['selected_bridge'] != user['bridge']['selection_result']['selected_bridge']:
                raise RuntimeError('Selected bridge differs from manifest')
            if bundle['bridge_candidates'] != user['bridge']['direct_candidates']:
                raise RuntimeError('Bridge candidates differ from manifest')
    installed = None
    ollama_version = None
    if check_service:
        _, version = local_llm.request('/api/version', timeout=15)
        _, tags = local_llm.request('/api/tags', timeout=15)
        installed = next((x for x in tags['models'] if x['name'] == EXPECTED_MODEL), None)
        if installed is None or installed['digest'] != EXPECTED_DIGEST:
            raise RuntimeError('Installed model digest mismatch; no inference attempted')
        ollama_version = version['version']
    return manifest, cfg, slots, installed, ollama_version


def execute_dialogue(bundle):
    plan = local_llm.generate(bundle['system_prompt'], bundle['user_prompt'])
    plan['truncation'] = truncation_diagnostic(plan['response'], 2048)
    messages = [{'role': 'system', 'content': bundle['system_prompt']},
                {'role': 'user', 'content': bundle['user_prompt']},
                {'role': 'assistant', 'content': plan['content']},
                {'role': 'user', 'content': bundle['formatting_user_prompt']}]
    formatted = local_llm.generate_messages(messages)
    formatted['truncation'] = truncation_diagnostic(formatted['response'], 2048)
    parsed = parse_movie_path(formatted['content'])
    return plan, formatted, parsed


def result_file(user_id, method, path_index):
    return OUT / 'users' / str(user_id) / f'{method}_path_{path_index}.json'


def save_adapter_attempt_factory(raw_dir, dialogue_index, role, counter):
    def save(record):
        counter[0] += 1
        path = raw_dir / f'dialogue_{dialogue_index}_{role}_transport_{counter[0]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError('Adapter raw-attempt file already exists')
        write_json(path, record)
        return str(path)
    return save


def record_for(user, method, path_index, bundle):
    started = now()
    started_clock = time.perf_counter()
    dialogues = []
    final = None
    raw_dir = OUT / 'raw' / str(user['user_id']) / f'{method}_path_{path_index}'
    transport_counter = [0]
    for dialogue_index in range(1, 4):  # first request plus at most two parse retries
        plan_counter = [0]
        old_save = local_llm.save_record
        try:
            local_llm.save_record = save_adapter_attempt_factory(raw_dir, dialogue_index, 'plan', plan_counter)
            plan = local_llm.generate(bundle['system_prompt'], bundle['user_prompt'])
            plan['truncation'] = truncation_diagnostic(plan['response'], 2048)
            plan_attempts = [read_json(path) for path in plan['attempt_records']]
            transport_counter[0] += len(plan_attempts)
            format_counter = [0]
            local_llm.save_record = save_adapter_attempt_factory(raw_dir, dialogue_index, 'format', format_counter)
            messages = [{'role': 'system', 'content': bundle['system_prompt']},
                        {'role': 'user', 'content': bundle['user_prompt']},
                        {'role': 'assistant', 'content': plan['content']},
                        {'role': 'user', 'content': bundle['formatting_user_prompt']}]
            formatted = local_llm.generate_messages(messages)
            formatted['truncation'] = truncation_diagnostic(formatted['response'], 2048)
            format_attempts = [read_json(path) for path in formatted['attempt_records']]
            transport_counter[0] += len(format_attempts)
        finally:
            local_llm.save_record = old_save
        parsed = parse_movie_path(formatted['content'])
        dialogue = {'dialogue_attempt': dialogue_index,
            'plan': {'content': plan['content'], 'thinking': plan['thinking'],
                     'raw_api_response': plan['raw_response'], 'response_metadata': plan['response'],
                     'elapsed_seconds': plan['elapsed_seconds'], 'transport_attempts': plan_attempts,
                     'truncation': plan['truncation']},
            'formatting': {'content': formatted['content'], 'thinking': formatted['thinking'],
                     'raw_api_response': formatted['raw_response'], 'response_metadata': formatted['response'],
                     'elapsed_seconds': formatted['elapsed_seconds'], 'transport_attempts': format_attempts,
                     'truncation': formatted['truncation']}, 'parser_result': parsed}
        dialogues.append(dialogue)
        final = (formatted, parsed)
        if parsed['parse_success']:
            break
    formatted, parsed = final
    generated_tokens = sum((d['plan']['response_metadata'].get('eval_count') or 0) +
                           (d['formatting']['response_metadata'].get('eval_count') or 0) for d in dialogues)
    record = {'record_version': 'Pilot-Generation-v1', 'user_id': user['user_id'],
        'target_id': user['target']['movie_id'], 'target_title': user['target']['title'],
        'method': method, 'path_index': path_index,
        'selected_bridge': bundle['selected_bridge'], 'bridge_type': bundle['bridge_type'],
        'bridge_candidates': bundle['bridge_candidates'], 'prompt_hash': bundle['prompt_hash'],
        'prompt_inputs': {'system_prompt': bundle['system_prompt'], 'user_prompt': bundle['user_prompt'],
                          'formatting_user_prompt': bundle['formatting_user_prompt']},
        'raw_response': formatted['content'], 'raw_api_response': formatted['raw_response'],
        'parsed_path': parsed['path'], 'parse_success': parsed['parse_success'],
        'parser_strategy': parsed['parse_strategy_used'], 'parser_diagnostics': parsed,
        'dialogue_attempts': dialogues, 'dialogue_attempt_count': len(dialogues),
        'parse_retry_count': len(dialogues) - 1, 'adapter_attempt_count': transport_counter[0],
        'generated_tokens': generated_tokens,
        'inference_seconds': time.perf_counter() - started_clock,
        'started_at': started, 'finished_at': now(),
        'model_name': EXPECTED_MODEL, 'model_digest': EXPECTED_DIGEST,
        'thinking_mode': False, 'max_output_tokens': 2048,
        'content_modified_or_repaired': False, 'title_resolution_performed': False}
    return record


def summary(records):
    failed = [{'user_id': x['user_id'], 'method': x['method'], 'path_index': x['path_index'],
               'error': x['parser_diagnostics']['error']} for x in records if not x['parse_success']]
    users = []
    for user_id in EXPECTED_USERS:
        group = [x for x in records if x['user_id'] == user_id]
        users.append({'user_id': user_id, 'planned_path_count': 4,
            'saved_path_count': len(group), 'baseline_path_count': sum(x['method'] == 'baseline' for x in group),
            'mi_bridge_path_count': sum(x['method'] == 'mi_bridge' for x in group),
            'parse_success_count': sum(x['parse_success'] for x in group),
            'parse_failure_count': sum(not x['parse_success'] for x in group),
            'adapter_requests': sum(x['adapter_attempt_count'] for x in group),
            'generated_tokens': sum(x['generated_tokens'] for x in group),
            'inference_seconds': sum(x['inference_seconds'] for x in group)})
    return {'expected_path_count': PLAN_COUNT, 'actual_requested_path_count': len(records),
        'baseline_path_count': sum(x['method'] == 'baseline' for x in records),
        'mi_bridge_path_count': sum(x['method'] == 'mi_bridge' for x in records),
        'parse_success_count': sum(x['parse_success'] for x in records),
        'parse_failure_count': len(failed), 'failed_paths': failed,
        'total_adapter_requests': sum(x['adapter_attempt_count'] for x in records),
        'total_generated_tokens': sum(x['generated_tokens'] for x in records),
        'total_inference_seconds': sum(x['inference_seconds'] for x in records),
        'per_user_summary': users}


def run():
    if OUT.exists():
        raise FileExistsError('generation_v1 already exists; refusing overwrite, resume, or extra generations')
    manifest, cfg, slots, installed, ollama_version = preflight(check_service=True)
    guard = protected_hashes()
    OUT.mkdir(parents=True)
    state = {'status': 'running', 'started_at': now(), 'PILOT_MANIFEST_SHA256': EXPECTED_MANIFEST_SHA,
        'METHOD_VERSION': 'MI-Bridge v1', 'BASELINE_VERSION': 'LLM-IPP-CoT-source-behavior baseline v2',
        'MODEL_NAME': EXPECTED_MODEL, 'MODEL_DIGEST': EXPECTED_DIGEST,
        'THINK_MODE': False, 'MAX_OUTPUT_TOKENS': 2048,
        'PARSER': 'repro/movie_path_parser.py:parse_movie_path', 'PARSER_SHA256': EXPECTED_PARSER_SHA,
        'LOCAL_ADAPTER': 'repro/local_llm.py', 'LOCAL_ADAPTER_SHA256': sha256(ROOT/'repro/local_llm.py'),
        'CONFIGURATION': cfg, 'CONFIGURATION_SHA256': EXPECTED_CONFIG_SHA,
        'OLLAMA_VERSION': ollama_version, 'INSTALLED_MODEL_METADATA': installed,
        'EXPECTED_PATH_COUNT': PLAN_COUNT, 'REQUEST_COUNT': 0, 'PARSE_SUCCESS_COUNT': 0,
        'PARSE_FAILURE_COUNT': 0, 'generation_order': [], 'protected_sha256_before': guard,
        'experiment_source_sha256': sha256(Path(__file__)), 'python_version': platform.python_version(),
        'definitions': {'REQUEST_COUNT': 'planned path slots executed, exactly 40; not HTTP calls',
            'TOTAL_ADAPTER_REQUESTS': 'actual /api/chat transport attempts across planning and formatting, including retries',
            'generated_tokens': 'sum Ollama eval_count for all planning/formatting responses including parse retries',
            'parse_retry': 'retry whole same dialogue slot only after final parser failure, max2; never replacement/cherry-pick',
            'transport_retry': 'unchanged local adapter policy, max2 after first attempt for transient errors only',
            'independence': 'fresh messages per planned slot; previous output never supplied to next slot'}}
    write_json(OUT/'generation_manifest.json', state)
    records = []
    try:
        for user_position, user, method, path_index in slots:
            if protected_hashes() != guard:
                raise RuntimeError('Frozen/protected input changed during generation')
            bundle = prompt_bundle(user, method)
            record = record_for(user, method, path_index, bundle)
            destination = result_file(user['user_id'], method, path_index)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError('Result slot already exists')
            write_json(destination, record)
            records.append(record)
            state['REQUEST_COUNT'] = len(records)
            state['PARSE_SUCCESS_COUNT'] = sum(x['parse_success'] for x in records)
            state['PARSE_FAILURE_COUNT'] = sum(not x['parse_success'] for x in records)
            state['generation_order'].append({'ordinal': len(records), 'user_id': user['user_id'],
                'method': method, 'path_index': path_index, 'parse_success': record['parse_success'],
                'dialogue_attempt_count': record['dialogue_attempt_count'],
                'adapter_attempt_count': record['adapter_attempt_count']})
            write_json(OUT/'generation_manifest.json', state)
            print(f'[user {user_position}/10][{method}][path {path_index}/2] '
                  f'{"success" if record["parse_success"] else "fail"}', flush=True)
        if len(records) != PLAN_COUNT:
            raise RuntimeError('Did not execute exactly forty planned slots')
        state.update(status='complete', finished_at=now(), **{
            'TOTAL_ADAPTER_REQUESTS': sum(x['adapter_attempt_count'] for x in records),
            'TOTAL_GENERATED_TOKENS': sum(x['generated_tokens'] for x in records),
            'TOTAL_INFERENCE_SECONDS': sum(x['inference_seconds'] for x in records)})
    except Exception as exc:
        state.update(status='failed', finished_at=now(), error_type=type(exc).__name__, error=str(exc))
        write_json(OUT/'generation_manifest.json', state)
        raise
    if protected_hashes() != guard:
        state.update(status='failed_protection_check', error='Protected input changed')
        write_json(OUT/'generation_manifest.json', state)
        raise RuntimeError('Protected input changed at completion')
    computed = summary(records)
    write_json(OUT/'summary.json', computed)
    state.update(protected_sha256_after=protected_hashes(), protection={
        'ORIGINAL_14_FILES_UNCHANGED': True, 'BASELINE_RESULTS_UNCHANGED': True,
        'MI_BRIDGE_METHOD_UNCHANGED': True, 'FORMAL_EVALUATION_PROTOCOL_UNCHANGED': True,
        'PILOT_MANIFEST_UNCHANGED': True, 'SASREC_CHECKPOINT_UNCHANGED': True},
        GENERATION_COMPLETE=True, summary_sha256=sha256(OUT/'summary.json'))
    write_json(OUT/'generation_manifest.json', state)
    print(json.dumps(computed, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        manifest, cfg, slots, installed, version = preflight(check_service=True)
        print(json.dumps({'preflight': 'passed', 'planned_paths': len(slots),
                          'model_digest': installed['digest'], 'ollama_version': version,
                          'manifest_sha256': manifest['manifest_sha256']}, indent=2))
    elif sys.argv[1:]:
        raise SystemExit('Usage: python -m repro.experiments.pilot.generate_pilot_paths [--preflight]')
    else:
        run()

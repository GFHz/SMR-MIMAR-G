"""Local-only Ollama adapter. No OpenAI SDK, credentials, or notebook execution."""
import ast
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

MODEL = 'qwen3:4b-q4_K_M'
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'local_config.json'
BASE_URL = 'http://127.0.0.1:11434'


def request(endpoint, payload=None, timeout=180):
    # Explicitly bypass system proxies for loopback traffic; never forward keys.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE_URL + endpoint, data=data,
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=timeout) as response:
        raw = response.read().decode('utf-8')
    return raw, json.loads(raw)


def save_record(record):
    folder = ROOT / 'artifacts'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ('local_raw_' + uuid.uuid4().hex + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return str(path)


def parse_list(content):
    """Strict diagnostic parser: no stripping fences or editing the response."""
    value = ast.literal_eval(content)
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError('Final content is not a Python list of strings.')
    return value


def generate(system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': user_prompt})
    return generate_messages(messages)


def generate_messages(messages):
    """Return full raw response plus separate content/thinking and record path.

    Fail closed on changed model digest. At most 2 retries after the first
    attempt, only for transient transport/5xx errors. Each attempt is recorded.
    """
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    if cfg['model'] != MODEL or not cfg.get('model_digest'):
        raise RuntimeError('Expected fixed model and recorded digest in local_config.json.')
    _, tags = request('/api/tags', timeout=15)
    entry = next((m for m in tags['models'] if m['name'] == MODEL), None)
    if not entry or entry['digest'] != cfg['model_digest']:
        raise RuntimeError('Local model missing or digest mismatch; no inference attempted.')
    if not messages or any(m.get('role') not in ('system', 'user', 'assistant')
                           or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Expected an explicit system/user/assistant message history.')
    payload = {
        'model': MODEL, 'messages': messages, 'stream': False,
        'think': cfg['thinking_mode'], 'keep_alive': '5m',
        'options': {'temperature': cfg['temperature'], 'num_ctx': cfg['context_length'],
                    'seed': cfg['seed'], 'num_predict': cfg['max_output_tokens']},
    }
    retries = cfg['max_retries']
    if not isinstance(retries, int) or not 0 <= retries <= 2:
        raise ValueError('max_retries must be between 0 and 2.')
    attempt_records = []
    for attempt in range(retries + 1):
        start = time.perf_counter()
        record = {'request': payload, 'config': cfg, 'attempt': attempt + 1}
        try:
            raw, response = request('/api/chat', payload, timeout=cfg['request_timeout_seconds'])
            record.update(raw_response=raw, response=response,
                          elapsed_seconds=time.perf_counter() - start)
            path = save_record(record)
            attempt_records.append(path)
            if response.get('error') or not response.get('done'):
                raise RuntimeError(f'Ollama returned an incomplete/error response; see {path}')
            message = response.get('message', {})
            return {'content': message.get('content', ''), 'thinking': message.get('thinking', ''),
                    'raw_response': raw, 'response': response, 'record_path': path,
                    'elapsed_seconds': record['elapsed_seconds'], 'attempt_records': attempt_records}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            record.update(error_type=type(exc).__name__, error=str(exc),
                          elapsed_seconds=time.perf_counter() - start)
            path = save_record(record)
            attempt_records.append(path)
            if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                raise RuntimeError(f'Ollama HTTP {exc.code}; see {path}') from exc
            if attempt == retries:
                raise RuntimeError(f'Ollama failed after {retries + 1} attempts; see {path}') from exc
            time.sleep(attempt + 1)

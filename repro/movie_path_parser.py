"""Extract an existing nonempty string list. Never repair or filter its items."""
import ast
import json
import re


def _decode(candidate):
    for name, decoder in [('json', json.loads), ('python_literal', ast.literal_eval)]:
        try:
            value = decoder(candidate)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            continue
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return value, name
    return None, None


def _balanced_spans(text):
    # Start only at outer brackets. Quotes and escapes inside strings protect
    # literal brackets; nested containers cannot turn into separate paths.
    start = None
    depth = 0
    quote = None
    escaped = False
    for i, char in enumerate(text):
        if start is None:
            if char == '[':
                start, depth = i, 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0:
                yield start, i + 1
                start = None


def parse_movie_path(raw_response):
    """Return selected values and auditable source offsets; latest valid wins."""
    if not isinstance(raw_response, str):
        raise TypeError('raw_response must be a string')
    spans = {}
    left = len(raw_response) - len(raw_response.lstrip())
    right = len(raw_response.rstrip())
    spans[(left, right)] = 'whole_response'
    for match in re.finditer(r'```[^\r\n]*\r?\n(.*?)```', raw_response, re.DOTALL):
        body = match.group(1)
        start = match.start(1) + len(body) - len(body.lstrip())
        end = match.start(1) + len(body.rstrip())
        spans.setdefault((start, end), 'fenced_code_block')
    for span in _balanced_spans(raw_response):
        spans.setdefault(span, 'balanced_brackets')
    candidates = []
    for (start, end), strategy in spans.items():
        value, decoder = _decode(raw_response[start:end])
        if value is not None:
            candidates.append({'start': start, 'end': end, 'path': value,
                               'strategy': strategy + ':' + decoder})
    candidates.sort(key=lambda c: (c['end'], c['start']))
    chosen = candidates[-1] if candidates else None
    return {'path': chosen['path'] if chosen else [], 'parse_success': chosen is not None,
            'parse_strategy_used': chosen['strategy'] if chosen else None,
            'candidate_list_count': len(candidates),
            'candidate_count_definition': 'unique complete legal nonempty string-list spans',
            'selected_span': [chosen['start'], chosen['end']] if chosen else None,
            'selected_raw_text': raw_response[chosen['start']:chosen['end']] if chosen else None,
            'error': None if chosen else 'No complete legal nonempty string list found',
            'semantic_content_changed': False}


def truncation_diagnostic(response, max_output_tokens):
    reason = response.get('done_reason', response.get('finish_reason'))
    count = response.get('eval_count')
    reached = count >= max_output_tokens if isinstance(count, int) else None
    if reason in ('length', 'max_tokens') or reached is True:
        truncated = True
    elif reason == 'stop' and response.get('done') is True:
        truncated = False
    else:
        truncated = 'Unknown'
    return {'done_reason': reason, 'generated_token_count': count,
            'max_output_tokens': max_output_tokens, 'reached_max_output_tokens': reached,
            'raw_response_suspected_truncated': truncated, 'truncated': truncated,
            'basis': 'Ollama termination metadata and token count; no content repair'}

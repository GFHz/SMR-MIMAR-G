"""Artifact IO, byte-level source guards, and literal extraction (no notebook exec)."""
import ast
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'repro' / 'artifacts'
DATA = ROOT / 'dataset' / 'ml-1m'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    key = os.environ.get('OPENAI_API_KEY')
    if key and key in text:
        raise RuntimeError('Refusing to persist content containing the API credential.')
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def verify_sources():
    manifest = read_json(ROOT / 'repro' / 'source_hashes.json')
    bad = [name for name, expected in manifest['sha256'].items()
           if not (ROOT / name).is_file() or sha256(ROOT / name) != expected]
    if bad:
        raise RuntimeError('Upstream source mismatch: ' + ', '.join(bad))
    return manifest


def cell_tree(filename, cell_number):
    verify_sources()
    cell = read_json(ROOT / filename)['cells'][cell_number - 1]
    return ast.parse(''.join(cell['source']))


def literal(filename, cell_number, variable):
    tree = cell_tree(filename, cell_number)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Source literal not found: ' + variable)


def source_string(filename, cell_number, prefix):
    matches = {n.value for n in ast.walk(cell_tree(filename, cell_number))
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value.startswith(prefix)}
    if len(matches) != 1:
        raise ValueError('Source string is missing or ambiguous: ' + prefix)
    return matches.pop()


def parse_source_path(raw):
    try:
        return ast.literal_eval(raw[raw.index('['):raw.index(']') + 1]), None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        return [], type(exc).__name__


def parse_source_score(raw):
    match = re.search(r'0\.\d+', raw)
    return float(match.group()) if match else None


if __name__ == '__main__':
    print('Original source hashes verified:', len(verify_sources()['sha256']))

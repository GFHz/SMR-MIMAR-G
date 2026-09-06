"""Static selection with an optional rejection interface, not a feedback simulation."""
import json
from fractions import Fraction
from pathlib import Path
from .retrieve_bridge_movies import load_movies, retrieve_bridge_movies
from repro.utils import ROOT, read_json, sha256, verify_sources


class BridgeRanking(list):
    """Ordered pairs with explicit retrieval context for the MI-Bridge v1 API.

    Unlike legacy count-only lists, this carries enough data for lazy retrieval.
    Re-wrap deserialized ranked_bridge_pairs with this class before selection.
    """
    def __init__(self, pairs, movies, history, target):
        super().__init__(pairs)
        self.movies = list(movies)
        self.history = list(history)
        self.target = target


def target_need(genre, counts, history_length):
    if history_length <= 0:
        raise ValueError('history_length must be positive')
    return Fraction(1) - Fraction(counts.get(genre, 0), history_length)


def bridge_score(interest, target_genre, counts, history_length):
    # Version 1 fixes lambda to exactly 1.0; no tuning parameter.
    return Fraction(counts.get(interest, 0), history_length) + target_need(target_genre, counts, history_length)


def generate_bridge_pairs(analysis):
    counts = {x['genre']: x['count'] for x in analysis['genre_statistics']}
    length = len(analysis['history'])
    pairs = []
    for interest in sorted({x['genre'] for x in analysis['top_interests']}):
        for genre in sorted(set(analysis['target']['genres'])):
            if interest == genre:
                continue
            pairs.append({'user_existing_interest': interest, 'target_related_interest': genre,
                          'user_frequency': float(Fraction(counts.get(interest, 0), length)),
                          'target_need': float(target_need(genre, counts, length)),
                          'bridge_score': float(bridge_score(interest, genre, counts, length))})
    return pairs


def rank_bridge_pairs(pairs):
    return [dict(pair, rank=rank) for rank, pair in enumerate(sorted(pairs, key=lambda p:
            (-p['bridge_score'], -p['user_frequency'], p['user_existing_interest'], p['target_related_interest'])), 1)]


def bridge_key(pair):
    """Rejections accept returned pair dictionaries or two-string tuple keys."""
    if isinstance(pair, dict):
        return pair['user_existing_interest'], pair['target_related_interest']
    if isinstance(pair, (tuple, list)) and len(pair) == 2 and all(isinstance(x, str) for x in pair):
        return tuple(pair)
    raise ValueError('A bridge must be a pair dictionary or a two-string tuple/list')


def select_bridge(ranked_bridge_pairs, rejected_bridges=None):
    rejected = {bridge_key(pair) for pair in (() if rejected_bridges is None else rejected_bridges)}
    if isinstance(ranked_bridge_pairs, BridgeRanking):
        from .find_fallback_bridge import find_fallback_bridge
        context = ranked_bridge_pairs
        for pair in context:
            if bridge_key(pair) in rejected:
                continue
            i, g = bridge_key(pair)
            if i == g:
                continue
            direct = retrieve_bridge_movies(pair, context.movies, context.history, context.target)
            if direct['candidate_count'] > 0:
                return {'bridge_type': 'direct', 'selected_bridge': dict(pair),
                        'existing_interest': i, 'target_interest': g,
                        'direct_candidate_count': direct['candidate_count'],
                        'direct_candidates': direct['candidates'], 'fallback_supported': True}
            fallback = find_fallback_bridge(i, g, context.movies, context.history, context.target)
            if fallback['selected_fallback'] is not None:
                return {'bridge_type': 'multi_hop', 'selected_bridge': dict(pair),
                        'existing_interest': i, 'target_interest': g,
                        'direct_candidate_count': 0, 'direct_candidates': [],
                        'fallback_supported': True, **fallback}
        return {'bridge_type': 'unavailable', 'selected_bridge': None, 'fallback_supported': True}
    # Compatibility for the original count-only API/tests. No catalog is
    # available here: do not pretend that fallback was searched. New production
    # callers must use BridgeRanking and receive explicit bridge_type statuses.
    for pair in ranked_bridge_pairs:
        count = pair['candidate_count']
        if not isinstance(count, int) or count < 0:
            raise ValueError('candidate_count must be a nonnegative integer')
        if bridge_key(pair) not in rejected and count > 0:
            return pair
    return None  # Legacy return contract only.


# Future Feedback-Aware Bridge Switching: positive -> continue current bridge;
# negative -> add its key to rejected_bridges and call select_bridge again.
# NOT EVALUATED IN CURRENT STAGE. No feedback generator or simulator is present.


def run_case():
    source = verify_sources()
    path = ROOT / 'repro/results/case_study/user_1_interest_analysis.json'
    analysis = read_json(path)
    if analysis['user_id'] != 1 or analysis['target']['title'] != 'This Is My Father (1998)' or len(analysis['history']) != 20 or len(analysis['top_interests']) != 5:
        raise ValueError('Fixed case / Top-K mismatch')
    # Reuse the earlier protected-file manifest and check all prior result files.
    protected = {ROOT / name for name in analysis['provenance']['input_hashes']}
    protected.update(p for folder in ('results', 'docs') for p in (ROOT/'repro'/folder).rglob('*') if p.is_file())
    output = ROOT/'repro/results/case_study/user_1_bridge_selection.json'
    protected.discard(output)
    before = {str(p.relative_to(ROOT)).replace('\\','/'): sha256(p) for p in sorted(protected)}
    for name, expected in analysis['provenance']['input_hashes'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Prior analysis input changed: ' + name)
    movies_path = ROOT/'dataset/ml-1m/movies.dat'
    if sha256(movies_path) != analysis['provenance']['dataset_hashes']['movies.dat']:
        raise ValueError('MovieLens catalog changed')
    movies = load_movies(movies_path)
    pairs = generate_bridge_pairs(analysis)
    enriched = [dict(pair, **retrieve_bridge_movies(pair, movies, analysis['history'], analysis['target'])) for pair in pairs]
    ranked = rank_bridge_pairs(enriched)
    # The one real case execution uses no rejected bridge and no feedback loop.
    selected = select_bridge(ranked, rejected_bridges=[])
    result = {'user_id': 1, 'target': analysis['target'],
              'method': {'name': 'Static Multi-Interest Bridge', 'lambda': 1.0, 'top_k': 5,
                         'score': 'UserFreq(i) + 1.0 * (1 - UserFreq(g))',
                         'candidate_order': 'movie_id ascending', 'candidate_limit': 20,
                         'candidate_count_definition': 'total eligible matches before top-20 storage cap'},
              'all_bridge_pairs': enriched, 'ranked_bridge_pairs': ranked,
              'selected_bridge_pair': selected,
              'bridge_movie_candidates': selected['candidates'] if selected else [],
              'rejected_bridges': [], 'selection_status': 'selected' if selected else 'no_available_bridge',
              'feedback_extension': {'supported_by_interface': True, 'evaluated': False},
              'provenance': {'analysis_sha256': sha256(path), 'movies_sha256': sha256(movies_path),
                             'original_sources': source, 'protected_sha256': before,
                             'method_source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}}}
    if any(sha256(ROOT/name) != expected for name, expected in before.items()):
        raise ValueError('Protected file changed during selection')
    verify_sources()
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if output.exists():
        if read_json(output) != result:
            raise ValueError('Refusing to overwrite differing bridge selection result')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    print(json.dumps({'top5': ranked[:5], 'selected': selected, 'original_14_files_unchanged': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run_case()

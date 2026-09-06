"""MI-Bridge v1: direct first, then at most one real catalog intermediate genre."""
import json
from .retrieve_bridge_movies import load_movies, retrieve_bridge_movies
from repro.utils import ROOT, read_json, sha256, verify_sources


def rank_fallbacks(candidates):
    return sorted(candidates, key=lambda c: (-c['fallback_score'],
                  -(c['left_candidate_count'] + c['right_candidate_count']),
                  c['intermediate_interest']))


def find_fallback_bridge(existing_interest, target_interest, movies, history, target):
    vocabulary = sorted({g for movie in movies for g in movie['genres']})
    i, g = existing_interest, target_interest
    if i == g or i not in vocabulary or g not in vocabulary:
        return {'selected_fallback': None, 'ranked_fallbacks': [], 'reason': 'invalid_endpoint_genres'}
    pair = {'user_existing_interest': i, 'target_related_interest': g}
    # Defensive gate also protects callers bypassing select_bridge.
    direct = retrieve_bridge_movies(pair, movies, history, target)
    if direct['candidate_count']:
        return {'selected_fallback': None, 'ranked_fallbacks': [], 'reason': 'direct_available'}
    candidates = []
    for m in vocabulary:
        if m in (i, g):
            continue
        left = retrieve_bridge_movies({'user_existing_interest': i, 'target_related_interest': m}, movies, history, target)
        right = retrieve_bridge_movies({'user_existing_interest': m, 'target_related_interest': g}, movies, history, target)
        if left['candidate_count'] and right['candidate_count']:
            candidates.append({'existing_interest': i, 'intermediate_interest': m, 'target_interest': g,
                               'left_candidate_count': left['candidate_count'],
                               'right_candidate_count': right['candidate_count'],
                               'left_candidates': left['candidates'], 'right_candidates': right['candidates'],
                               'fallback_score': min(left['candidate_count'], right['candidate_count'])})
    ranked = rank_fallbacks(candidates)
    return {'selected_fallback': ranked[0] if ranked else None, 'ranked_fallbacks': ranked,
            'reason': 'available' if ranked else 'no_one_intermediate_route'}


def find_real_example(movies, history, target, top_interests):
    """Sanity only: deterministic vocabulary pair scan with frozen exclusions.

    Prefer the fixed user's Top-5 x target genres. If none works, scan ordered
    pairs from the real catalog vocabulary (not a fabricated user preference).
    """
    vocabulary = sorted({g for movie in movies for g in movie['genres']})
    primary = [(i, g) for i in sorted(top_interests) for g in sorted(target['genres']) if i != g]
    pairs = primary + [(i, g) for i in vocabulary for g in vocabulary if i != g and (i, g) not in primary]
    for i, g in pairs:
        pair = {'user_existing_interest': i, 'target_related_interest': g}
        if retrieve_bridge_movies(pair, movies, history, target)['candidate_count']:
            continue
        fallback = find_fallback_bridge(i, g, movies, history, target)
        if fallback['selected_fallback'] is not None:
            return {'scope': 'fixed_user_top5_target_genres' if (i, g) in primary else 'catalog_vocabulary_pair_sanity_only',
                    'direct_candidate_count': 0, **fallback}
    return None


def run_sanity():
    from .select_bridge import BridgeRanking, select_bridge
    verify_sources()
    analysis_path = ROOT/'repro/results/case_study/user_1_interest_analysis.json'
    previous_path = ROOT/'repro/results/case_study/user_1_bridge_selection.json'
    analysis, previous = read_json(analysis_path), read_json(previous_path)
    if analysis['user_id'] != 1 or analysis['target']['title'] != 'This Is My Father (1998)':
        raise ValueError('Fixed case mismatch')
    protected = dict(analysis['provenance']['input_hashes'])
    protected.update({str(p.relative_to(ROOT)): sha256(p) for p in (analysis_path, previous_path, ROOT/'repro/movie_path_parser.py')})
    for name, expected in protected.items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Protected hash mismatch: ' + name)
    path = ROOT/'dataset/ml-1m/movies.dat'
    if sha256(path) != analysis['provenance']['dataset_hashes']['movies.dat']:
        raise ValueError('MovieLens catalog hash mismatch')
    movies = load_movies(path)
    history, target = analysis['history'], analysis['target']
    selected = select_bridge(BridgeRanking(previous['ranked_bridge_pairs'], movies, history, target), [])
    assert selected['bridge_type'] == 'direct'
    assert (selected['existing_interest'], selected['target_interest']) == ("Children's", 'Romance')
    assert selected['direct_candidate_count'] == 6
    assert selected['direct_candidates'] == previous['bridge_movie_candidates']
    example = find_real_example(movies, history, target, [x['genre'] for x in analysis['top_interests']])
    result = {'method_version': 'MI-Bridge v1', 'fallback_max_intermediate_genres': 1,
              'user_1_selection': selected, 'example_fallback_case': example,
              'example_exclusion_context': 'User 1 frozen 20 history movies and fixed target; no other user sampled',
              'example_note': 'Vocabulary-only examples are not claims about User 1 Top-5 interests or target genres.',
              'feedback_interface_supported': True, 'feedback_evaluated': False,
              'protected_sha256': protected, 'original_14_files_unchanged': True,
              'baseline_results_unchanged': True, 'user_1_result_unchanged': True,
              'movies_sha256': sha256(path)}
    if any(sha256(ROOT/name) != expected for name, expected in protected.items()):
        raise ValueError('Protected input changed during sanity check')
    verify_sources()
    output = ROOT/'repro/results/case_study/bridge_fallback_sanity.json'
    if output.exists():
        if read_json(output) != result:
            raise ValueError('Refusing to overwrite differing sanity result')
    else:
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'user_1_bridge_type': selected['bridge_type'], 'direct_candidate_count': 6,
                      'example_scope': example['scope'] if example else None,
                      'example': example['selected_fallback'] if example else None}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run_sanity()

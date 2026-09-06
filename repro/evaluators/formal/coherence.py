"""Genre-overlap formula independently implemented from ProRL-style definition.

Reference: hongruhou89/ProRL evaluator.py, snapshot
506e91355377f546dcf51f684fc871fe57a9d5c8. No history-boundary edge or embeddings.
"""


def genre_coherence(intermediates, target):
    sequence = list(intermediates) + [target]
    if any(x.get('genres') is None for x in sequence):
        return {'Coherence': None, 'COHERENCE_DEFINED': False,
                'adjacent_pair_count': None, 'pairs': [], 'reason': 'unresolved_metadata'}
    pairs = []
    for left, right in zip(sequence, sequence[1:]):
        shared = sorted(set(left['genres']) & set(right['genres']))
        pairs.append({'left_raw_id': left['raw_item_id'], 'right_raw_id': right['raw_item_id'],
                      'shared_genres': shared, 'pair_coherence': int(bool(shared))})
    return {'Coherence': sum(x['pair_coherence'] for x in pairs) / len(pairs) if pairs else None,
            'COHERENCE_DEFINED': bool(pairs), 'adjacent_pair_count': len(pairs), 'pairs': pairs,
            'reason': None if pairs else 'fewer_than_two_items'}

"""Frozen, explicit protocol rules. No model-generation configuration changes."""
from collections import Counter

PROTOCOL_VERSION = 'Formal-Evaluation-v1'
IOI_EPS = 1e-12
MAPPING_NOTICE = (
    'Evaluation uses the public ProRL MovieLens-1M SASRec checkpoint with a deterministically '
    'reconstructed RecBole token mapping. The mapping is strongly supported by exact '
    'reconstruction checks, but the original training-time mapping artifact was not publicly '
    'available; therefore compatibility is marked as LIKELY_COMPATIBLE rather than fully verified.'
)


def protocol_definition(checkpoint_path, checkpoint_sha256):
    return {
        'PROTOCOL_VERSION': PROTOCOL_VERSION, 'EVALUATOR_NAME': 'ProRL-style SASRec evaluator',
        'CHECKPOINT_PATH': str(checkpoint_path), 'CHECKPOINT_SHA256': checkpoint_sha256,
        'MAPPING_STATUS': 'LIKELY_COMPATIBLE', 'MAPPING_NOTICE': MAPPING_NOTICE,
        'HISTORY_LENGTH': 20, 'TARGET_EXCLUDED_FROM_EXTENSION': True,
        'IOI_EPS': IOI_EPS, 'IOI': 'natural_log(P_after + 1e-12) - natural_log(P_before + 1e-12)',
        'IOR': 'R_before - R_after; 1-based rank; higher is better',
        'TARGET_PROBABILITY': 'unchanged SASRecEvaluator.get_target_score: softmax across all 3884 rows, including PAD, no history masking',
        'RANK_TIES': 'delegate unchanged SASRecEvaluator.get_target_rank; torch.argsort(probs, descending=True), no new tiebreaker; tied order follows frozen torch 2.8.0+cpu implementation, not guaranteed cross-version/device',
        'PROXY_ACCEPTABILITY': 'mean of sequential sigmoid(sequence_embedding dot item_embedding), scoring each item BEFORE append, delegated to existing get_path_item_scores; NOT softmax probability or real CTR',
        'PROXY_EMPTY': 'null; PROXY_ACCEPTABILITY_DEFINED=False',
        'COHERENCE_TYPE': 'genre_overlap',
        'COHERENCE_SEQUENCE': 'all evaluation intermediates + target; no last-history-to-first-path edge',
        'COHERENCE_DENOMINATOR': 'number of adjacent pairs; length<2 => null and COHERENCE_DEFINED=False',
        'COHERENCE_UNRESOLVED': 'null for formally invalid paths; no silent dropping',
        'TITLE_RESOLUTION': 'exact first; unique The/A/An prefix/suffix normalization only, allowing one redundant matching article at both ends; case, other words, punctuation and year unchanged; >1 match ambiguous',
        'UNRESOLVED_POLICY': 'strict_invalid_with_optional_diagnostic_drop',
        'AMBIGUOUS_POLICY': 'treated as unresolved for formal eligibility, retain ambiguous resolution_status',
        'TARGET_FIRST_POLICY': 'exclude target from evaluation extension, preserve original position metadata',
        'TARGET_IDENTITY': 'resolved canonical MovieLens item ID, retaining raw titles and 1-based original positions; exclude ALL target occurrences',
        'RAW_TARGET_EXACT_PRESENT': 'additional audit field only; raw list is never rewritten',
        'AGGREGATION_POLICY': 'mean over formally valid paths only',
        'UNDEFINED_MEAN_POLICY': 'exclude null values per metric; record defined_path_count; denominator 0 => null, never substitute zero',
        'DIAGNOSTIC_AGGREGATION': 'DIAGNOSTIC_ONLY records are excluded even if status is valid',
        'SR_DENOMINATOR': 'all nonempty string-list parsed paths, including unresolved paths; numerator target_present among these paths',
        'TARGET_LAST_RATE_DENOMINATOR': 'same parsed-path denominator as SR, not formal-valid-only',
        'VALID_ITEM_RATE': 'resolved item occurrences / raw path length; null for no parsed items',
        'DUPLICATE_ITEM_COUNT': 'resolved canonical-ID occurrences beyond first; unresolved/ambiguous tokens not counted as valid items',
        'HISTORY_REUSE_RATE': 'resolved non-target history item occurrences / max(all non-target path occurrences including unresolved,1)',
        'NEW_INTERMEDIATE_COUNT': 'resolved non-target occurrences not in fixed history; unresolved excluded',
        'BRIDGE_CANDIDATE_USAGE_COUNT': 'resolved non-target occurrences in saved algorithm candidate IDs; count occurrences, not unique IDs',
        'MECHANISM_AGGREGATION': 'path-level counts retained, valid-path means only; partial resolved counts for invalid paths are not main results',
        'BRIDGE_SOURCE': 'repro/results/case_study/user_1_bridge_selection.json, bridge_movie_candidates; no hardcoded movies',
        'PARSE_FAILURE': 'invalid_parse, no formal metrics, excluded from parsed-path SR denominator; never reparsed/repaired here',
        'EMPTY_INTERMEDIATES': 'target-only valid parsed path => IoI=0, IoR=0, proxy/coherence=null',
        'SEQUENCE_OVERFLOW': 'raise error above frozen max50; never truncate or change model config',
        'DIRECTIONS': {'IoI': 'higher', 'IoR': 'higher', 'Proxy Acceptability': 'higher', 'Coherence': 'higher', 'SR': 'higher'},
        'SCOPE': 'one fixed user, four frozen paths, no significance tests or general superiority claims',
    }


def prepare_path(parsed, resolver, target, history_raw_ids, bridge_raw_ids):
    original = parsed.get('path')
    parse_ok = (parsed.get('parse_success') is True and isinstance(original, list)
                and bool(original) and all(isinstance(x, str) for x in original))
    raw_path = list(original) if isinstance(original, list) else original
    items = []
    if parse_ok:
        for position, title in enumerate(original, 1):
            item = resolver.resolve(title)
            item.update(position=position, is_target=item['raw_item_id'] == target['raw_item_id'])
            items.append(item)
    intermediates = [x for x in items if not x['is_target']]
    missing = [x for x in intermediates if x['resolution_status'] != 'resolved']
    positions = [x['position'] for x in items if x['is_target']]
    ids = [x['raw_item_id'] for x in items if x['resolution_status'] == 'resolved']
    resolved_intermediates = [x for x in intermediates if x['resolution_status'] == 'resolved']
    history_count = sum(x['raw_item_id'] in history_raw_ids for x in resolved_intermediates)
    mechanism = {
        'HISTORY_REUSE_RATE': history_count / max(len(intermediates), 1) if parse_ok else None,
        'NEW_INTERMEDIATE_COUNT': sum(x['raw_item_id'] not in history_raw_ids for x in resolved_intermediates) if parse_ok else None,
        'BRIDGE_CANDIDATE_USAGE_COUNT': sum(x['raw_item_id'] in bridge_raw_ids for x in resolved_intermediates) if parse_ok else None,
    }
    status = 'invalid_parse' if not parse_ok else ('invalid_unresolved' if missing else 'valid')
    return {'raw_path': raw_path, 'items': items, 'evaluation_intermediates': intermediates,
            'FORMAL_METRIC_STATUS': status, 'STRICT_EVALUATION_VALID': status == 'valid',
            'DIAGNOSTIC_ONLY': False, 'mechanism_metrics': mechanism,
            'mechanism_denominators': {'history_item_count': history_count,
                                      'number_of_non_target_path_items': len(intermediates)},
            'validity': {'parse_success': parse_ok, 'path_length': len(items),
                'valid_item_rate': len(ids) / len(items) if items else None,
                'target_present': bool(positions), 'target_is_last': bool(items) and items[-1]['is_target'],
                'target_original_position': positions[0] if positions else None, 'target_original_positions': positions,
                'raw_target_exact_present': parse_ok and target['resolved_title'] in original,
                'unresolved_item_count': len(missing), 'unresolved_titles': [x['raw_title'] for x in missing],
                'ambiguous_item_count': sum(x['resolution_status'] == 'ambiguous' for x in missing),
                'duplicate_item_count': sum(n - 1 for n in Counter(ids).values())}}

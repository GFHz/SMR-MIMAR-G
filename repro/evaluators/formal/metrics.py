"""Unified formal metrics using unchanged SASRec primitives; no new rank logic."""
import math
from statistics import mean
from .coherence import genre_coherence
from .protocol import IOI_EPS


def interest_increase(before, after):
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (before, after)):
        raise ValueError('Invalid target probabilities')
    return math.log(after + IOI_EPS) - math.log(before + IOI_EPS)


def rank_increase(before, after):
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in (before, after)):
        raise ValueError('Ranks must be positive 1-based integers')
    return before - after


def score_intermediates(intermediates, target, history, adapter, evaluator):
    target_id = adapter.raw_item_id_to_internal(target['raw_item_id'])
    ids = [adapter.raw_item_id_to_internal(x['raw_item_id']) for x in intermediates]
    if target_id in history or target_id in ids:
        raise ValueError('Target leakage detected')
    if len(history) + len(ids) > evaluator.max_length:
        raise ValueError('No silent truncation: extended history exceeds frozen checkpoint length')
    before = evaluator.get_target_score(history, target_id)
    after = evaluator.get_target_score(history + ids, target_id)
    rank_before = evaluator.get_target_rank(history, target_id)
    rank_after = evaluator.get_target_rank(history + ids, target_id)
    proxy = evaluator.get_path_item_scores(history, ids)
    coherence = genre_coherence(intermediates, target)
    return {'P_before': before, 'P_after': after, 'IoI': interest_increase(before, after),
            'R_before': rank_before, 'R_after': rank_after, 'IoR': rank_increase(rank_before, rank_after),
            'ProxyAcceptability': proxy['SASREC_PROXY_ACCEPTABILITY'],
            'PROXY_ACCEPTABILITY_DEFINED': bool(ids), 'path_item_scores': proxy['path_item_scores'],
            'proxy_denominator': len(ids), **coherence,
            'mapped_evaluation_intermediates': ids, 'history_before_length': len(history),
            'history_after_length': len(history) + len(ids), 'target_in_extension': False}


def evaluate_prepared(prepared, target, history, adapter, evaluator, diagnostic_drop=False):
    if prepared['FORMAL_METRIC_STATUS'] == 'valid':
        metrics = score_intermediates(prepared['evaluation_intermediates'], target, history, adapter, evaluator)
        return {**prepared, 'metrics': metrics}, None
    empty = {key: None for key in ('P_before', 'P_after', 'IoI', 'R_before', 'R_after', 'IoR',
                                   'ProxyAcceptability', 'Coherence')}
    empty.update(PROXY_ACCEPTABILITY_DEFINED=False, COHERENCE_DEFINED=False)
    diagnostic = None
    if diagnostic_drop and prepared['FORMAL_METRIC_STATUS'] == 'invalid_unresolved':
        kept = [x for x in prepared['evaluation_intermediates'] if x['resolution_status'] == 'resolved']
        diagnostic = {'DIAGNOSTIC_ONLY': True, 'mode': 'diagnostic_drop_unresolved',
                      'excluded_items': [x for x in prepared['evaluation_intermediates'] if x['resolution_status'] != 'resolved'],
                      'metrics': score_intermediates(kept, target, history, adapter, evaluator)}
    return {**prepared, 'metrics': empty}, diagnostic


def aggregate(paths):
    eligible = [p for p in paths if not p.get('DIAGNOSTIC_ONLY', False)]
    valid = [p for p in eligible if p['FORMAL_METRIC_STATUS'] == 'valid']
    parsed = [p for p in eligible if p['validity']['parse_success']]
    main = {'total_path_count': len(eligible), 'valid_path_count': len(valid),
            'parsed_path_count': len(parsed), 'metric_defined_path_counts': {},
            'SR': sum(p['validity']['target_present'] for p in parsed) / len(parsed) if parsed else None,
            'TargetLastRate': sum(p['validity']['target_is_last'] for p in parsed) / len(parsed) if parsed else None}
    for name in ('IoI', 'IoR', 'ProxyAcceptability', 'Coherence'):
        values = [p['metrics'][name] for p in valid if p['metrics'][name] is not None]
        main[name + '_mean'] = mean(values) if values else None
        main['metric_defined_path_counts'][name] = len(values)
    mechanism = {'valid_path_count': len(valid), 'total_path_count': len(eligible)}
    for name in ('HISTORY_REUSE_RATE', 'NEW_INTERMEDIATE_COUNT', 'BRIDGE_CANDIDATE_USAGE_COUNT'):
        values = [p['mechanism_metrics'][name] for p in valid if p['mechanism_metrics'][name] is not None]
        mechanism[name + '_mean'] = mean(values) if values else None
    return main, mechanism

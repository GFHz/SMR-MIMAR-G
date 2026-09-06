"""Independent exact-ID adapter, reference ProRL/evaluator.py field2token_id.

Source: https://github.com/hongruhou89/ProRL at
506e91355377f546dcf51f684fc871fe57a9d5c8. Mapping interfaces rewritten here;
no semantic tokenizer and no copied evaluator body or fuzzy title correction.
"""
import csv
from .checkpoint_loader import DATA


class DataAdapter:
    def __init__(self, dataset):
        self.user_map = dataset.field2token_id['user_id']
        self.item_map = dataset.field2token_id['item_id']
        self.item_tokens = dataset.field2id_token['item_id']
        self.title_map = {}
        with (DATA / 'ml-1m-sas/ml-1m-sas.item').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                title, raw_id = row['title:token'], int(row['item_id:token'])
                if title in self.title_map:
                    raise ValueError(f'Ambiguous exact title: {title}')
                self.title_map[title] = raw_id

    def raw_user_id_to_internal(self, user_id):
        return int(self.user_map[str(user_id)])

    def raw_item_id_to_internal(self, item_id):
        index = int(self.item_map[str(item_id)])
        if index == 0 or str(self.item_tokens[index]) != str(item_id):
            raise ValueError('Item mapping roundtrip failed or padding used as item')
        return index

    def movie_title_to_raw_item_id(self, title):
        return self.title_map.get(title)

    def movie_title_to_internal_id(self, title):
        raw = self.movie_title_to_raw_item_id(title)
        return None if raw is None else self.raw_item_id_to_internal(raw)

    def history_raw_ids_to_internal(self, history):
        return [self.raw_item_id_to_internal(raw) for raw in history]

    def path_titles_to_internal(self, path):
        """Preserve original order, positions, and unresolved placeholders."""
        items = []
        for position, title in enumerate(path, 1):
            raw = self.movie_title_to_raw_item_id(title)
            internal = None if raw is None else self.raw_item_id_to_internal(raw)
            items.append({'position': position, 'title': title,
                          'raw_id': raw, 'internal_id': internal})
        return {'items': items, 'unresolved_items': [x for x in items if x['internal_id'] is None]}

    def evaluation_intermediates(self, path, target_raw_id, mode='STRICT_FAIL_MODE'):
        """Adapted rule: exclude ALL target occurrences, not just the final item."""
        if mode not in ('STRICT_FAIL_MODE', 'DROP_UNRESOLVED_DIAGNOSTIC_MODE'):
            raise ValueError('Unknown unresolved-item policy')
        mapping = self.path_titles_to_internal(path)
        target_positions = [x['position'] for x in mapping['items'] if x['raw_id'] == target_raw_id]
        unresolved = mapping['unresolved_items']
        strict_failure = bool(unresolved) and mode == 'STRICT_FAIL_MODE'
        intermediates = [x for x in mapping['items']
                         if x['raw_id'] != target_raw_id and x['internal_id'] is not None]
        return {**mapping, 'mode': mode, 'target_original_positions': target_positions,
                'target_original_position': target_positions[0] if target_positions else None,
                'status': 'unresolved_strict_failure' if strict_failure else 'ready',
                'diagnostic_only': mode == 'DROP_UNRESOLVED_DIAGNOSTIC_MODE',
                'evaluation_intermediates': None if strict_failure else intermediates,
                'error': 'Unresolved titles; path excluded, no score computed.' if strict_failure else None}

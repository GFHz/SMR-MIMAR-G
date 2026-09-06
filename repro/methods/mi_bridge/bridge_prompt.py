"""Append optional bridge planning information; leave baseline instructions intact."""
import json


def build_bridge_prompt(baseline_system_prompt, baseline_user_prompt, selected_bridge,
                        bridge_type, bridge_candidates):
    if bridge_type != 'direct':
        raise ValueError('This case study only supports the already selected direct bridge.')
    if '[BRIDGE CONTEXT]' in baseline_user_prompt:
        raise ValueError('Refusing a second bridge context block.')
    i = selected_bridge['user_existing_interest']
    g = selected_bridge['target_related_interest']
    context = ('\n\n[BRIDGE CONTEXT]\n'
               'The user has multiple interests.\n'
               f'The algorithm-selected interest transition direction is: {i} -> {g}.\n'
               f'{i} is an existing strong interest of the user.\n'
               f'{g} is a target-related interest that is weaker in the user history.\n'
               'The following unseen direct bridge candidates were retrieved using MovieLens genre metadata.\n'
               f'These candidates contain both {i} AND {g}:\n'
               + json.dumps([m['title'] for m in bridge_candidates], ensure_ascii=False) + '\n'
               'During path planning, you may use this bridge direction and these candidate movies '
               'to help transition gradually from the existing interest toward the target-side interest.\n'
               '[/BRIDGE CONTEXT]\n')
    return {'system_prompt': baseline_system_prompt,
            'user_prompt': baseline_user_prompt + context, 'bridge_context': context}

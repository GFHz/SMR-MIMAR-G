"""Strict single-next-item prompt for Dynamic MI-Bridge v0.1 only."""
from __future__ import annotations

import json


DYNAMIC_STEP_SYSTEM_PROMPT = """You are selecting exactly one next movie for the current planning step.

You MUST choose exactly one movie title from the supplied catalog-grounded candidate list.

Output ONLY the exact movie title.
Do not output explanations.
Do not output numbering.
Do not output bullets.
Do not output a list.
Do not output multiple movies.
Do not output genres.
Do not output reasoning.
Do not output any text before or after the movie title.

The selected movie must not be in the user's current history.
The selected movie must not be the target movie unless the target is explicitly included in the allowed candidate list."""


def build_next_item_prompt(base_system_prompt, base_user_prompt, step_context):
    """Replace the incompatible full-path system instruction for dynamic steps.

    ``base_system_prompt`` remains in the interface so callers cannot
    accidentally alter the frozen baseline/static prompt. It is intentionally
    not forwarded: dynamic planning is a distinct one-item interaction mode.
    """
    if not isinstance(base_system_prompt, str) or not isinstance(base_user_prompt, str):
        raise TypeError("Prompts must be strings")
    bridge = step_context["selected_bridge"]
    if bridge["bridge_type"] not in ("direct", "multi_hop"):
        raise ValueError("A feasible bridge is required")
    candidates = (bridge["direct_candidates"] if bridge["bridge_type"] == "direct"
                  else bridge["selected_fallback"]["left_candidates"] + bridge["selected_fallback"]["right_candidates"])
    context = (
        "\n\n[DYNAMIC MI-BRIDGE STEP]\n"
        f"Current planning step: {step_context['step']}\n"
        f"Current existing interest: {bridge['existing_interest']}\n"
        f"Current target-side interest: {bridge['target_interest']}\n"
        f"Current bridge type: {bridge['bridge_type']}\n"
        "Allowed candidates:\n"
        + json.dumps([x["title"] for x in candidates], ensure_ascii=False) + "\n"
        "\nSelect exactly one movie from the allowed candidates.\n\n"
        "Return only the exact title.\n"
        "[/DYNAMIC MI-BRIDGE STEP]\n"
    )
    return {"system_prompt": DYNAMIC_STEP_SYSTEM_PROMPT,
            "user_prompt": base_user_prompt.rstrip() + context,
            "dynamic_context": context,
            "replaced_base_system_prompt": True}

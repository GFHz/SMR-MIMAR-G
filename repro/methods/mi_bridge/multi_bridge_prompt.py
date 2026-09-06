"""Advisory Top-K context builder for MI-Bridge v2; v1 prompt is untouched."""
from __future__ import annotations

import json


OPEN = "[MULTI-INTEREST BRIDGE CONTEXT]"
CLOSE = "[/MULTI-INTEREST BRIDGE CONTEXT]"
MAX_CANDIDATES_PER_BRIDGE = 5
SOFT_NOVELTY_GUIDANCE = (
    "Use the selected historical interests as semantic anchors rather than simply replaying "
    "the user's history. When possible, prefer novel catalog-grounded intermediate items "
    "that preserve continuity with the user's existing interests while moving toward the "
    "target. Historical items are not strictly forbidden, and the final path remains your "
    "planning decision."
)


def _candidate_titles(bridge, max_candidates_per_bridge):
    candidates = bridge["candidates"]
    if bridge["mode"] == "direct":
        return [movie["title"] for movie in candidates[:max_candidates_per_bridge]]
    return {
        "left": [movie["title"] for movie in candidates["left"][:max_candidates_per_bridge]],
        "right": [movie["title"] for movie in candidates["right"][:max_candidates_per_bridge]],
    }


def build_multi_bridge_prompt(baseline_system_prompt, baseline_user_prompt, selected_bridges,
                              max_candidates_per_bridge=MAX_CANDIDATES_PER_BRIDGE,
                              novelty_guidance=False):
    if not selected_bridges:
        raise ValueError("selected_bridges must contain at least one feasible bridge")
    if (not isinstance(max_candidates_per_bridge, int) or isinstance(max_candidates_per_bridge, bool)
            or max_candidates_per_bridge < 1):
        raise ValueError("max_candidates_per_bridge must be a positive integer")
    if OPEN in baseline_user_prompt or CLOSE in baseline_user_prompt:
        raise ValueError("Refusing a second multi-bridge context block")
    lines = [
        "", "", OPEN,
        "The user has multiple relevant historical interests that may support the transition toward the target.",
        "Several explicit bridge directions have been identified as advisory suggestions for transition planning.",
        "Preferred bridge directions:",
    ]
    for index, bridge in enumerate(selected_bridges, 1):
        lines.extend([
            "",
            f"Bridge {index}:",
            f"Existing interest: {bridge['existing_interest']}",
            f"Target-side interest: {bridge['target_interest']}",
            f"Catalog-grounded mode: {bridge['mode']}",
        ])
        if bridge["mode"] == "multi_hop":
            lines.append(f"Intermediate interest: {bridge['fallback_intermediate_genre']}")
        lines.append("Candidate movies: " + json.dumps(
            _candidate_titles(bridge, max_candidates_per_bridge), ensure_ascii=False))
    lines.extend([
        "",
        "When planning the influence path, jointly consider these bridge directions.",
        "Preserve natural continuity with the user's existing interests and gradually move toward target-side interests.",
        "You may combine information from multiple bridges.",
        "You do not need to use every bridge or every candidate.",
        "The bridge directions prefer the overall transition structure rather than constraining every individual path node.",
    ])
    if novelty_guidance:
        lines.extend(["", "Soft novelty guidance:", SOFT_NOVELTY_GUIDANCE])
    lines.extend([CLOSE, ""])
    context = "\n".join(lines)
    prompt_candidates = [_candidate_titles(bridge, max_candidates_per_bridge) for bridge in selected_bridges]
    return {
        "system_prompt": baseline_system_prompt,
        "user_prompt": baseline_user_prompt + context,
        "multi_bridge_context": context,
        "max_candidates_per_bridge": max_candidates_per_bridge,
        "prompt_candidates_per_bridge": prompt_candidates,
        "novelty_guidance": bool(novelty_guidance),
    }

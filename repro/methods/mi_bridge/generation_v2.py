"""Separate auditable generation API for MI-Bridge v2 (not run on import)."""
from __future__ import annotations

from copy import deepcopy

from repro import local_llm
from repro.movie_path_parser import parse_movie_path
from repro.run_minimal_baseline import with_attempts
from repro.utils import write_json

from .multi_bridge_prompt import build_multi_bridge_prompt
from .topk_bridge import METHOD_VERSION, select_topk_bridges


def prepare_v2_generation(system_prompt, user_prompt, bridge_ranking, user_interests, k=3,
                          prefer_distinct_existing_interests=True, max_candidates_per_bridge=5,
                          novelty_guidance=False):
    selected = select_topk_bridges(bridge_ranking, k, prefer_distinct_existing_interests)
    if not selected:
        raise ValueError("No feasible bridge is available for MI-Bridge v2")
    prompt = build_multi_bridge_prompt(
        system_prompt, user_prompt, selected, max_candidates_per_bridge, novelty_guidance)
    return {
        "method_version": METHOD_VERSION,
        "top_k": k,
        "distinct_existing_interest_policy": prefer_distinct_existing_interests,
        "user_interests": deepcopy(user_interests),
        "full_bridge_ranking": deepcopy(list(bridge_ranking)),
        "selected_bridges": selected,
        "prompt_candidate_cap": max_candidates_per_bridge,
        "novelty_guidance": bool(novelty_guidance),
        **prompt,
    }


def generate_v2(prepared, formatting_user_prompt, user_id, target, output_path=None):
    """Use the frozen adapter, retry wrapper, two-turn behavior, and parser."""
    plan = with_attempts(local_llm.generate(prepared["system_prompt"], prepared["user_prompt"]))
    messages = [
        {"role": "system", "content": prepared["system_prompt"]},
        {"role": "user", "content": prepared["user_prompt"]},
        {"role": "assistant", "content": plan["content"]},
        {"role": "user", "content": formatting_user_prompt},
    ]
    formatted = with_attempts(local_llm.generate_messages(messages))
    parsed = parse_movie_path(formatted["content"])
    record = {
        "method_version": METHOD_VERSION,
        "user_id": user_id,
        "target": deepcopy(target),
        "top_k": prepared["top_k"],
        "user_interests": deepcopy(prepared["user_interests"]),
        "full_bridge_ranking": deepcopy(prepared["full_bridge_ranking"]),
        "selected_bridges": deepcopy(prepared["selected_bridges"]),
        "prompt_candidate_cap": prepared["prompt_candidate_cap"],
        "novelty_guidance": prepared.get("novelty_guidance", False),
        "prompt_candidates_per_bridge": deepcopy(prepared["prompt_candidates_per_bridge"]),
        "multi_bridge_context": prepared["multi_bridge_context"],
        "raw_prompt": {"system_prompt": prepared["system_prompt"], "user_prompt": prepared["user_prompt"]},
        "raw_model_response": {"plan": plan, "formatting": formatted},
        "parsed_path": parsed["path"],
        "parse_status": {"success": parsed["parse_success"], "strategy": parsed.get("parse_strategy_used"), "error": parsed.get("error")},
    }
    if output_path is not None:
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite v2 record: {output_path}")
        write_json(output_path, record)
    return record

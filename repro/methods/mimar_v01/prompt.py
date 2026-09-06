"""One-item MIMAR prompt and strict exact-candidate parser."""
import ast
import json

SYSTEM_PROMPT = """Select exactly one next movie from the supplied allowed candidates.
TARGET ITEM IS REFERENCE ONLY.
THE TARGET ITEM MUST NEVER BE OUTPUT.
OUTPUT MUST BE EXACTLY ONE TITLE FROM ALLOWED CANDIDATES.
DO NOT OUTPUT ANY TITLE NOT PRESENT IN ALLOWED CANDIDATES.
DO NOT OUTPUT MULTIPLE TITLES.
DO NOT EXPLAIN.
DO NOT REWRITE OR ABBREVIATE A TITLE."""

CORRECTION_INSTRUCTION = ("INVALID OUTPUT. Choose exactly one movie title from the ALLOWED "
                          "CANDIDATES list. The target item is forbidden. Output the exact title only.")


def build_prompt(demographics, long_term, active, target, route_row, candidates):
    titles = [x["title"] for x in candidates]
    context = {
        "demographics": demographics,
        "long_term_interests": long_term,
        "current_active_interests": active,
        "target_reference_only_not_selectable": target,
        "all_target_genres": target["genres"],
        "current_route": {"interest": route_row["interest"], "target_genre": route_row["target_genre"]},
        "route_score_components": {k: route_row[k] for k in
                                   ("long_term", "active", "cooccurrence", "penalty", "continuity_bonus", "route_score")},
        "route_note": "The route is a direction for this planning step; candidates need not all contain both endpoint genres."
    }
    user_prompt = (json.dumps(context, ensure_ascii=False, indent=2)
                   + "\n\nTARGET ITEM IS REFERENCE ONLY. THE TARGET ITEM MUST NEVER BE OUTPUT.\n"
                   + "\nALLOWED CANDIDATES:\n" + json.dumps(titles, ensure_ascii=False, indent=2)
                   + "\n\nYour entire response must be exactly one title copied verbatim from the ALLOWED CANDIDATES list.")
    return {"system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt,
            "allowed_titles": titles}


def correction_prompt(prompt):
    corrected = dict(prompt)
    corrected["user_prompt"] = prompt["user_prompt"] + "\n\n" + CORRECTION_INSTRUCTION
    return corrected


def parse_one_title(raw, allowed_titles):
    allowed = set(allowed_titles)
    if not isinstance(raw, str) or not raw.strip():
        return {"success": False, "title": None, "error": "empty_or_non_string"}
    text = raw.strip()
    if text in allowed:
        return {"success": True, "title": text, "error": None}
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except Exception:
            continue
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str) and value[0] in allowed:
            return {"success": True, "title": value[0], "error": None}
        if isinstance(value, (list, tuple, dict, set)):
            return {"success": False, "title": None, "error": "not_exactly_one_item"}
    return {"success": False, "title": None, "error": "not_exact_allowed_candidate"}

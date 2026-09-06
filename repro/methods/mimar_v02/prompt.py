"""MIMAR-v0.2 exact-title prompt; route labels are explicitly non-items."""
import json
from repro.methods.mimar_v01.prompt import parse_one_title

MAX_LLM_RETRIES=2
SYSTEM_PROMPT="""Select exactly one next movie from the supplied allowed candidates.
TARGET ITEM IS REFERENCE ONLY. THE TARGET ITEM MUST NEVER BE OUTPUT.
ROUTE LABELS ARE ABSTRACT GENRE CONCEPTS. THEY ARE NOT MOVIE TITLES.
NEVER OUTPUT THE ROUTE START GENRE OR TARGET GENRE AS A TITLE.
Choose exactly one movie title copied verbatim from ALLOWED CANDIDATES.
Do not output explanations, numbering, reasoning, multiple titles, rewrites, or abbreviations."""
CORRECTION="INVALID OUTPUT. Copy exactly one movie title from ALLOWED CANDIDATES. Do not output the target or route genre labels."


def build_prompt(demographics,long_term,active,target,coverage,need,route,candidates):
    titles=[x["title"] for x in candidates]
    context={"demographics":demographics,"long_term_interests":long_term,
        "current_active_interests":active,"target_reference_only_not_selectable":target,
        "all_target_genres":target["genres"],"target_coverage":coverage,"target_need":need,
        "current_route":{"start_interest_genre":route["interest"],"target_attribute_genre":route["target_genre"]},
        "route_score_components":{k:route[k] for k in ("long_term","active","need","penalty","continuity_bonus","route_score")},
        "route_feasibility":{"count_interest":route["count_interest"],"count_target_genre":route["count_target_genre"],
                             "count_joint":route["count_joint"],"normalized_cooccurrence":route["normalized_cooccurrence"],"feasible":route["feasible"]},
        "route_note":"Route labels are abstract genre concepts and must never be output as movie titles."}
    user=(json.dumps(context,ensure_ascii=False,indent=2)+"\n\nALLOWED CANDIDATES:\n"
          +json.dumps(titles,ensure_ascii=False,indent=2)
          +"\n\nYour entire response must be exactly one title copied verbatim from the list above.")
    return {"system_prompt":SYSTEM_PROMPT,"user_prompt":user,"allowed_titles":titles}


def correction_prompt(prompt):
    return {**prompt,"user_prompt":prompt["user_prompt"]+"\n\n"+CORRECTION}

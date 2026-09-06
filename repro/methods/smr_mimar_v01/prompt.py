"""One-item exact-title prompt for simultaneous static routes."""
import ast,json
MAX_LLM_RETRIES=2
SYSTEM_PROMPT="""Select exactly one next movie from the supplied catalog-grounded candidates.
The routes are simultaneous migration directions, not a single route to follow.
A candidate may support multiple routes at the same time.
You do not need to select one route before selecting the movie.
Choose the candidate that forms the most natural next step in the overall migration toward the target.
ROUTE LABELS ARE GENRE CONCEPTS, NOT MOVIE TITLES.
The target item is reference only and must not be selected.
Output exactly one title copied verbatim from ALLOWED CANDIDATES.
Do not explain, number, rewrite, abbreviate, or output multiple titles."""
CORRECTION="INVALID OUTPUT. Copy exactly one movie title verbatim from the unchanged ALLOWED CANDIDATES."

def build_prompt(demographics,long_term,target,route_set,current_path,candidates):
    context={"demographics":demographics,"frozen_long_term_multi_interest_profile":long_term,
      "target_reference_only":target,"all_target_genres":target["genres"],
      "static_selected_route_set":[{"start_interest":r["interest"],"target_attribute":r["target_genre"],"route_prior":r["route_prior"]} for r in route_set],
      "current_generated_path":list(current_path)}
    allowed=[{"title":x["title"],"genres":x["genres"],"supported_routes":x["supported_routes"],"route_support_count":x["route_support_count"]} for x in candidates]
    user=json.dumps(context,ensure_ascii=False,indent=2)+"\n\nALLOWED CANDIDATES:\n"+json.dumps(allowed,ensure_ascii=False,indent=2)+"\n\nYour entire response must be exactly one title copied verbatim from the list above."
    return {"system_prompt":SYSTEM_PROMPT,"user_prompt":user,"allowed_titles":[x["title"] for x in candidates]}

def correction_prompt(prompt):return {**prompt,"user_prompt":prompt["user_prompt"]+"\n\n"+CORRECTION}
def parse_one_title(raw,allowed_titles):
    if not isinstance(raw,str) or not raw.strip():return {"success":False,"title":None,"error":"empty_or_non_string"}
    text=raw.strip();allowed=set(allowed_titles)
    if text in allowed:return {"success":True,"title":text,"error":None}
    for loader in (json.loads,ast.literal_eval):
        try:value=loader(text)
        except Exception:continue
        if isinstance(value,list) and len(value)==1 and isinstance(value[0],str) and value[0] in allowed:return {"success":True,"title":value[0],"error":None}
        if isinstance(value,(list,tuple,dict,set)):return {"success":False,"title":None,"error":"not_exactly_one_item"}
    return {"success":False,"title":None,"error":"not_exact_allowed_candidate"}

"""One-user all-accept mechanism smoke test for MIMAR-v0.2."""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from repro.local_llm import MODEL, request
from repro.methods.mi_bridge.retrieve_bridge_movies import load_movies
from repro.methods.mimar_v01.feedback import ACCEPT
from repro.methods.mimar_v02.coverage import coverage_complete, target_need
from repro.methods.mimar_v02.planner import MIMARV02Planner

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "repro/results/mimar_v02/smoke_all_accept"
MANIFEST = ROOT / "repro/results/pilot/pilot_manifest.json"
MOVIES = ROOT / "dataset/ml-1m/movies.dat"
CONFIG = ROOT / "repro/local_config.json"
CONTROLLED_USERS = [419, 5021, 2677, 3113, 2249]
SMOKE_MAX_INTERMEDIATE_STEPS = 6


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")


def protected_files(include_v02=True):
    files=[]
    for directory in (ROOT/"repro/methods", ROOT/"repro/evaluators"):
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts: continue
            if not include_v02 and "mimar_v02" in path.parts: continue
            files.append(path)
    files += [ROOT/name for name in read_json(ROOT/"repro/source_hashes.json")["sha256"]]
    return sorted(set(files))


def hashes(files):
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in files}


def llm_select(prompt, config):
    payload={"model":MODEL,"messages":[
        {"role":"system","content":prompt["system_prompt"]},
        {"role":"user","content":prompt["user_prompt"]}],
        "stream":False,"think":config["thinking_mode"],"keep_alive":"5m",
        "options":{"temperature":config["temperature"],"num_ctx":config["context_length"],
                   "seed":config["seed"],"num_predict":config["max_output_tokens"]}}
    started=time.perf_counter(); raw,response=request("/api/chat",payload,timeout=config["request_timeout_seconds"])
    if response.get("error") or not response.get("done"):
        raise RuntimeError("Ollama returned incomplete/error response")
    return response.get("message",{}).get("content",""),{
        "request":payload,"raw_http_response":raw,"response":response,
        "elapsed_seconds":time.perf_counter()-started}


def select_user(manifest, catalog):
    by_id={x["id"]:x for x in catalog}; eligible=[]
    for source in manifest["users"]:
        if source["user_id"] not in CONTROLLED_USERS: continue
        history=[deepcopy(by_id[int(i)]) for i in source["positive_movie_ids"]]
        target=deepcopy(by_id[int(source["target"]["movie_id"])])
        if len(target["genres"])<2: continue
        planner=MIMARV02Planner(history,target,catalog,source["demographics"],user_id=source["user_id"],path_id=1)
        plan=planner.plan(); attrs={x["target_genre"] for x in plan["feasible_routes"]}
        active={g for g,v in planner.active.items() if v>=.05}
        if len(active)>=2 and len(attrs)>=2: eligible.append((source["user_id"],source,history,target))
    if not eligible: raise RuntimeError("No controlled user satisfies the declared eligibility criteria")
    return min(eligible,key=lambda x:x[0])


def main():
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    frozen_files=protected_files(include_v02=False); formula_files=protected_files(include_v02=True)
    frozen_before=hashes(frozen_files); formulas_before=hashes(formula_files)
    manifest=read_json(MANIFEST); catalog=load_movies(MOVIES); config=read_json(CONFIG)
    user_id,source,history,target=select_user(manifest,catalog)
    planner=MIMARV02Planner(history,target,catalog,source["demographics"],user_id=user_id,path_id=1)
    OUT.mkdir(parents=True)
    initial={"user_id":user_id,"target":target,"target_genres":target["genres"],
        "selection_rule":"smallest controlled user_id with >=2 target genres, >=2 initial active interests, and feasible routes toward >=2 target genres",
        "long_term_interests":planner.long_term,"initial_short_term_interests":planner.active,
        "initial_active_interest_set":sorted(g for g,v in planner.active.items() if v>=.05),
        "coverage_0":planner.coverage,"need_0":target_need(planner.coverage)}
    save_new(OUT/"initial_state.json",initial)
    steps=[]; invalid_count=0; retry_count=0
    while planner.status in ("ready","running") and len(planner.accepted)<SMOKE_MAX_INTERMEDIATE_STEPS:
        calls=[]
        def choose(prompt):
            content,api=llm_select(prompt,config); calls.append({"prompt":deepcopy(prompt),"raw_output":content,"api":api}); return content
        previous=planner.previous_route; record=planner.step(choose,lambda item,state:ACCEPT)
        if "plan" not in record: break
        plan=record["plan"]; row=plan["selected_route"]
        routes=[]
        for rank,item in enumerate(plan["route_table"],1):
            routes.append({**item,"rank":rank,"C_INCLUDED_IN_SCORE":False})
        route=(row["interest"],row["target_genre"]) if row else None
        switch=bool(previous and route!=previous)
        if switch:
            reason="route reranking after accepted-state and target-Need updates"
        elif previous:
            reason="route recomputed; previous route remained highest-scoring"
        else: reason="initial route selection"
        new_routes=[]
        for interest in record.get("newly_activated",[]):
            for item in routes:
                if item["interest"]==interest:
                    new_routes.append({**item,"score_gap_to_selected":None if row is None else row["route_score"]-item["route_score"]})
        step={"step":record["step"],"user_id":user_id,"target":target,
            "active_interests":record["active_before"],"coverage_before":record["coverage_before"],
            "need_before":record["need_before"],"full_route_table":routes,"selected_route":row,
            "selected_start_interest":None if row is None else row["interest"],
            "selected_target_attribute":None if row is None else row["target_genre"],
            "previous_route":previous,"route_switch":switch,"route_switch_reason":reason,
            "allowed_candidates":[{"title":x["title"],"genres":x["genres"],"tier":x["tier"]} for x in plan["candidates"]],
            "exact_llm_prompt":record["prompt"],"llm_attempts":record["llm_attempts"],
            "raw_llm_output":record["raw"],"parse_result":record["parsed"],
            "selected_item":record.get("selected_item"),"selected_item_genres":None if not record.get("selected_item") else record["selected_item"]["genres"],
            "feedback":"ACCEPT","active_before":record["active_before"],"active_after":record.get("active_after"),
            "newly_activated":record.get("newly_activated",[]),"newly_deactivated":record.get("deactivated",[]),
            "new_interest_routes":new_routes,"new_interest_route_win":bool(row and row["interest"] in record.get("newly_activated",[])),
            "coverage_updates":record.get("coverage_updates",{}),"coverage_after":record.get("coverage_after"),
            "need_after":target_need(record.get("coverage_after",record["coverage_before"])),
            "planner_status":record["status"],"api_calls":calls}
        save_new(OUT/f"step_{record['step']}.json",step);steps.append(step)
        retry_count+=record.get("llm_retry_count",0)
        invalid_count+=sum(not x["parse_result"]["success"] for x in record["llm_attempts"])
        if planner.status not in ("ready","running"): break
    selected=[(x["selected_start_interest"],x["selected_target_attribute"]) for x in steps if x["selected_route"]]
    target_attrs=[x[1] for x in selected]; target_switches=sum(a!=b for a,b in zip(target_attrs,target_attrs[1:]))
    route_switches=sum(a!=b for a,b in zip(selected,selected[1:]))
    coverage_updates=sum(sum(1 for v in x["coverage_updates"].values() if v>0) for x in steps)
    final_status=planner.status
    if len(planner.accepted)>=SMOKE_MAX_INTERMEDIATE_STEPS and final_status=="running": final_status="smoke_max_intermediate_steps"
    prompt_checks={
        "abstract_genre_concepts":all("ABSTRACT GENRE CONCEPTS" in x["exact_llm_prompt"]["system_prompt"] for x in steps),
        "not_movie_titles":all("NOT MOVIE TITLES" in x["exact_llm_prompt"]["system_prompt"] for x in steps),
        "forbid_genre_as_title":all("NEVER OUTPUT THE ROUTE START GENRE OR TARGET GENRE AS A TITLE" in x["exact_llm_prompt"]["system_prompt"] for x in steps),
        "exact_allowed_copy":all("copied verbatim from ALLOWED CANDIDATES" in x["exact_llm_prompt"]["user_prompt"] for x in steps)}
    frozen_after=hashes(frozen_files); formulas_after=hashes(formula_files)
    frozen_changed=sorted(k for k in frozen_before if frozen_before[k]!=frozen_after[k])
    formula_changed=sorted(k for k in formulas_before if formulas_before[k]!=formulas_after[k])
    summary={"method":"MIMAR-v0.2","mechanism_only":True,"all_valid_intermediates":"ACCEPT",
        "user_id":user_id,"target":target,"target_genre_count":len(target["genres"]),
        "intermediate_attempts":len(steps),"accepted_intermediates":len(planner.accepted),
        "cooccurrence_used_only_as_gate":all(not x["C_INCLUDED_IN_SCORE"] for s in steps for x in s["full_route_table"]),
        "cooccurrence_included_in_route_score":False,"target_coverage_updates":coverage_updates,
        "target_attribute_switch_count":target_switches,
        "new_interest_activation_count":sum(len(x["newly_activated"]) for x in steps),
        "new_interest_route_win_count":sum(x["new_interest_route_win"] for x in steps),
        "route_switch_count":route_switches,"coverage_stop_triggered":coverage_complete(planner.coverage),
        "invalid_llm_output_count":invalid_count,"llm_retry_count":retry_count,
        "prompt_checks":prompt_checks,"final_coverage":planner.coverage,"final_need":target_need(planner.coverage),
        "final_path":planner.snapshot()["final_path"],"stop_reason":final_status,
        "sasrec_used_during_planning":False,"formal_evaluator_modified":False,
        "frozen_methods_modified":bool(frozen_changed),"mimar_v02_formulas_modified":bool(formula_changed),
        "frozen_changed_files":frozen_changed,"v02_changed_during_run":formula_changed}
    summary["complete"]=bool(steps and len(planner.accepted)==len(steps) and all(prompt_checks.values())
        and not frozen_changed and not formula_changed)
    save_new(OUT/"summary.json",summary)
    save_new(OUT/"run_metadata.json",{"timestamp":datetime.now(timezone.utc).isoformat(),
        "num_paths":1,"smoke_max_intermediate_steps":SMOKE_MAX_INTERMEDIATE_STEPS,"llm_config":config,
        "frozen_hashes_before":frozen_before,"frozen_hashes_after":frozen_after,
        "v02_hashes_before":formulas_before,"v02_hashes_after":formulas_after,"formal_evaluation_run":False})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=="__main__": main()

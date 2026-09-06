"""Generate/evaluate the one-shot Dynamic-No-Bridge controlled ablation."""
from __future__ import annotations

import json, statistics, time
from collections import Counter
from datetime import datetime, timezone

from repro import local_llm
from repro.evaluators.formal.metrics import evaluate_prepared
from repro.evaluators.formal.protocol import PROTOCOL_VERSION, prepare_path
from repro.evaluators.formal.title_resolver import TitleResolver
from repro.experiments.pilot.evaluate_pilot_formal import CHECKPOINT, EXPECTED_CHECKPOINT_SHA, protected_snapshot, sha
from repro.experiments.pilot.run_catalog_grounded_ablation import EXPECTED_SELECTED, preflight
from repro.methods.dynamic_no_bridge.method import (CANDIDATE_LIMIT, MAX_INTERMEDIATE_STEPS,
    METHOD_VERSION, WINDOW_SIZE, allowed_candidates, interest_profile, validate_one)
from repro.methods.dynamic_no_bridge.prompt import build_prompt
from repro.utils import ROOT, read_json, verify_sources, write_json

OUT=ROOT/"repro/results/dynamic_no_bridge_v01/controlled_5user"
DYNAMIC_MI=ROOT/"repro/results/dynamic_mi_bridge_v01/controlled_5user"
STATIC_EVAL=ROOT/"repro/results/pilot/catalog_grounded_ablation_v1/evaluation/path_level_metrics.json"
USERS=[419,5021,2677,3113,2249]


def norm(user,pool):
    h=[{"id":x["movie_id"],"title":x["title"],"genres":list(x["genres"])} for x in user["history"]]
    t={"id":user["target"]["movie_id"],"title":user["target"]["title"],"genres":list(user["target"]["genres"])}
    p=[{"id":x["movie_id"],"title":x["title"],"genres":list(x["genres"])} for x in pool]
    return h,t,p


def saver(folder):
    n=[0]
    def save(record):
        n[0]+=1; path=folder/f"transport_attempt_{n[0]}.json"
        if path.exists(): raise FileExistsError(path)
        write_json(path,record); return str(path)
    return save,n


def generate(user,pool,index):
    window,target,pool=norm(user,pool); initial=list(window); used=[]; trajectory=[]; invalid=0; retries=0
    for step in range(1,MAX_INTERMEDIATE_STEPS+1):
        stats,top5=interest_profile(window); candidates=allowed_candidates(pool,window,[x["id"] for x in used],target["id"])
        if not candidates:
            stop="no_valid_continuation"; break
        prompt=build_prompt(user,window,top5,target,candidates,step)
        save_fn,count=saver(OUT/"generation/raw"/str(user["user_id"])/f"dynamic_no_bridge_path_{index}"/f"step_{step:02d}")
        old=local_llm.save_record
        try:
            local_llm.save_record=save_fn; response=local_llm.generate(prompt["system_prompt"],prompt["user_prompt"])
        finally: local_llm.save_record=old
        retries+=max(count[0]-1,0); result=validate_one(response["content"],candidates)
        before=list(window); dropped=added=None
        if result["success"]:
            dropped=window[0]; added=result["resolved_item"]; used.append(added); window=window[1:]+[added]; stop=None
        else:
            invalid+=1; stop="no_valid_continuation"
        after_stats,after_top5=interest_profile(window)
        trajectory.append({"step":step,"window_before":before,"genre_frequency_profile":stats,
            "top_5_interests":top5,"target_side_user_freq":{g:next((x["frequency"] for x in stats if x["genre"]==g),0.0) for g in target["genres"]},
            "allowed_candidate_count":len(candidates),"allowed_candidates":candidates,"prompt":prompt,
            "raw_llm_output":response["content"],"thinking":response["thinking"],"response_metadata":response["response"],
            "parser_result":{"success":result["success"],"title":result["title"],"error":result["error"]},
            "resolved_item":result["resolved_item"],"dropped_oldest_item":dropped,"added_item":added,
            "window_after":window,"updated_genre_frequency_profile":after_stats,"updated_top_5_interests":after_top5,
            "stop_reason":stop})
        if stop: break
    else: stop="max_intermediate_steps"
    # Endpoint normalization is deterministic and outside the LLM interaction.
    normalized_path=[x["title"] for x in used]+[target["title"]]
    return {"method":METHOD_VERSION,"user_id":user["user_id"],"path_index":index,"parsed_path":normalized_path,
        "parse_success":True,"intermediates":[x["title"] for x in used],"target_appended_by_normalization":True,
        "intermediate_count":len(used),"invalid_generation_count":invalid,"transport_retry_count":retries,"stop_reason":stop,
        "initial_target_genre_freq":trajectory[0]["target_side_user_freq"],
        "final_target_genre_freq":{g:next((x["frequency"] for x in interest_profile(window)[0] if x["genre"]==g),0.0) for g in target["genres"]},
        "trajectory":trajectory,"window_invariant":len(initial)==20 and all(len(x["window_before"])==20 and len(x["window_after"])==20 for x in trajectory)}


def evaluate(records,users,pools,method):
    from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
    from repro.evaluators.sasrec.data_adapter import DataAdapter
    from repro.evaluators.sasrec.evaluator import SASRecEvaluator
    model,dataset,_,_=load_checkpoint(); ev,adapter=SASRecEvaluator(model),DataAdapter(dataset)
    resolver=TitleResolver.from_movies_dat(ROOT/"dataset/ml-1m/movies.dat"); by={x["user_id"]:x for x in users}; rows=[]
    for r in records:
        u=by[r["user_id"]]; target=resolver.resolve(u["target"]["title"]); hist=[x["movie_id"] for x in u["history"]]
        prepared=prepare_path({"path":r["parsed_path"],"parse_success":True},resolver,target,set(hist),set())
        poolids={x["movie_id"] for x in pools[str(r["user_id"])]}
        violations=[x["raw_title"] for x in prepared["evaluation_intermediates"] if x["resolution_status"]!="resolved" or x["raw_item_id"] not in poolids]
        compliant=not violations
        if prepared["FORMAL_METRIC_STATUS"]=="valid" and not compliant:
            prepared["FORMAL_METRIC_STATUS"]="invalid_catalog_violation"; prepared["STRICT_EVALUATION_VALID"]=False
        scored,_=evaluate_prepared(prepared,target,adapter.history_raw_ids_to_internal(hist),adapter,ev,diagnostic_drop=False)
        m,v=scored["metrics"],scored["validity"]
        rows.append({"user_id":r["user_id"],"method":method,"path_index":r["path_index"],"formal_metric_status":scored["FORMAL_METRIC_STATUS"],
            "catalog_compliant":compliant,"parse_success":v["parse_success"],"target_is_last":v["target_is_last"],
            "IoI":m["IoI"],"IoR":m["IoR"],"ProxyAcceptability":m["ProxyAcceptability"],"Coherence":m["Coherence"],
            "PHRR":scored["mechanism_metrics"]["HISTORY_REUSE_RATE"],"HistoryReuseRate":scored["mechanism_metrics"]["HISTORY_REUSE_RATE"],
            "NewIntermediateCount":scored["mechanism_metrics"]["NEW_INTERMEDIATE_COUNT"]})
    return rows


METRICS=("IoI","IoR","ProxyAcceptability","Coherence","PHRR","HistoryReuseRate","NewIntermediateCount")
def avg(v):
    v=[x for x in v if x is not None]; return statistics.mean(v) if v else None
def aggregate(rows):
    valid=[x for x in rows if x["formal_metric_status"]=="valid"]
    return {"ValidRate":len(valid)/len(rows),"ComplianceRate":sum(x["catalog_compliant"] for x in rows)/len(rows),
        **{m:avg(x[m] for x in valid) for m in METRICS},"NormalizedTargetLastRate":avg(float(x["target_is_last"]) for x in rows if x["parse_success"]),
        "VALID_PATH_COUNT":len(valid),"TOTAL_PATH_COUNT":len(rows)}
def user_means(rows):
    return {(uid,m):{k:avg(x[k] for x in rows if x["user_id"]==uid and x["method"]==m and x["formal_metric_status"]=="valid") for k in METRICS+("target_is_last",)}
            for uid in USERS for m in sorted({x["method"] for x in rows})}
def pairs(rows,methods):
    means=user_means(rows); out={}
    for left,right in methods:
        out[f"{left}_vs_{right}"]={}
        for metric in ("IoI","IoR","ProxyAcceptability","Coherence","PHRR","target_is_last"):
            vals=[(means[uid,left][metric],means[uid,right][metric]) for uid in USERS if means[uid,left][metric] is not None and means[uid,right][metric] is not None]
            lower=metric=="PHRR"; rb=sum((b<a if lower else b>a) for a,b in vals); lb=sum((a<b if lower else a>b) for a,b in vals)
            out[f"{left}_vs_{right}"][metric]={"right_better":rb,"left_better":lb,"tie":len(vals)-rb-lb,"n":len(vals)}
    return out


def main():
    if OUT.exists(): raise FileExistsError(f"Refusing overwrite: {OUT}")
    verify_sources(); before=protected_snapshot()
    if sha(CHECKPOINT)!=EXPECTED_CHECKPOINT_SHA or PROTOCOL_VERSION!="Formal-Evaluation-v1": raise RuntimeError("Frozen evaluator mismatch")
    _,users,pools,_=preflight(check_service=True)
    if [x["user_id"] for x in users]!=USERS or EXPECTED_SELECTED!=USERS: raise RuntimeError("User mismatch")
    OUT.mkdir(parents=True,exist_ok=False); records=[]
    write_json(OUT/"protocol.json",{"method":METHOD_VERSION,"users":USERS,"window_size":WINDOW_SIZE,
        "max_intermediate_steps":MAX_INTERMEDIATE_STEPS,"paths_per_user":2,"candidate_limit":CANDIDATE_LIMIT,
        "candidate_order":"frozen controlled pool order; no ranking","bridge_constructed":False,"formal_protocol":PROTOCOL_VERSION,
        "target_endpoint_normalization":"intermediates + predefined target","configuration":read_json(local_llm.CONFIG)})
    for uid in USERS:
        user=next(x for x in users if x["user_id"]==uid)
        for i in (1,2):
            t=time.perf_counter(); r=generate(user,pools[str(uid)],i); r["elapsed_seconds"]=time.perf_counter()-t
            write_json(OUT/"generation/users"/str(uid)/f"dynamic_no_bridge_path_{i}.json",r); records.append(r)
            print(f"generated {len(records)}/10 user={uid} path={i} items={r['intermediate_count']} stop={r['stop_reason']}",flush=True)
    no_bridge=evaluate(records,users,pools,"Dynamic-No-Bridge")
    static_raw=read_json(STATIC_EVAL); static=[{**x,"method":"Static MI-Bridge","catalog_compliant":x["CATALOG_COMPLIANT"],
        "PHRR":x["HistoryReuseRate"]} for x in static_raw if x["method"]=="mi_bridge"]
    # Re-evaluate frozen Dynamic MI intermediates after deterministic target append.
    mi_records=[]
    for uid in USERS:
        u=next(x for x in users if x["user_id"]==uid)
        for i in (1,2):
            r=read_json(DYNAMIC_MI/"generation/users"/str(uid)/f"dynamic_path_{i}.json")
            mi_records.append({**r,"parsed_path":r["parsed_path"]+[u["target"]["title"]]})
    dynamic_mi=evaluate(mi_records,users,pools,"Dynamic MI-Bridge")
    allrows=static+no_bridge+dynamic_mi; methods={m:aggregate([x for x in allrows if x["method"]==m]) for m in ("Static MI-Bridge","Dynamic-No-Bridge","Dynamic MI-Bridge")}
    pair=pairs(allrows,[("Static MI-Bridge","Dynamic-No-Bridge"),("Dynamic-No-Bridge","Dynamic MI-Bridge"),("Static MI-Bridge","Dynamic MI-Bridge")])
    dynamics={"initial_vs_final_target_genre_frequencies":[{"user_id":r["user_id"],"path_index":r["path_index"],"initial":r["initial_target_genre_freq"],"final":r["final_target_genre_freq"]} for r in records],
        "mean_intermediate_count":avg(r["intermediate_count"] for r in records),"stop_reason_distribution":dict(Counter(r["stop_reason"] for r in records))}
    write_json(OUT/"evaluation/path_level_metrics.json",allrows); write_json(OUT/"evaluation/method_level_metrics.json",methods)
    write_json(OUT/"evaluation/paired_comparisons.json",pair); write_json(OUT/"dynamic_no_bridge_summary.json",dynamics)
    if protected_snapshot()!=before: raise RuntimeError("Protected files changed")
    attribution=("combination" if methods["Dynamic-No-Bridge"]["IoR"]>methods["Static MI-Bridge"]["IoR"] and methods["Dynamic MI-Bridge"]["IoR"]>methods["Dynamic-No-Bridge"]["IoR"] else "mixed")
    summary={"completed_at":datetime.now(timezone.utc).isoformat(),"methods":methods,"paired":pair,"dynamic_no_bridge":dynamics,
        "research_answers":{"no_bridge_improves_static_mean_IoR":methods["Dynamic-No-Bridge"]["IoR"]>methods["Static MI-Bridge"]["IoR"],
        "dynamic_mi_improves_no_bridge_mean_IoR":methods["Dynamic MI-Bridge"]["IoR"]>methods["Dynamic-No-Bridge"]["IoR"],"attribution":attribution},
        "no_significance_claim":True,"protection":{"formal_evaluator_modified":False,"static_mi_bridge_modified":False,"dynamic_mi_bridge_modified":False}}
    write_json(OUT/"summary.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

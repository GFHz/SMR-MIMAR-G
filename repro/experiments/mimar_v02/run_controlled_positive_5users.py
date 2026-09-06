"""Frozen five-user OFFLINE-POSITIVE pre-study for MIMAR-v0.2."""
from __future__ import annotations

import hashlib, json, statistics, time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from repro.local_llm import MODEL, request
from repro.methods.mi_bridge.retrieve_bridge_movies import load_movies
from repro.methods.mimar_v01.feedback import ACCEPT
from repro.methods.mimar_v02.coverage import coverage_complete, target_need
from repro.methods.mimar_v02.planner import MAX_INTERMEDIATE_STEPS, MIMARV02Planner

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"repro/results/mimar_v02/controlled_positive_5users"
MANIFEST=ROOT/"repro/results/pilot/pilot_manifest.json"
POOLS=ROOT/"repro/results/pilot/catalog_grounded_ablation_v1/candidate_pools.json"
MOVIES=ROOT/"dataset/ml-1m/movies.dat"; CONFIG=ROOT/"repro/local_config.json"
V01=ROOT/"repro/results/mimar_v01/controlled_positive_5users"
USERS=[419,5021,2677,3113,2249]; PATHS_PER_USER=2

def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_new(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def mean(v):
 v=[x for x in v if x is not None];return statistics.mean(v) if v else None
def protected_hashes():
 fs=[]
 for d in (ROOT/"repro/methods",ROOT/"repro/evaluators"):
  fs += [p for p in d.rglob("*.py") if "__pycache__" not in p.parts]
 fs += [ROOT/n for n in read_json(ROOT/"repro/source_hashes.json")["sha256"]]
 fs += [MANIFEST,POOLS,ROOT/"repro/results/dynamic_no_bridge_v01/controlled_5user/evaluation/method_level_metrics.json"]
 fs += [p for p in V01.rglob("*") if p.is_file()]
 return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(set(fs))}
def normalize(x):return {"id":int(x.get("id",x.get("movie_id"))),"title":x["title"],"genres":list(x["genres"])}
def llm_select(prompt,config):
 payload={"model":MODEL,"messages":[{"role":"system","content":prompt["system_prompt"]},{"role":"user","content":prompt["user_prompt"]}],"stream":False,"think":config["thinking_mode"],"keep_alive":"5m","options":{"temperature":config["temperature"],"num_ctx":config["context_length"],"seed":config["seed"],"num_predict":config["max_output_tokens"]}}
 started=time.perf_counter();raw,response=request("/api/chat",payload,timeout=config["request_timeout_seconds"])
 if response.get("error") or not response.get("done"):raise RuntimeError("Incomplete Ollama response")
 return response.get("message",{}).get("content",""),{"request":payload,"raw_http_response":raw,"response":response,"elapsed_seconds":time.perf_counter()-started}

def generate(user,pool,catalog,config,path_id):
 by_id={x["id"]:x for x in catalog};history=[deepcopy(by_id[int(i)]) for i in user["positive_movie_ids"]]
 target=deepcopy(by_id[int(user["target"]["movie_id"])]);controlled=[normalize(x) for x in pool]
 planner=MIMARV02Planner(history,target,catalog,user["demographics"],candidate_catalog=controlled,user_id=user["user_id"],path_id=path_id)
 initial_active=deepcopy(planner.active);trajectory=[];calls=0
 while planner.status in ("ready","running") and len(planner.accepted)<MAX_INTERMEDIATE_STEPS:
  api_calls=[]
  def choose(prompt):
   nonlocal calls
   raw,api=llm_select(prompt,config);calls+=1;api_calls.append({"prompt":deepcopy(prompt),"raw_output":raw,"api":api});return raw
  record=planner.step(choose,lambda item,state:ACCEPT)
  if "plan" not in record:break
  plan=record["plan"];row=plan["selected_route"]
  invalid=[x["parse_result"].get("error") for x in record["llm_attempts"] if not x["parse_result"]["success"]]
  trajectory.append({"step":record["step"],"target":target,"target_genres":target["genres"],"long_term_interests":record["long_term_interests"],"active_interests_before":record["active_before"],"coverage_before":record["coverage_before"],"need_before":record["need_before"],"complete_route_table":[{**x,"C_INCLUDED_IN_SCORE":False} for x in plan["route_table"]],"selected_route":row,"selected_start_interest":None if row is None else row["interest"],"selected_target_attribute":None if row is None else row["target_genre"],"candidate_tier_counts":dict(Counter(str(x["tier"]) for x in plan["candidates"])),"allowed_candidates":plan["candidates"],"llm_attempts":record["llm_attempts"],"llm_retry_count":record["llm_retry_count"],"invalid_output_reasons":invalid,"api_calls":api_calls,"selected_item":record.get("selected_item"),"intermediate_genres":None if not record.get("selected_item") else record["selected_item"]["genres"],"feedback":record.get("feedback"),"active_interests_after":record.get("active_after"),"newly_activated":record.get("newly_activated",[]),"newly_deactivated":record.get("deactivated",[]),"coverage_updates":record.get("coverage_updates",{}),"coverage_after":record.get("coverage_after"),"need_after":target_need(record.get("coverage_after",record["coverage_before"])),"route_switched":record.get("route_switched"),"status":record["status"]})
  if planner.status not in ("ready","running"):break
 routes=[(x["selected_start_interest"],x["selected_target_attribute"]) for x in trajectory if x["selected_route"] and x["selected_item"]]
 attrs=[x[1] for x in routes];starts=[x[0] for x in routes]
 activated=[]
 for x in trajectory:activated += x["newly_activated"]
 wins=0
 for idx,x in enumerate(trajectory):
  prior={g for prev in trajectory[:idx] for g in prev["newly_activated"]}
  if x["selected_start_interest"] in prior:wins+=1
 final=[x["title"] for x in planner.accepted]+[target["title"]]
 return {"method":"MIMAR-v0.2","protocol":"OFFLINE-POSITIVE","user_id":user["user_id"],"path_index":path_id,"target":target,"long_term_interests":planner.long_term,"initial_active_interests":initial_active,"trajectory":trajectory,"selected_route_sequence":[list(x) for x in routes],"selected_start_interest_sequence":starts,"selected_target_attribute_sequence":attrs,"coverage_sequence":[x["coverage_before"] for x in trajectory]+([trajectory[-1]["coverage_after"]] if trajectory else [planner.coverage]),"need_sequence":[x["need_before"] for x in trajectory]+([trajectory[-1]["need_after"]] if trajectory else [target_need(planner.coverage)]),"accepted_intermediates":planner.accepted,"parsed_path":final,"parse_success":True,"target_appended_by_normalization":True,"stop_reason":planner.status,"path_length":len(final),"route_switch_count":sum(a!=b for a,b in zip(routes,routes[1:])),"target_attribute_switch_count":sum(a!=b for a,b in zip(attrs,attrs[1:])),"distinct_route_count":len(set(routes)),"distinct_start_interest_count":len(set(starts)),"distinct_target_attribute_count":len(set(attrs)),"new_interest_activation_count":len(activated),"new_interest_route_win_count":wins,"coverage_stop_triggered":coverage_complete(planner.coverage),"invalid_llm_outputs":sum(len(x["invalid_output_reasons"]) for x in trajectory),"llm_retry_events":sum(x["llm_retry_count"]>0 for x in trajectory),"total_llm_generations":calls,"route_penalties_remained_zero":not planner.memory.route_penalties,"sasrec_used_during_planning":False}

def main():
 if OUT.exists():raise FileExistsError(f"Refusing overwrite: {OUT}")
 before=protected_hashes();manifest=read_json(MANIFEST);pools=read_json(POOLS);config=read_json(CONFIG);catalog=load_movies(MOVIES)
 users=[next(x for x in manifest["users"] if x["user_id"]==uid) for uid in USERS];OUT.mkdir(parents=True);records=[]
 save_new(OUT/"protocol.json",{"method":"MIMAR-v0.2","protocol":"OFFLINE-POSITIVE PRE-STUDY","users":USERS,"paths_per_user":2,"max_intermediate_steps":MAX_INTERMEDIATE_STEPS,"all_valid_intermediates_receive":"ACCEPT","synthetic_rejection":False,"candidate_source":"same frozen 100-item controlled pools","llm_config":config,"sasrec_during_planning":False})
 for user in users:
  for path_id in (1,2):
   started=time.perf_counter();r=generate(user,pools[str(user["user_id"])]["items"],catalog,config,path_id);r["elapsed_seconds"]=time.perf_counter()-started
   save_new(OUT/"generation"/str(user["user_id"])/f"path_{path_id}.json",r);records.append(r);print(f"generated {len(records)}/10 user={user['user_id']} path={path_id}",flush=True)
 from repro.experiments.dynamic_no_bridge.run_controlled_5user import aggregate,evaluate
 evaluation=evaluate(records,users,{uid:x["items"] for uid,x in pools.items()},"MIMAR-v0.2")
 metrics=aggregate(evaluation);save_new(OUT/"evaluation/path_level_metrics.json",evaluation);save_new(OUT/"evaluation/method_metrics.json",metrics)
 rows=[]
 for r in records:
  ev=next(x for x in evaluation if x["user_id"]==r["user_id"] and x["path_index"]==r["path_index"])
  rows.append({"user_id":r["user_id"],"path_id":r["path_index"],"exact_path":r["parsed_path"],"route_sequence":r["selected_route_sequence"],"target_attribute_sequence":r["selected_target_attribute_sequence"],"coverage_sequence":r["coverage_sequence"],"IoI":ev.get("IoI"),"IoR":ev.get("IoR"),"formal_metric_status":ev["formal_metric_status"]})
 save_new(OUT/"evaluation/path_diagnostics.json",rows)
 valid=sum(x["formal_metric_status"]=="valid" for x in evaluation);successful=sum(bool(x["accepted_intermediates"]) for x in records);target_only=sum(not x["accepted_intermediates"] for x in records)
 mechanism={"mean_path_length":mean(x["path_length"] for x in records),"mean_route_switch_count":mean(x["route_switch_count"] for x in records),"mean_target_attribute_switch_count":mean(x["target_attribute_switch_count"] for x in records),"mean_distinct_route_count":mean(x["distinct_route_count"] for x in records),"mean_distinct_start_interest_count":mean(x["distinct_start_interest_count"] for x in records),"mean_distinct_target_attribute_count":mean(x["distinct_target_attribute_count"] for x in records),"total_new_interest_activations":sum(x["new_interest_activation_count"] for x in records),"total_new_interest_route_wins":sum(x["new_interest_route_win_count"] for x in records),"coverage_stop_rate":mean(x["coverage_stop_triggered"] for x in records),"invalid_llm_outputs":sum(x["invalid_llm_outputs"] for x in records),"llm_retry_events":sum(x["llm_retry_events"] for x in records),"all_route_penalties_zero":all(x["route_penalties_remained_zero"] for x in records)}
 after=protected_hashes();changed=sorted(k for k in before if before[k]!=after[k])
 summary={"complete":valid==10 and successful==10 and not changed,"evaluator_valid_paths":valid,"generation_successful_paths":successful,"target_only_paths":target_only,"mimar_v02":metrics,"mechanism":mechanism,"path_diagnostics":rows,"v01_reference":{"IoI":1.882053,"IoR":191.6,"TOTAL_NEW_INTEREST_ACTIVATIONS":13,"NEW_INTEREST_ACTIVATIONS_FOLLOWED_BY_ROUTE_SWITCH":0,"MEAN_ROUTE_SWITCH_COUNT":0.2,"MEAN_DISTINCT_ROUTE_COUNT":1.0,"MEAN_DISTINCT_START_INTEREST_COUNT":1.0,"MEAN_DISTINCT_TARGET_ATTRIBUTE_COUNT":0.8},"comparison_is_descriptive":True,"protection":{"changed_files":changed,"formal_evaluator_modified":False,"frozen_methods_modified":False,"mimar_v02_formulas_modified":False},"completed_at":datetime.now(timezone.utc).isoformat()}
 save_new(OUT/"summary.json",summary);save_new(OUT/"run_metadata.json",{"protected_before":before,"protected_after":after,"generation_completed_before_evaluator_load":True,"new_llm_paths":10,"formal_protocol":"Formal-Evaluation-v1"})
 if changed:raise RuntimeError(f"Protected files changed: {changed}")
 print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()

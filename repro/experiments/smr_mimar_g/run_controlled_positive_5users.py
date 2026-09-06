"""Five-user controlled OFFLINE-POSITIVE experiment for SMR-MIMAR-G."""
from __future__ import annotations
import hashlib,json,time
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path

from repro.methods.mi_bridge.retrieve_bridge_movies import load_movies
from repro.methods.smr_mimar_g.planner import SMRMIMARGPlanner
from repro.methods.smr_mimar_v01.planner import MAX_INTERMEDIATE_STEPS
from repro.experiments.mimar_v02.run_controlled_positive_5users import read_json,save_new,llm_select,normalize,mean

ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/"repro/results/smr_mimar_g/controlled_positive_5users";MANIFEST=ROOT/"repro/results/pilot/pilot_manifest.json";POOLS=ROOT/"repro/results/pilot/catalog_grounded_ablation_v1/candidate_pools.json";MOVIES=ROOT/"dataset/ml-1m/movies.dat";CONFIG=ROOT/"repro/local_config.json";USERS=[419,5021,2677,3113,2249]
PROTECTED=[ROOT/"repro/results/smr_mimar_v01/controlled_positive_5users",ROOT/"repro/results/mimar_v01/controlled_positive_5users",ROOT/"repro/results/mimar_v02/controlled_positive_5users",ROOT/"repro/results/mimar_v03/controlled_positive_5users",ROOT/"repro/results/pilot/catalog_grounded_ablation_v1"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot():
 fs=[]
 for d in (ROOT/"repro/methods",ROOT/"repro/evaluators",*PROTECTED):fs += [p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
 return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(fs))}

def generate(user,pool,catalog,config,path_id):
 by={x["id"]:x for x in catalog};history=[deepcopy(by[int(i)]) for i in user["positive_movie_ids"]];target=deepcopy(by[int(user["target"]["movie_id"])])
 planner=SMRMIMARGPlanner(history,target,catalog,user["demographics"],candidate_catalog=[normalize(x) for x in pool]);trajectory=[];calls=0
 while planner.status in ("ready","running") and len(planner.accepted)<MAX_INTERMEDIATE_STEPS:
  api=[]
  def choose(prompt):
   nonlocal calls
   raw,meta=llm_select(prompt,config);calls+=1;api.append({"prompt":deepcopy(prompt),"raw_output":raw,"api":meta});return raw
  r=planner.step(choose)
  if "plan" not in r:break
  trajectory.append({"step":r["step"],"fixed_route_set":r["fixed_route_set"],"current_generated_path":r["plan"]["current_generated_path"],"legal_candidate_count":len(r["plan"]["candidates"]),"legal_candidates":r["plan"]["candidates"],"prompt":r["prompt"],"llm_attempts":r["llm_attempts"],"llm_retry_count":r["llm_retry_count"],"parser_result":r["parser_result"],"selected_item":r.get("selected_item"),"selected_item_accepted":r.get("selected_item_accepted"),"target_overlap_guard":r.get("target_overlap_guard"),"rejected_for_path_item":r.get("rejected_for_path_item"),"api_calls":api,"status":r["status"]})
  if planner.status not in ("ready","running"):break
 selected=planner.accepted;counts=[x["route_support_count"] for x in selected];guard=next((x["target_overlap_guard"] for x in trajectory if (x.get("target_overlap_guard") or {}).get("guard_triggered")),None)
 return {"method":"SMR-MIMAR-G","protocol":"OFFLINE-POSITIVE","user_id":user["user_id"],"path_index":path_id,"target":target,"long_term_interest_profile":planner.long_term_profile,"top_interests":planner.top_interests,"all_feasible_routes":planner.all_feasible_routes,"selected_static_route_set":planner.route_set,"trajectory":trajectory,"accepted_intermediates":selected,"parsed_path":[x["title"] for x in selected]+[target["title"]],"parse_success":True,"target_appended_by_normalization":True,"stop_reason":planner.status,"path_length":len(selected)+1,"static_route_set_size":len(planner.route_set),"distinct_start_interest_count":len({x["interest"] for x in planner.route_set}),"distinct_target_attribute_count":len({x["target_genre"] for x in planner.route_set}),"selected_items_supporting_multiple_routes":sum(x>=2 for x in counts),"mean_selected_item_route_support_count":mean(counts),"guard_triggered":guard is not None,"guard_trigger_step":None if guard is None else guard["step"],"guard_trigger_detail":guard,"invalid_llm_outputs":sum(sum(not a["parse_result"]["success"] for a in t["llm_attempts"]) for t in trajectory),"llm_retry_events":sum(t["llm_retry_count"]>0 for t in trajectory),"total_llm_generations":calls,"generation_successful":bool(selected),"early_stop":len(selected)<MAX_INTERMEDIATE_STEPS,"sasrec_used_during_planning":False}

def prefix_audit(records,users):
 from repro.evaluators.formal.metrics import interest_increase
 from repro.evaluators.sasrec.checkpoint_loader import load_checkpoint
 from repro.evaluators.sasrec.data_adapter import DataAdapter
 from repro.evaluators.sasrec.evaluator import SASRecEvaluator
 model,dataset,_,_=load_checkpoint();ev=SASRecEvaluator(model);adapter=DataAdapter(dataset);by={x["user_id"]:x for x in users};rows=[]
 for r in records:
  u=by[r["user_id"]];h=adapter.history_raw_ids_to_internal([x["movie_id"] for x in u["history"]]);target=adapter.raw_item_id_to_internal(u["target"]["movie_id"]);p0=ev.get_target_score(h,target);rank0=ev.get_target_rank(h,target);current=list(h);prefix=[]
  for step,item in enumerate(r["accepted_intermediates"],1):
   current.append(adapter.raw_item_id_to_internal(item["id"]));p=ev.get_target_score(current,target);rank=ev.get_target_rank(current,target);prefix.append({"step":step,"target_rank":rank,"target_probability":p,"prefix_IoR":rank0-rank,"prefix_IoI":interest_increase(p0,p)})
  if prefix:
   best=max(prefix,key=lambda x:x["prefix_IoR"]);final=prefix[-1];rows.append({"user_id":r["user_id"],"path_id":r["path_index"],"prefix_table":prefix,"maximum_prefix_IoR":best["prefix_IoR"],"maximum_prefix_IoR_step":best["step"],"final_intermediate_IoR":final["prefix_IoR"],"IoR_regret":best["prefix_IoR"]-final["prefix_IoR"],"best_prefix_before_final":best["step"]<len(prefix)})
 return {"paths":rows,"mean_max_prefix_IoR":mean(x["maximum_prefix_IoR"] for x in rows),"mean_final_intermediate_IoR":mean(x["final_intermediate_IoR"] for x in rows),"mean_IoR_regret":mean(x["IoR_regret"] for x in rows),"best_prefix_before_final_count":sum(x["best_prefix_before_final"] for x in rows)}

def main():
 if (OUT/"summary.json").exists():raise FileExistsError(f"Refusing overwrite completed result: {OUT}")
 before=snapshot();manifest=read_json(MANIFEST);pools=read_json(POOLS);config=read_json(CONFIG);catalog=load_movies(MOVIES);users=[next(x for x in manifest["users"] if x["user_id"]==uid) for uid in USERS];OUT.mkdir(parents=True,exist_ok=True);records=[]
 if not (OUT/"protocol.json").exists():save_new(OUT/"protocol.json",{"method":"SMR-MIMAR-G","users":USERS,"paths_per_user":2,"max_intermediate_steps":6,"candidate_source":"same frozen controlled 100-item pools","all_valid_accepted_intermediates":"ACCEPT","guard":"stop before adding candidate when new TargetOverlap < previous accepted TargetOverlap","no_guard_retry":True,"formal_evaluator_after_generation_only":True,"llm_config":config})
 for u in users:
  for pid in (1,2):
   path=OUT/"generation"/str(u["user_id"])/f"path_{pid}.json"
   if path.exists():r=read_json(path);label="preserved"
   else:
    started=time.perf_counter();r=generate(u,pools[str(u["user_id"])]["items"],catalog,config,pid);r["elapsed_seconds"]=time.perf_counter()-started;save_new(path,r);label="generated"
   records.append(r);print(f"{label} {len(records)}/10 user={u['user_id']} path={pid} items={len(r['accepted_intermediates'])} guard={r['guard_triggered']}",flush=True)
 if (OUT/"evaluation/path_level_metrics.json").exists():
  evaluation=read_json(OUT/"evaluation/path_level_metrics.json");metrics=read_json(OUT/"evaluation/method_metrics.json");prefix=read_json(OUT/"prefix_audit.json")
 else:
  from repro.experiments.dynamic_no_bridge.run_controlled_5user import aggregate,evaluate
  evaluation=evaluate(records,users,{uid:x["items"] for uid,x in pools.items()},"SMR-MIMAR-G");metrics=aggregate(evaluation);save_new(OUT/"evaluation/path_level_metrics.json",evaluation);save_new(OUT/"evaluation/method_metrics.json",metrics);prefix=prefix_audit(records,users);save_new(OUT/"prefix_audit.json",prefix)
 valid=sum(x["formal_metric_status"]=="valid" for x in evaluation);success=sum(x["generation_successful"] for x in records);target_only=10-success;early=sum(x["early_stop"] for x in records);guards=[x for x in records if x["guard_triggered"]]
 guard_events=[{"user_id":x["user_id"],"path_id":x["path_index"],"step":x["guard_trigger_step"],"previous_accepted_item":x["accepted_intermediates"][-1] if x["accepted_intermediates"] else None,"previous_overlap":x["guard_trigger_detail"]["previous_target_overlap"],"rejected_for_path_item":x["guard_trigger_detail"]["rejected_for_path_item"],"new_overlap":x["guard_trigger_detail"]["new_target_overlap"],"overlap_delta":x["guard_trigger_detail"]["overlap_delta"],"final_accepted_intermediate_sequence":[i["title"] for i in x["accepted_intermediates"]]} for x in guards]
 if not (OUT/"guard_events.json").exists():save_new(OUT/"guard_events.json",guard_events)
 user_rows=[]
 for uid in USERS:
  rr=[x for x in records if x["user_id"]==uid];ee=[x for x in evaluation if x["user_id"]==uid];pp=[x for x in prefix["paths"] if x["user_id"]==uid];user_rows.append({"user_id":uid,"mean_IoI":mean(x["IoI"] for x in ee),"mean_IoR":mean(x["IoR"] for x in ee),"path_lengths":[x["path_length"] for x in rr],"guard_triggered":[x["guard_triggered"] for x in rr],"trigger_steps":[x["guard_trigger_step"] for x in rr],"mean_max_prefix_IoR":mean(x["maximum_prefix_IoR"] for x in pp),"mean_final_IoR_regret":mean(x["IoR_regret"] for x in pp),"exact_paths":[x["parsed_path"] for x in rr]})
 user_path=OUT/"evaluation/user_level_results.json"
 save_new(user_path if not user_path.exists() else OUT/"evaluation/user_level_results_complete.json",user_rows)
 after=snapshot();changed=sorted(k for k in before if before[k]!=after[k]);summary={"complete":valid==10 and success==10 and not changed,"evaluator_valid_paths":valid,"generation_successful_paths":success,"target_only_paths":target_only,"early_stop_paths":early,"guard_triggered_paths":len(guards),"metrics":metrics,"mean_path_length":mean(x["path_length"] for x in records),"prefix":{k:v for k,v in prefix.items() if k!="paths"},"guard_trigger_mean_step":mean(x["guard_trigger_step"] for x in guards),"invalid_llm_outputs":sum(x["invalid_llm_outputs"] for x in records),"llm_retry_events":sum(x["llm_retry_events"] for x in records),"original_smr_reference":{"IoI":1.349992,"IoR":-166.9,"mean_max_prefix_IoR":692.6,"mean_IoR_regret":859.5},"comparison_is_descriptive":True,"user_level":user_rows,"protection":{"changed_files":changed,"formal_evaluator_modified":False,"frozen_methods_modified":False,"smr_mimar_g_modified":False},"completed_at":datetime.now(timezone.utc).isoformat()};save_new(OUT/"summary.json",summary);save_new(OUT/"run_metadata.json",{"protected_before":before,"protected_after":after,"generation_completed_before_evaluator_load":True,"formal_protocol":"Formal-Evaluation-v1"});print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()

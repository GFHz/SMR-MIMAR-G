"""MIMAR-v0.2 planner; no evaluator dependency."""
from copy import deepcopy

from repro.methods.mimar_v01.candidate_builder import build_candidates
from repro.methods.mimar_v01.cooccurrence import GenreCooccurrence
from repro.methods.mimar_v01.feedback import ACCEPT,apply_feedback
from repro.methods.mimar_v01.interest_profile import ACTIVATION_THRESHOLD,initial_active_interests,long_term_interests
from repro.methods.mimar_v01.route_state import FeedbackMemory
from .coverage import coverage_complete,initial_coverage,target_need,update_coverage
from .prompt import MAX_LLM_RETRIES,build_prompt,correction_prompt,parse_one_title
from .route_scoring import route_table

METHOD="MIMAR-v0.2"; MAX_INTERMEDIATE_STEPS=8; MAX_PLANNING_STEPS=24


class MIMARV02Planner:
    def __init__(self,full_positive_history,target,movies,demographics=None,candidate_catalog=None,
                 user_id=None,path_id=None):
        self.history=deepcopy(list(full_positive_history)); self.target=deepcopy(target)
        self.movies=deepcopy(list(movies)); self.candidate_catalog=deepcopy(list(candidate_catalog or movies))
        self.demographics=deepcopy(demographics or {}); self.long_term=long_term_interests(self.history)
        self.user_id=user_id; self.path_id=path_id
        self.active=initial_active_interests(self.history); self.coverage=initial_coverage(target["genres"])
        self.cooccurrence=GenreCooccurrence(self.movies); self.memory=FeedbackMemory()
        self.original_history_ids={int(x["id"]) for x in self.history}; self.used_ids=set()
        self.accepted=[]; self.previous_route=None; self.steps=[]; self.status="ready"

    def plan(self):
        table=route_table(self.long_term,self.active,self.target["genres"],self.coverage,
                          self.cooccurrence,self.memory,self.previous_route)
        feasible=[row for row in table if row["feasible"]]
        for row in feasible:
            route=(row["interest"],row["target_genre"])
            candidates=build_candidates(self.candidate_catalog,route,self.active,self.cooccurrence,
                self.original_history_ids,self.used_ids,self.target["id"],self.memory)
            if candidates:return {"route_table":table,"feasible_routes":feasible,"selected_route":row,"candidates":candidates}
        return {"route_table":table,"feasible_routes":feasible,"selected_route":None,"candidates":[],
                "stop_detail":"NO_FEASIBLE_ROUTE" if not feasible else "NO_LEGAL_CANDIDATE"}

    def step(self,select_item,feedback_provider):
        if self.status not in ("ready","running"):raise RuntimeError("Planner already stopped")
        if coverage_complete(self.coverage):self.status="target_coverage_complete";return {"status":self.status}
        if len(self.accepted)>=MAX_INTERMEDIATE_STEPS or len(self.steps)>=MAX_PLANNING_STEPS:
            self.status="safety_cap";return {"status":self.status}
        plan=self.plan()
        if plan["selected_route"] is None:
            self.status=plan["stop_detail"].lower();return {"status":self.status,"plan":plan}
        route_row=plan["selected_route"]; route=(route_row["interest"],route_row["target_genre"])
        prompt=build_prompt(self.demographics,self.long_term,self.active,self.target,self.coverage,
                            target_need(self.coverage),route_row,plan["candidates"])
        attempts=[]; parsed=None; raw=None
        for index in range(MAX_LLM_RETRIES+1):
            attempt_prompt=prompt if index==0 else correction_prompt(prompt)
            raw=select_item(deepcopy(attempt_prompt)); parsed=parse_one_title(raw,prompt["allowed_titles"])
            attempts.append({"attempt":index+1,"prompt":deepcopy(attempt_prompt),"raw_output":raw,"parse_result":deepcopy(parsed)})
            if parsed["success"]:break
        record={"user_id":self.user_id,"path_id":self.path_id,"step":len(self.steps)+1,
            "long_term_interests":deepcopy(self.long_term),"target_genres":deepcopy(self.target["genres"]),
            "plan":deepcopy(plan),"coverage_before":deepcopy(self.coverage),"need_before":target_need(self.coverage),
            "active_before":deepcopy(self.active),"previous_route":self.previous_route,"prompt":prompt,
            "llm_attempts":attempts,"llm_retry_count":len(attempts)-1,"raw":raw,"parsed":parsed}
        if not parsed["success"]:
            self.status="invalid_llm_selection_after_retries";record["status"]=self.status;self.steps.append(record);return deepcopy(record)
        item=next(x for x in plan["candidates"] if x["title"]==parsed["title"])
        outcome=feedback_provider(deepcopy(item),deepcopy(record)); old_active={g for g,v in self.active.items() if v>=ACTIVATION_THRESHOLD}
        self.active=apply_feedback(self.active,self.memory,route,item,outcome)
        coverage_changes={g:0.0 for g in self.coverage}
        if outcome==ACCEPT:
            self.coverage,coverage_changes=update_coverage(self.coverage,item["genres"])
            self.accepted.append(deepcopy(item));self.used_ids.add(int(item["id"]))
        new_active={g for g,v in self.active.items() if v>=ACTIVATION_THRESHOLD}
        switched=self.previous_route is not None and route!=self.previous_route;self.previous_route=route
        self.status="target_coverage_complete" if coverage_complete(self.coverage) else (
            "safety_cap" if len(self.accepted)>=MAX_INTERMEDIATE_STEPS else "running")
        record.update({"selected_item":deepcopy(item),"feedback":outcome,"active_after":deepcopy(self.active),
            "newly_activated":sorted(new_active-old_active),"deactivated":sorted(old_active-new_active),
            "coverage_after":deepcopy(self.coverage),"coverage_updates":coverage_changes,
            "route_switched":switched,"status":self.status})
        self.steps.append(record);return deepcopy(record)

    def snapshot(self):
        return {"method":METHOD,"status":self.status,"long_term_interests":deepcopy(self.long_term),
            "active_interests":deepcopy(self.active),"target_coverage":deepcopy(self.coverage),
            "accepted_intermediates":deepcopy(self.accepted),"final_path":deepcopy(self.accepted)+[deepcopy(self.target)],
            "steps":deepcopy(self.steps),"evaluator_used_during_planning":False}

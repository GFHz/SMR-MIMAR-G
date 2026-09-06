"""Static-route planner with no dynamic interest, coverage, or switch state."""
from copy import deepcopy
from repro.methods.mimar_v01.cooccurrence import GenreCooccurrence
from .candidates import legal_candidates
from .prompt import MAX_LLM_RETRIES,build_prompt,correction_prompt,parse_one_title
from .routes import feasible_routes,ranked_long_term_interests,select_diverse_routes

METHOD="SMR-MIMAR-v0.1";MAX_INTERMEDIATE_STEPS=6
class SMRMIMARPlanner:
    def __init__(self,full_positive_history,target,catalog,demographics=None,candidate_catalog=None):
        self.history=deepcopy(list(full_positive_history));self.target=deepcopy(target);self.catalog=deepcopy(list(catalog));self.candidate_catalog=deepcopy(list(candidate_catalog or catalog));self.demographics=deepcopy(demographics or {})
        self.top_interests,self.long_term_profile=ranked_long_term_interests(self.history);self.cooccurrence=GenreCooccurrence(self.catalog)
        self.all_feasible_routes=feasible_routes(self.top_interests,self.target["genres"],self.cooccurrence);self.route_set=select_diverse_routes(self.all_feasible_routes)
        self.original_history_ids={int(x["id"]) for x in self.history};self.used_ids=set();self.accepted=[];self.steps=[];self.status="ready"
    def plan(self):
        candidates=legal_candidates(self.candidate_catalog,self.route_set,self.original_history_ids,self.used_ids,self.target["id"])
        return {"fixed_route_set":deepcopy(self.route_set),"current_generated_path":[x["title"] for x in self.accepted],"candidates":candidates}
    def step(self,select_item):
        if self.status not in ("ready","running"):raise RuntimeError("Planner stopped")
        if len(self.accepted)>=MAX_INTERMEDIATE_STEPS:self.status="max_intermediate_steps";return {"status":self.status}
        plan=self.plan()
        if not plan["candidates"]:self.status="no_legal_candidate";return {"status":self.status,"plan":plan}
        prompt=build_prompt(self.demographics,self.top_interests,self.target,self.route_set,plan["current_generated_path"],plan["candidates"]);attempts=[];parsed=None
        for index in range(MAX_LLM_RETRIES+1):
            sent=prompt if index==0 else correction_prompt(prompt);raw=select_item(deepcopy(sent));parsed=parse_one_title(raw,prompt["allowed_titles"]);attempts.append({"attempt":index+1,"prompt":sent,"raw_output":raw,"parse_result":parsed})
            if parsed["success"]:break
        record={"step":len(self.steps)+1,"fixed_route_set":deepcopy(self.route_set),"plan":plan,"prompt":prompt,"llm_attempts":attempts,"llm_retry_count":len(attempts)-1,"parser_result":parsed}
        if not parsed["success"]:self.status="invalid_llm_selection_after_retries";record["status"]=self.status;self.steps.append(record);return deepcopy(record)
        item=next(x for x in plan["candidates"] if x["title"]==parsed["title"]);self.accepted.append(deepcopy(item));self.used_ids.add(int(item["id"]));self.status="max_intermediate_steps" if len(self.accepted)>=MAX_INTERMEDIATE_STEPS else "running";record.update(selected_item=deepcopy(item),status=self.status);self.steps.append(record);return deepcopy(record)
    def snapshot(self):return {"method":METHOD,"status":self.status,"long_term_interests":deepcopy(self.top_interests),"all_feasible_routes":deepcopy(self.all_feasible_routes),"selected_static_route_set":deepcopy(self.route_set),"accepted_intermediates":deepcopy(self.accepted),"final_path":deepcopy(self.accepted)+[deepcopy(self.target)],"steps":deepcopy(self.steps),"sasrec_used_during_planning":False}

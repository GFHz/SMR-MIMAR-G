"""Minimal post-selection guard around frozen SMR-MIMAR-v0.1 planning."""
from copy import deepcopy
from repro.methods.smr_mimar_v01.planner import SMRMIMARPlanner
from .guard import overlap_guard,target_overlap

METHOD="SMR-MIMAR-G"
class SMRMIMARGPlanner(SMRMIMARPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.last_accepted_overlap=None;self.guard_log=[];self.rejected_for_path_item=None
    def step(self,select_item):
        record=super().step(select_item)
        item=record.get("selected_item")
        if item is None:return record
        new=target_overlap(item["genres"],self.target["genres"]);decision=overlap_guard(self.last_accepted_overlap,new)
        guard={"step":record["step"],"title":item["title"],"genres":deepcopy(item["genres"]),"target_overlap":new,
               "previous_target_overlap":self.last_accepted_overlap,"new_target_overlap":new,
               "overlap_delta":decision["overlap_delta"],"guard_triggered":decision["guard_triggered"],
               "first_item_bypass":decision["first_item_bypass"]}
        if decision["guard_triggered"]:
            removed=self.accepted.pop();self.used_ids.remove(int(removed["id"]));self.rejected_for_path_item=deepcopy(removed)
            self.status="target_overlap_decrease";guard["rejected_for_path_item"]=deepcopy(removed)
            guard["final_accepted_intermediate_sequence"]=[x["title"] for x in self.accepted];guard["stop_reason"]="TARGET_OVERLAP_DECREASE"
            record["selected_item_accepted"]=False;record["rejected_for_path_item"]=deepcopy(removed);record["status"]=self.status
        else:
            self.last_accepted_overlap=new;record["selected_item_accepted"]=True
        self.guard_log.append(deepcopy(guard));record["target_overlap_guard"]=guard
        return record
    def snapshot(self):
        result=super().snapshot();result.update(method=METHOD,target_overlap_guard_log=deepcopy(self.guard_log),rejected_for_path_item=deepcopy(self.rejected_for_path_item));return result

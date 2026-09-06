"""Pure metadata-only target-overlap guard."""
def target_overlap(item_genres,target_genres):
    target=set(target_genres)
    if not target:raise ValueError("Target genres must be non-empty")
    return len(set(item_genres)&target)/len(target)

def overlap_guard(previous_overlap,new_overlap):
    if previous_overlap is None:return {"guard_triggered":False,"overlap_delta":None,"first_item_bypass":True}
    delta=new_overlap-previous_overlap
    return {"guard_triggered":delta<0,"overlap_delta":delta,"first_item_bypass":False}

"""Catalog-grounded direct support for a fixed route set."""
TOP_K_CANDIDATES=20

def supported_routes(movie,route_set):
    genres=set(movie["genres"])
    return [(r["interest"],r["target_genre"]) for r in route_set if r["interest"] in genres and r["target_genre"] in genres]

def legal_candidates(catalog,route_set,original_history_ids,used_ids,target_id,limit=TOP_K_CANDIDATES):
    excluded=set(map(int,original_history_ids))|set(map(int,used_ids))|{int(target_id)};rows=[]
    for movie in catalog:
        if int(movie["id"]) in excluded:continue
        support=supported_routes(movie,route_set)
        if support:rows.append({**movie,"supported_routes":[list(x) for x in support],"route_support_count":len(support),"support_tier":"DIRECT"})
    rows.sort(key=lambda x:int(x["id"]))
    return rows[:limit]

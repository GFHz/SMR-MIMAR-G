"""Frozen long-term profile, static route prior, and diverse route selection."""
from repro.methods.mimar_v01.interest_profile import long_term_interests
from repro.methods.mimar_v01.cooccurrence import GenreCooccurrence

ALPHA=0.35
GAMMA=0.30
TOP_K_USER_INTERESTS=5
ROUTE_SET_SIZE=4

def ranked_long_term_interests(full_positive_history,k=TOP_K_USER_INTERESTS):
    profile=long_term_interests(full_positive_history)
    rows=sorted(({"genre":g,"L":v["score"],**v} for g,v in profile.items()),key=lambda x:(-x["L"],x["genre"]))
    return rows[:k],profile

def feasible_routes(top_interests,target_genres,cooccurrence):
    rows=[]
    for interest_row in top_interests:
        i=interest_row["genre"]
        for g in sorted(set(target_genres)):
            detail=cooccurrence.detail(i,g)
            feasible=detail["count_interest"]>0 if i==g else detail["count_joint"]>0
            if feasible:
                c=detail["normalized_cooccurrence"]
                rows.append({"interest":i,"target_genre":g,"L":interest_row["L"],
                    "count_interest":detail["count_interest"],"count_target_genre":detail["count_target_genre"],
                    "count_joint":detail["count_joint"],"C":c,"feasible":True,
                    "route_prior":ALPHA*interest_row["L"]+GAMMA*c})
    return sorted(rows,key=lambda x:(-x["route_prior"],-x["L"],x["interest"],x["target_genre"]))

def select_diverse_routes(ranked_routes,size=ROUTE_SET_SIZE):
    remaining=list(ranked_routes);chosen=[];starts=set();targets=set();limit=min(size,len(remaining))
    while len(chosen)<limit:
        scored=[]
        for rank,row in enumerate(remaining):
            new_start=row["interest"] not in starts;new_target=row["target_genre"] not in targets
            novelty=int(new_start)+int(new_target)
            reason="BOTH" if novelty==2 else "NEW_START_INTEREST" if new_start else "NEW_TARGET_ATTRIBUTE" if new_target else "SCORE_FILL"
            scored.append((-novelty,rank,row,reason))
        _,_,row,reason=min(scored,key=lambda x:(x[0],x[1]))
        selected={**row,"diversity_reason":reason};chosen.append(selected)
        starts.add(row["interest"]);targets.add(row["target_genre"]);remaining.remove(row)
    return chosen

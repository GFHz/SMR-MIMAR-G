"""Feasibility-gated MIMAR-v0.2 route scoring."""
from repro.methods.mimar_v01.interest_profile import ACTIVATION_THRESHOLD, active_genres
from repro.methods.mimar_v01.route_scoring import ALPHA, BETA, DELTA_ROUTE, KAPPA

ETA = 0.30


def route_table(long_term, active, target_genres, coverage, cooccurrence, memory, previous_route=None):
    needs={genre:1.0-coverage[genre] for genre in coverage}
    rows=[]
    for interest in active_genres(active,ACTIVATION_THRESHOLD):
        for target_genre in sorted(set(target_genres)):
            route=(interest,target_genre); detail=cooccurrence.detail(*route)
            feasible=(detail["count_interest"]>0 if interest==target_genre else detail["count_joint"]>0)
            l_value=long_term.get(interest,{}).get("score",0.0); s_value=active[interest]
            penalty=memory.penalty(route); continuity=KAPPA if previous_route==route else 0.0
            score=ALPHA*l_value+BETA*s_value+ETA*needs[target_genre]-DELTA_ROUTE*penalty+continuity
            rows.append({"interest":interest,"target_genre":target_genre,
                "count_interest":detail["count_interest"],"count_target_genre":detail["count_target_genre"],
                "count_joint":detail["count_joint"],"normalized_cooccurrence":detail["normalized_cooccurrence"],
                "feasible":feasible,"long_term":l_value,"active":s_value,"need":needs[target_genre],
                "penalty":penalty,"continuity_bonus":continuity,"route_score":score})
    return sorted(rows,key=lambda x:(not x["feasible"],-x["route_score"],-x["long_term"],
                                     -x["active"],x["interest"],x["target_genre"]))


def feasible_ranked_routes(*args,**kwargs):
    return [row for row in route_table(*args,**kwargs) if row["feasible"]]

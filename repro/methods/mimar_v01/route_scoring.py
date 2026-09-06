"""Migration-route construction and deterministic ranking."""
from .interest_profile import ACTIVATION_THRESHOLD, active_genres

ALPHA = 0.35
BETA = 0.35
GAMMA = 0.30
DELTA_ROUTE = 0.50
KAPPA = 0.02


def rank_routes(long_term, active, target_genres, cooccurrence, memory, previous_route=None):
    rows = []
    for interest in active_genres(active, ACTIVATION_THRESHOLD):
        for target_genre in sorted(set(target_genres)):
            route = (interest, target_genre)
            l_value = long_term.get(interest, {}).get("score", 0.0)
            s_value = active[interest]
            c_value = cooccurrence.score(*route)
            penalty = memory.penalty(route)
            continuity = KAPPA if previous_route == route else 0.0
            score = ALPHA*l_value + BETA*s_value + GAMMA*c_value - DELTA_ROUTE*penalty + continuity
            rows.append({"interest": interest, "target_genre": target_genre,
                         "long_term": l_value, "active": s_value,
                         "cooccurrence": c_value, "penalty": penalty,
                         "continuity_bonus": continuity, "route_score": score,
                         "cooccurrence_detail": cooccurrence.detail(*route)})
    return sorted(rows, key=lambda x: (-x["route_score"], -x["long_term"],
                                       -x["active"], x["interest"], x["target_genre"]))

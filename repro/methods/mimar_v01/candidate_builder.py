"""Catalog-grounded tiered candidates without evaluator information."""
TOP_K_CANDIDATES = 20


def build_candidates(movies, route, active, cooccurrence, original_history_ids,
                     used_path_ids, target_id, memory, limit=TOP_K_CANDIDATES):
    interest, target_genre = route
    active_set = {g for g, value in active.items() if value >= 0.05}
    excluded = set(original_history_ids) | set(used_path_ids) | {int(target_id)}
    tiered = []
    for movie in movies:
        item_id, genres = int(movie["id"]), set(movie["genres"])
        if item_id in excluded or memory.cooling_down(item_id):
            continue
        tier = None
        support = 0.0
        if interest in genres and target_genre in genres:
            tier, support = 1, 1.0
        elif interest in genres:
            links = [cooccurrence.score(g, target_genre) for g in genres if g != interest]
            if links and max(links) > 0:
                tier, support = 2, max(links)
        elif target_genre in genres and genres.intersection(active_set):
            tier, support = 3, max(active[g] for g in genres.intersection(active_set))
        if tier is not None:
            tiered.append({**movie, "tier": tier, "support": support})
    tiered.sort(key=lambda x: (x["tier"], -x["support"], int(x["id"])))
    return tiered[:limit]

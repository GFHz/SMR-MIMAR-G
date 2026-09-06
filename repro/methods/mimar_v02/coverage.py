"""Count-based multi-attribute target coverage."""
TARGET_COVERAGE_INCREMENT = 0.5
TARGET_COVERAGE_STOP_THRESHOLD = 1.0


def initial_coverage(target_genres):
    return {genre: 0.0 for genre in sorted(set(target_genres))}


def target_need(coverage):
    return {genre: 1.0-value for genre,value in coverage.items()}


def update_coverage(coverage, accepted_item_genres):
    genres=set(accepted_item_genres); updated=dict(coverage); changes={}
    for genre,value in coverage.items():
        new=min(1.0,value+TARGET_COVERAGE_INCREMENT) if genre in genres else value
        updated[genre]=new; changes[genre]=new-value
    return updated,changes


def coverage_complete(coverage):
    return bool(coverage) and all(value>=TARGET_COVERAGE_STOP_THRESHOLD for value in coverage.values())

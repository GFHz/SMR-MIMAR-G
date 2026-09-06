"""Deterministic long-term memory and bounded short-term active interests."""
from collections import Counter

LONG_TERM_WEIGHTS = {"frequency": 0.5, "persistence": 0.3, "recency": 0.2}
RECENT_WINDOW = 20
ACTIVE_DECAY = 0.95
ACCEPT_BOOST = 0.15
REJECT_GENRE_DROP = 0.05
ACTIVATION_THRESHOLD = 0.05


def _clip(value):
    return max(0.0, min(1.0, float(value)))


def long_term_interests(positive_history):
    """L(i)=.5 frequency+.3 temporal persistence+.2 mean normalized recency.

    Persistence is the fraction of four chronological bins containing i.
    Recency is the mean normalized 1-based position of occurrences.
    """
    items = list(positive_history)
    if not items:
        raise ValueError("Full positive history must be non-empty")
    n = len(items)
    genres = sorted({g for item in items for g in item["genres"]})
    result = {}
    for genre in genres:
        positions = [idx + 1 for idx, item in enumerate(items) if genre in item["genres"]]
        frequency = len(positions) / n
        bins = {min(3, (pos - 1) * 4 // n) for pos in positions}
        persistence = len(bins) / 4
        recency = sum(pos / n for pos in positions) / len(positions)
        score = (LONG_TERM_WEIGHTS["frequency"] * frequency
                 + LONG_TERM_WEIGHTS["persistence"] * persistence
                 + LONG_TERM_WEIGHTS["recency"] * recency)
        result[genre] = {"score": _clip(score), "frequency": frequency,
                         "persistence": persistence, "recency": recency,
                         "count": len(positions)}
    return result


def initial_active_interests(positive_history, window_size=RECENT_WINDOW):
    recent = list(positive_history)[-window_size:]
    if not recent:
        raise ValueError("Recent interaction state must be non-empty")
    counts = Counter(g for item in recent for g in item["genres"])
    denominator = len(recent)
    return {genre: count / denominator for genre, count in sorted(counts.items())}


def update_active_interests(active, item_genres, accepted):
    updated = {genre: _clip(value * ACTIVE_DECAY) for genre, value in active.items()}
    change = ACCEPT_BOOST if accepted else -REJECT_GENRE_DROP
    for genre in item_genres:
        updated[genre] = _clip(updated.get(genre, 0.0) + change)
    return dict(sorted(updated.items()))


def active_genres(active, threshold=ACTIVATION_THRESHOLD):
    return sorted(genre for genre, value in active.items() if value >= threshold)

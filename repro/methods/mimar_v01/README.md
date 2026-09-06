# MIMAR-v0.1

Multi-Interest Multi-Attribute Feedback-Aware Route Planning is a new, evaluator-independent method. It does not modify or import the frozen Static/Dynamic MI-Bridge methods or Formal-Evaluation-v1.

## Interest state

For a full chronological positive history of length `N`:

`L(i) = 0.5 Frequency(i) + 0.3 Persistence(i) + 0.2 Recency(i)`.

- `Frequency(i)`: occurrence count divided by `N`.
- `Persistence(i)`: fraction of four chronological bins containing genre `i`.
- `Recency(i)`: mean normalized 1-based position of occurrences.

`S_0(i)` is genre frequency in the latest 20 positive interactions. After feedback all active strengths are multiplied by `0.95`. Acceptance adds `0.15` to each selected-item genre; rejection subtracts `0.05`. Values are clipped to `[0,1]`; activation threshold is `0.05`. Deactivation never removes `L(i)`.

## Target and route

Every MovieLens target genre has equal importance. Current routes are the Cartesian product of active genres and all target genres. Dataset feasibility is:

`C(i,g) = count(i,g) / sqrt(count(i) count(g))`.

This gives `C(i,i)=1` for a nonempty genre. The cached matrix records marginal, joint, and normalized values.

Route score:

`0.35 L(i) + 0.35 S_t(i) + 0.30 C(i,g) - 0.50 P_t(i,g) + 0.02 I[route=previous]`.

Ties use score, long-term strength, active strength descending, then interest and target-genre strings ascending. These constants are engineering defaults and were not tuned against IoR.

Route penalties start at zero. Every feedback step applies `P <- 0.8 P`; rejection then adds `0.5` only to the current route. Rejected items receive a recoverable two-step cooldown.

## Candidate tiers

Original-history IDs, accepted-path IDs, target, and currently cooling-down items are excluded.

1. item contains both route genres;
2. item contains `i` plus another genre with nonzero dataset co-occurrence to `g`;
3. item contains `g` and at least one current active genre.

Ordering is tier ascending, tier support descending, then MovieLens ID ascending; at most 20 are exposed. No SASRec, IoI, IoR, embeddings, or evaluator scores are used.

The LLM receives demographics, `L`, `S_t`, all target genres, the current route and score components, and exact allowed titles. It must return one exact title. Parsing permits an exact bare title or a singleton string list; there is no fuzzy matching or semantic repair.

`MAX_INTERMEDIATE_STEPS=8`, with a 24-decision infrastructure guard. Planning may stop earlier if every target genre reaches active strength `0.8` or no useful route/candidate remains. The normalized final path is accepted intermediates plus the predefined target. Formal evaluation is intentionally outside this module.

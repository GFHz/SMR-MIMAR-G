# SSD-PR (Structure-Semantic Decoupling for Proactive Recommendation)

SSD-PR follows a structure-semantic decoupling principle. Explicit, deterministic components handle long-term interest profiling, migration-route construction, and catalog grounding, while the frozen LLM mainly performs semantic item selection among legal candidates.

SSD-PR uses complete chronological positive history (`rating >= 4`) for a static long-term profile. It has no dynamic interest state, Need/Coverage state, Commitment, route switching, or feedback penalty.

For history length `n` and 1-based positions `P_i` containing genre `i`:

```text
F(i) = |P_i| / n
b(p) = min(3, floor(4(p-1)/n))
P(i) = number of occupied bins / 4
R(i) = mean_{p in P_i}(p/n)
L(i) = clip_[0,1](0.5 F(i) + 0.3 P(i) + 0.2 R(i))
```

Genres are ranked by `(-L(i), genre)` and the Top-5 are retained. All target genres are preserved.

Across the MovieLens catalog:

```text
C(i,g) = N(i,g) / sqrt(N(i) N(g))
RoutePrior(i,g) = 0.35 L(i) + 0.30 C(i,g)
```

For `i != g`, feasibility requires `N(i,g) > 0`; for `i == g`, `N(i) > 0` suffices. Global order is `(-RoutePrior, -L, interest, target genre)`. A deterministic diversity rule retains at most four static routes.

Support is DIRECT only. Candidate `x` must contain both endpoints of at least one route. Complete history, prior intermediates, and target are excluded. Candidates are ordered by MovieLens ID and truncated to 20. This deterministic ordering is an implementation choice for reproducibility; MovieLens item ID is not treated as a semantic relevance score. Qwen selects one exact allowed title; the strict parser has no fuzzy matching or semantic repair. A step has one request and at most two correction requests.

```text
O(x) = |G_x intersection G_T| / |G_T|
```

The first intermediate bypasses comparison. The Target-Overlap Non-regression Constraint accepts continuation when `O(new) >= O(previous)`. If a later selected item has `O(new) < O(previous)`, it is excluded, no guard retry occurs, extension stops, and the predefined target is appended. Its purpose is to prevent subsequent intermediates from weakening an already established target-attribute connection. This is a structural stopping rule, not a score or simulated rejection, and it does not guarantee shortest paths. The locally greedy rule may prevent detours that temporarily reduce target overlap before later recovering it; tolerance thresholds, multi-step cumulative proximity, and limited rollback/lookahead remain future work. At most six intermediates are accepted. SASRec is post-hoc only.

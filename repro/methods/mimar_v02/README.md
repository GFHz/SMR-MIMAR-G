# MIMAR-v0.2

MIMAR-v0.2 is a new module; MIMAR-v0.1 remains frozen. The conceptual revision separates **route feasibility** from **route desirability**.

In v0.1, normalized genre co-occurrence contributed directly to route ranking. In v0.2 the same statistic

`C(i,g)=count(i,g)/sqrt(count(i)count(g))`

is logged but serves only as a feasibility gate. A cross-genre route is feasible when `count(i,g)>0`; an identity route `(i,i)` is feasible when the genre exists. If no directly feasible route or no legal candidate exists, the path stops cleanly; v0.2 does not invent an indirect fallback or silently restore `C` to the score.

For each target genre, coverage starts at zero. An accepted item adds `0.5` to every target genre it contains, clipped to one. `Need_t(g)=1-Coverage_t(g)`. Intermediate planning stops when every target genre reaches coverage `1.0`.

Among feasible routes:

`RouteScore_t(i,g)=0.35 L(i)+0.35 S_t(i)+0.30 Need_t(g)-0.50 P_t(i,g)+0.02 I[route=previous]`.

The `0.02` continuity bonus and v0.1 feedback memory (`rho=0.8`, rejection increment `0.5`, two-step item cooldown) are unchanged. Co-occurrence is absent from this additive expression. Explicit need permits different target attributes to become preferable as coverage accumulates; switching is score-driven, never forced.

Interest extraction/update and tiered catalog candidate construction are reused unchanged from v0.1. Original history, used intermediates, target, and cooling-down rejected items remain excluded. No SASRec, IoI, IoR, target rank, or evaluator score is imported into planning.

The prompt explicitly distinguishes abstract route genre labels from movie titles. It permits only an exact allowed title, retains the exact-only parser, and allows at most two correction retries without state changes.

# SSD-PR (Structure-Semantic Decoupling for Proactive Recommendation)

The internal module path retains the historical development identifier `smr_mimar_g` for reproducibility; the final public method name is SSD-PR.

This variant reuses SMR-MIMAR-v0.1 unchanged and adds one structural guard.
The first valid intermediate is accepted. Later candidates are accepted only
when their fraction of target genres is at least that of the previous accepted
item. A decrease ends the intermediate sequence, excludes that candidate from
the path, and appends the predefined target through normal endpoint
normalization.

Target overlap is not a score, candidate filter, retry signal, or user-feedback
event. The method has no dynamic interests, penalties, coverage, commitment,
SASRec planning signal, or additional threshold.

# SMR-MIMAR-v0.1

Static MI-Bridge selects one interest–target bridge pair. SMR-MIMAR instead
builds one diverse, static set of up to four feasible routes between the top
five full-history long-term interests and every target genre. Route prior is
`0.35 L(i) + 0.30 C(i,g)` and is computed once.

Selection is deterministic. At each greedy slot, routes introducing both a new
start and target attribute are preferred, then routes introducing either one;
ties retain static-prior order. Remaining slots are score fills.

An eligible movie directly supports every selected route whose two endpoint
genres it contains, so one multi-tag movie can support multiple routes. Legal
candidates are ordered by MovieLens item ID. The LLM jointly sees the fixed
route set and exact candidates, but selects only the concrete next item.

There is no short-term state, activation/deactivation, route switching,
coverage/need, feedback penalty, commitment, SASRec planning signal, or dynamic
route update. The route set stays fixed for all six possible intermediates.
This deliberately avoids the instability observed in dynamic switching. No
superiority claim is made before evaluation.

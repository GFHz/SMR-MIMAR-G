# SMR-MIMAR-G

Static Multi-Route Multi-Interest Multi-Attribute Planning with Target-Overlap Guard for Proactive Recommendation.

## 1. Overview

Proactive recommendation plans a sequence of intermediate items intended to move a user from established interests toward a predefined target item. SMR-MIMAR-G is a catalog-grounded, evaluator-independent planning method built as an adapted local-LLM pre-study around the LLM-IPP research setting.

This repository reports an exploratory experiment with five MovieLens-1M users. It is not a full large-scale benchmark and does not reproduce the original GPT-3.5 LLM-IPP results.

## 2. Core Idea

```text
Full Positive History
  -> Long-Term Multi-Interest Profile
  -> All Target Genres
  -> Feasible Interest-to-Target Routes
  -> Diverse Static Route Set
  -> Catalog-Grounded Candidates
  -> LLM Exact Item Selection
  -> Target-Overlap Guard
  -> Target
  -> Post-hoc SASRec Evaluation
```

The planner deterministically computes interests, routes, legal candidates and exclusions. The LLM only selects one exact title from a supplied candidate list. SASRec is used only after generation.

## 3. Method

### 3.1 Long-Term Multi-Interest

For the complete timestamp-ordered rating>=4 history of length `n`, let `P_i` be the 1-based positions containing genre `i`:

```text
F(i) = |P_i| / n
b(p) = min(3, floor(4(p-1)/n))
P(i) = number of distinct occupied bins / 4
R(i) = mean_{p in P_i}(p/n)
L(i) = clip_[0,1](0.5 F(i) + 0.3 P(i) + 0.2 R(i))
```

Genres are ranked by `(-L, genre ascending)` and the Top-5 are retained.

### 3.2 Multi-Attribute Target

Every distinct MovieLens genre attached to the target is retained. The target is sampled outside the user’s complete positive history and is reference-only during intermediate selection.

### 3.3 Genre Co-occurrence

Over the complete MovieLens catalog:

```text
C(i,g) = N(i,g) / sqrt(N(i)N(g))
```

when both marginal counts are nonzero; otherwise it is zero. For `i == g`, `C(i,i)=1` when `N(i)>0`.

### 3.4 RoutePrior

Route feasibility is:

```text
if i == g: feasible iff N(i) > 0
else:      feasible iff N(i,g) > 0
```

The prior is:

```text
RoutePrior(i,g) = 0.35 L(i) + 0.30 C(i,g)
```

Routes are globally ordered by `(-RoutePrior, -L, interest, target genre)`.

### 3.5 Diverse Static Route Set

For route `r`:

```text
Novelty(r) = 1[start interest unseen] + 1[target genre unseen]
```

The greedy selector chooses maximum novelty first. Global route rank breaks novelty ties. Up to four routes are selected once and remain fixed for the whole path. Multiple routes may share a start or target genre.

### 3.6 DIRECT Candidate Support

```text
SupportedRoutes(x) = {
  (i,g) in R* where i in Genres(x) and g in Genres(x)
}

RouteSupportCount(x) = |SupportedRoutes(x)|
```

Only DIRECT support is implemented. There is no Tier 2, Tier 3, relaxed endpoint, embedding or multi-hop fallback. Legal movies must support at least one route. Full positive history, previously accepted intermediates and target are excluded. Remaining movies are ordered by integer MovieLens ID and truncated to 20.

`RouteSupportCount` is prompt metadata only; it does not rank candidates.

### 3.7 LLM Candidate Selection

The local Qwen model receives demographics, Top-5 `L(i)` rows, target title/genres, static `R*`, RoutePrior values, current generated path, and exact candidate titles/genres/support metadata. It must return exactly one supplied title.

### 3.8 Strict Parser and Retry

The parser accepts an exact allowed bare title or a singleton JSON/Python string list containing an exact allowed title. It rejects multiple items, explanations, unknown titles and malformed containers. There is no fuzzy matching, normalization or semantic repair. A step receives one initial request and at most two unchanged-candidate correction requests.

### 3.9 Target-Overlap Guard

```text
TargetOverlap(x) =
  |Genres(x) intersect TargetGenres| / |TargetGenres|
```

The first intermediate bypasses comparison. For later items:

```text
if new_overlap < previous_overlap:
    remove the newly selected item from the accepted path
    stop without retry
    append the predefined target
```

Equal overlap is allowed. There is no margin parameter, feedback penalty or SASRec signal.

### 3.10 Stopping Conditions

In implementation order: planner already stopped (API error), six accepted intermediates, no legal candidate, invalid LLM output after all three generations, strict overlap decrease, or reaching six after accepting the current item. A missing feasible route has no separate status and appears as `no_legal_candidate`. Every normal final path appends the predefined target.

### 3.11 Post-hoc Evaluator

Formal-Evaluation-v1 maps paths to the frozen ProRL-style SASRec checkpoint after planning. SASRec, IoI and IoR never affect route or item selection.

## 4. Exact Parameters

| Parameter | Value |
|---|---|
| positive rating threshold | >= 4 |
| TOP_K_USER_INTERESTS | 5 |
| ROUTE_SET_SIZE | 4 |
| TOP_K_CANDIDATES | 20 |
| MAX_INTERMEDIATE_STEPS | 6 |
| MAX_LLM_RETRIES | 2 |
| maximum LLM generations/step | 3 |
| alpha / gamma | 0.35 / 0.30 |
| frequency/persistence/recency weights | 0.5 / 0.3 / 0.2 |
| temporal bins | 4 |
| candidate support | DIRECT only, >=1 route |
| candidate order | MovieLens ID ascending |
| interest order | `(-L, genre)` |
| global route order | `(-RoutePrior,-L,interest,target genre)` |
| diversity order | maximum novelty; global rank breaks ties |
| overlap guard | strict decrease only |
| model/runtime | qwen3:4b-q4_K_M / Ollama |
| known Ollama version | 0.33.3 |
| quantization | Q4_K_M |
| temperature / seed | 0.1 / 42 |
| think | false |
| num_ctx / num_predict | 4096 / 2048 |
| request timeout / keep_alive | 180 s / 5m |
| controlled pool | 100 unique items/user |
| controlled pool seed | 20260905 |
| paths/user | 2 |
| IoI epsilon | 1e-12 |

The exhaustive source-linked table is in `repro/results/smr_mimar_g/final_implementation_audit/FINAL_PARAMETER_TABLE.md`.

## 5. Repository Structure

```text
configs/                         safe local-LLM example configuration
data/README.md                   external-data placement instructions
docs/METHOD.md                   complete implementation specification
docs/EXPERIMENT.md               final protocol and commands
docs/FINAL_RESULTS.md            all 10 frozen paths and result table
docs/FINAL_IMPLEMENTATION_AUDIT.md read-only implementation audit
repro/methods/mimar_v01/         shared long-term interest/co-occurrence code
repro/methods/smr_mimar_v01/     static multi-route planner
repro/methods/smr_mimar_g/       target-overlap guard wrapper
repro/evaluators/formal/         Formal-Evaluation-v1 metrics/protocol
repro/evaluators/sasrec/         ProRL-style SASRec loader/wrapper
repro/experiments/smr_mimar_g/   exact final controlled runner
repro/results/smr_mimar_g/       frozen final paths, metrics and audit
```

Additional imported support modules are retained because the exact frozen runner references them. They are not the final method.

## 6. Environment

Use Python 3.10. Install `requirements.txt`. Install Ollama separately, start its local service, and obtain only `qwen3:4b-q4_K_M`. The model is not distributed. Copy `configs/local_config.example.json` to `repro/local_config.json`; it contains no credential.

## 7. Dataset

Use MovieLens-1M, not MovieLens Latest. Place original `ratings.dat`, `movies.dat`, and `users.dat` under `dataset/ml-1m/`. Raw data are not distributed. Positive feedback is rating >=4. See `data/README.md`.

## 8. Controlled Candidate Pool

Each frozen 100-item pool starts with Last-20, target, and saved Static MI-Bridge direct candidates, then receives deterministic random catalog fillers. **Last-20 is only part of controlled-pool construction. It is not the SMR-MIMAR-G interest history.** SMR-MIMAR-G uses the complete positive history for `L(i)` and excludes that complete history from intermediate candidates. At most 20 direct-support candidates reach the LLM.

## 9. Running the Experiment

Prerequisites: external MovieLens files, ProRL-formatted evaluator data, SASRec checkpoint, local config, Ollama service and exact Qwen model. Then run from repository root:

```powershell
python -m repro.experiments.smr_mimar_g.run_controlled_positive_5users
```

The runner refuses to overwrite completed outputs. Preserve or relocate the included frozen result directory before a fresh reproduction. Exact asset paths and caveats are in `docs/EXPERIMENT.md`.

## 10. Evaluation Metrics

```text
IoI = log(P_after + 1e-12) - log(P_before + 1e-12)
IoR = R_before - R_after
```

Rank is 1-based over the full SASRec vocabulary. Proxy Acceptability is the mean sequential `sigmoid(sequence embedding · item embedding)` and is **not CTR**. Coherence is the adjacent-pair genre-overlap rate over intermediates plus target. HistoryReuseRate is resolved historical non-target occurrences divided by all non-target occurrences. In the final runner, PHRR is only an alias for HistoryReuseRate.

## 11. Final Controlled Experiment

Users and targets:

- 419: Clean Slate (Coup de Torchon) (1981) [Crime]
- 5021: Drunken Master (Zui quan) (1979) [Action, Comedy]
- 2677: Jingle All the Way (1996) [Adventure, Children's, Comedy]
- 3113: League of Their Own, A (1992) [Comedy, Drama]
- 2249: Timecop (1994) [Action, Sci-Fi]

Two paths were generated per user. All valid parser/catalog/guard intermediates were treated as accepted. There was no real or simulated rejection.

## 12. Final Preliminary Results

| Method | IoI | IoR | Proxy | Coherence |
|---|---:|---:|---:|---:|
| Static MI-Bridge | 0.6578132613 | 25.5 | 0.8841134284 | 0.9652777778 |
| Dynamic-No-Bridge | 0.6541540310 | 267.6 | 0.8668649547 | 0.8500000000 |
| MIMAR-v0.1 | 1.8820530684 | 191.6 | 0.8055583143 | 1.0000000000 |
| MIMAR-v0.2 | 0.6761665923 | 90.9 | 0.7666419392 | 0.9200000000 |
| MIMAR-v0.3 | 0.2319171814 | -88.0 | 0.7836425823 | 0.9666666667 |
| SMR-MIMAR-v0.1 | 1.3499917349 | -166.9 | 0.5920387775 | 0.9333333333 |
| **SMR-MIMAR-G (final)** | **2.3745645593** | **416.0** | **0.6067911714** | **0.9333333333** |

SMR-MIMAR-G mean normalized path length was 4.6 including target; generation success was 10/10. These values come from only five users. Each user’s two paths were identical under the fixed seed. There was no significance test, and methods have different stopping/candidate rules. The results do not establish general superiority.

## 13. Why the Guard Exists

The frozen post-hoc diagnostic found:

```text
Pearson(TargetOverlap, DeltaIoR)  = 0.054458
Spearman(TargetOverlap, DeltaIoR) = 0.067951
```

Therefore TargetOverlap is not used as a score. Directional groups were:

```text
overlap increase: mean DeltaIoR = 24.75
overlap same:     mean DeltaIoR = 74.035714
overlap decrease: mean DeltaIoR = -1283.0
```

The final method uses overlap only as a monotonic backward guard. Because this guard was motivated by post-hoc analysis on the same small setting, its final result remains exploratory and carries development-set overfitting risk.

## 14. Important Method History

Static MI-Bridge introduced one explicit interest-to-target bridge. Dynamic MIMAR explored evolving interests, target attributes and route switching, but switching exhibited instability. SMR-MIMAR replaced this with a diverse static route set. Prefix analysis then showed late-path degradation, and the target-overlap audit identified severe negative rank movement specifically when structural overlap decreased. SMR-MIMAR-G is the final guarded variant. Historical variants are comparisons, not the final method.

## 15. Limitations

- five users only
- two paths per user
- duplicate repeated paths because the same request seed was reused
- one dataset
- one local 4B model
- synthetic unobserved targets
- offline all-accept assumption
- no real rejection feedback
- DIRECT candidate support only
- no broader fallback tier
- coarse genre metadata
- SASRec mapping status `LIKELY_COMPATIBLE`
- PHRR is only a HistoryReuseRate alias
- methods differ in stopping and candidate filtering
- guard designed after a post-hoc audit on the same small setting
- development-set overfitting risk
- no independent held-out validation
- no significance tests
- no evidence of general superiority over LLM-IPP, ITMPRec, T-PRA, or ProRL

## 16. Reproducibility Notes

- Pilot/cohort seed: 20260905.
- Per-user target/pool seed: `20260905 + user_id`.
- Every Ollama request seed: 42.
- Interest order: `(-L, genre)`.
- Target genres: sorted unique strings.
- Global route order: `(-RoutePrior,-L,interest,target genre)`.
- Diversity ties: prior global rank.
- Candidate order: integer MovieLens ID ascending.
- Model digest and all generation settings are recorded in the safe example config.
- Fixed request seed does not guarantee bitwise identity across hardware/runtime versions.

## 17. Citation

This is a student research pre-study, not a peer-reviewed publication.

```bibtex
@misc{gao2026smrmimarg,
  title={SMR-MIMAR-G: Static Multi-Route Multi-Interest Multi-Attribute Planning with Target-Overlap Guard for Proactive Recommendation},
  author={Gao, Feihong},
  year={2026},
  note={Research pre-study}
}
```


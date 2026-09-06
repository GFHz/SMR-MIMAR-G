# SSD-PR

SSD-PR (Structure-Semantic Decoupling for Proactive Recommendation) is a catalog-grounded pre-study of proactive recommendation and influence-path planning with a frozen local LLM. SSD-PR separates explicit structural planning from LLM-based semantic item selection and reports a controlled five-user comparison with a local LLM-IPP-style baseline. The results are exploratory and do not establish general superiority.

## Overview

LLM-IPP-style receives demographics, chronological positive history, and a target, then lets the frozen LLM implicitly infer interests and plan an ordered influence path. SSD-PR instead follows a structure-semantic decoupling principle: deterministic components construct interpretable interest-to-target routes and legal catalog candidates, while the LLM selects an exact item from those candidates.

```text
Complete Positive History
  -> Explicit Long-term Multi-interest Profile
  -> Multi-interest × Multi-target-attribute Routes
  -> Catalog-grounded Candidate Set
  -> Frozen LLM Item Selection
  -> Target-Overlap Non-regression Constraint
  -> Predefined Target
```

For the controlled local comparison, both LLM-IPP-style and SSD-PR are provided with the same complete positive user history.

## Method

1. Positive interactions are MovieLens-1M ratings `>= 4`, ordered by timestamp.
2. For genre `i`, let `F(i)` be frequency, `P(i)` four-bin temporal persistence, and `R(i)` normalized recency: `L(i) = clip_[0,1](0.5 F(i) + 0.3 P(i) + 0.2 R(i))`. The five highest-scoring interests are retained.
3. All target genres are preserved.
4. Feasible `interest -> target genre` routes use `C(i,g) = N(i,g) / sqrt(N(i) N(g))` and `RoutePrior(i,g) = 0.35 L(i) + 0.30 C(i,g)`.
5. A deterministic diversity-aware selector retains at most four static routes.
6. Candidate support is DIRECT only: a candidate must contain both endpoints of at least one selected route. Complete positive history, prior intermediates, and target are excluded. Eligible candidates are sorted by MovieLens item ID and truncated to 20. This deterministic ordering is an implementation choice for reproducibility; MovieLens item ID is not treated as a semantic relevance score.
7. Frozen Qwen selects one exact title from at most 20 catalog-grounded candidates. The strict parser performs no fuzzy matching or semantic repair.
8. `O(x) = |Genres(x) intersect TargetGenres| / |TargetGenres|`. The Target-Overlap Non-regression Constraint accepts `O(new) >= O(previous)` and stops extension only when `O(new) < O(previous)`, preventing later intermediates from weakening an already established target-attribute connection.

SASRec, IoI, IoR, target probability, and target rank are never used during planning.

## Key Ideas

1. **Structure-Semantic Decoupling.** Explicit components perform long-term interest extraction, route construction, and catalog legality checks; the LLM performs semantic selection inside the legal set.
2. **Multi-interest × Multi-target-attribute Routing.** Multiple long-term interests and all target genres form interpretable migration directions.
3. **Catalog-grounded Selection and Target-Overlap Constraint.** Every intermediate has real catalog support, and a simple non-regression condition stops structurally backward extension.

## Experimental Setup

- Dataset/positive interaction: MovieLens-1M, rating `>= 4`
- Users: `419, 5021, 2677, 3113, 2249`
- Complete positive-history lengths: `77, 95, 22, 27, 48`
- Paths per user: 2
- Model/runtime: `qwen3:4b-q4_K_M` / Ollama
- Temperature/seed: `0.1 / 42`; thinking disabled; context/output: `4096 / 2048`

Neither planner performs supervised training, fine-tuning, LoRA, QLoRA, reinforcement learning, or parameter updates. Both use the same frozen local Qwen model and the same complete chronological positive history. This is not a strict reproduction of the original LLM-IPP GPT experiment; it is a local LLM-IPP-style same-protocol baseline. Formal-Evaluation-v1 and SASRec are post-hoc only.

## Main Results

| Metric | LLM-IPP-style | SSD-PR |
|---|---:|---:|
| IoI | 0.1027 | 2.3746 |
| IoR | 313.125 | 416.000 |
| Proxy Acceptability | 0.9296 | 0.6068 |
| Coherence | 0.8500 | 0.9333 |

In this five-user descriptive pre-study, SSD-PR shows stronger target guidance: IoR increases by approximately `32.85%`, and Coherence is higher. Proxy Acceptability decreases substantially, indicating a trade-off between target guidance and short-term compatibility. These results do not support a statistical-significance or general-superiority claim.

## Repository Structure

```text
configs/            local runtime configuration example
data/               dataset placement instructions only
docs/               method, protocol, results, and implementation audit
repro/methods/       planner and supporting modules
repro/evaluators/    post-hoc evaluation
repro/experiments/   final comparison runner
repro/results/       frozen experimental outputs
```

Historical internal identifiers remain where required for reproducibility, but SSD-PR is the final public-facing method.

## Reproduction

Install Python 3.10 dependencies and run non-LLM tests:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s repro/methods/smr_mimar_g/tests -p "test_*.py"
```

Frozen CSV/JSON outputs can be inspected without regeneration. To reproduce generation, prepare [external data](data/README.md), copy `configs/local_config.example.json` to `repro/local_config.json`, start Ollama with the exact model, preserve the included result directory elsewhere, and run:

```powershell
python -m repro.experiments.main_comparison.run_main_comparison
```

## Data

See [data/README.md](data/README.md).

## Limitations

- Five users and two paths per user only; the fixed seed produced identical paired paths, so the effective independent sample is small.
- No statistical-significance claim; only MovieLens-1M and coarse genres are studied.
- One frozen local Qwen3-4B setting; no claim that results persist after task-specific training.
- DIRECT support can structurally favor coherence.
- Candidate truncation currently uses deterministic item-ID ordering rather than a relevance-aware ranking. Future work could rank legal candidates using route support strength, popularity, or learned relevance while preserving catalog constraints.
- The Target-Overlap Non-regression Constraint is locally greedy and may prevent detours that temporarily reduce target overlap before later recovering it. Future work could study a tolerance threshold, multi-step cumulative target proximity, or limited rollback/lookahead.
- The target-overlap mechanism was designed in the same development setting and needs independent validation; it does not guarantee shortest paths.
- Stronger guidance is accompanied by lower Proxy Acceptability.
- SASRec mapping is `LIKELY_COMPATIBLE`, not proven identical to the unavailable training-time mapping.
- The local baseline must not be conflated with original LLM-IPP paper results.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 GFHz.

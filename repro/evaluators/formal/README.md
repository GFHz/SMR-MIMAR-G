# Formal-Evaluation-v1

Formal-Evaluation-v1 is the unified post-hoc evaluation protocol for the final SSD-PR versus LLM-IPP-style experiment. It does not generate paths, train models, or modify either planner. SASRec is used only after path generation is complete.

## Responsibilities

- `title_resolver.py`: resolve generated titles against the MovieLens catalog using exact matching followed only by unique The/A/An article normalization.
- `protocol.py`: separate the predefined target, apply strict path eligibility rules, and assemble formal and diagnostic outputs.
- `metrics.py`: compute IoI, IoR, and sequential SASRec proxy scores.
- `coherence.py`: compute adjacent genre-overlap Coherence over `intermediates + target`, excluding the history-to-path boundary.

## Primary metrics

```text
IoI = log(P_after + 1e-12) - log(P_before + 1e-12)
IoR = R_before - R_after
```

`P_before` and `P_after` are frozen-SASRec target probabilities; `R_before` and `R_after` are 1-based target ranks. Positive IoI and IoR indicate increased target probability and improved target rank, respectively.

Proxy Acceptability is the arithmetic mean of sequential sigmoid scores from the frozen SASRec model. It is a model-based compatibility proxy, not empirical CTR or observed user feedback.

Coherence is the mean indicator of non-empty genre overlap between adjacent items in `intermediates + target`.

Structural and validity diagnostics—including title-resolution status, history reuse, and target position—remain available in raw outputs for auditing. They are not treated as primary recommendation-quality metrics in the final report.

## Tests and use

From the repository root, run:

```powershell
python -m unittest discover -s repro/evaluators/formal/tests -p "test_*.py"
```

One integration test requires the external MovieLens `movies.dat` file described in `data/README.md`. The final main-comparison runner calls this evaluator; there is no separate final-evaluation command documented here.

The implementation reuses the independently written SASRec adapter in `repro/evaluators/sasrec/`. Mapping status is `LIKELY_COMPATIBLE`, not proven identical to an unavailable training-time mapping artifact. This is not a strict reproduction of the original LLM-IPP evaluator.

Reference design: [ProRL evaluator.py](https://github.com/hongruhou89/ProRL/blob/506e91355377f546dcf51f684fc871fe57a9d5c8/evaluator.py).

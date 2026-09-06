# Formal-Evaluation-v1

Unified evaluation of the four frozen User 1 paths. This module does not generate
paths, train models, alter the MI-Bridge method, or overwrite previous results.
The protocol is specified in `protocol.py` and `repro/docs/FORMAL_EVALUATION_PROTOCOL.md`.

## Responsibilities

- `title_resolver.py`: exact MovieLens catalog match, then unique The/A/An article
  normalization. Preserve raw/resolved titles, status and resolution type.
- `protocol.py`: target separation, strict eligibility, validity and the three
  mechanism metrics. Immutable protocol constants, including IoI epsilon 1e-12.
- `metrics.py`: delegate unchanged SASRec target probabilities/ranks and sequential
  sigmoid proxy scoring; natural-log IoI, rank difference IoR, valid-only means.
- `coherence.py`: independently implemented adjacent genre-overlap average for
  intermediates + target, without a history-boundary edge.
- `tests/`: synthetic unit fixtures plus actual catalog resolution; no user feedback.

Scoring dependencies are the existing `.venv-repro`: torch 2.8.0+cpu, RecBole 1.2.1,
numpy 1.26.4, pandas 2.2.3. No installation or parameter change in this stage.

From repository root:

```powershell
# Repeat rule tests without writing evaluation output:
.\.venv-repro\Scripts\python.exe -B -X utf8 -m unittest discover -s repro/evaluators/formal/tests -v

# One-time evaluation, refusing to overwrite an existing formal_evaluation directory:
.\.venv-repro\Scripts\python.exe -B -X utf8 -m repro.experiments.case_study.evaluate_user1_formal
```

The runner reads existing baseline v2/Ours paths and saved bridge candidates,
checks the full frozen token maps, runs tests, uses the existing checkpoint for
CPU inference, and writes eight new JSON files under
`repro/results/case_study/mi_bridge_v1/formal_evaluation/`.

No Qwen/Ollama/OpenAI calls. No re-parsing or re-generation of raw model outputs.
Only evaluation-time canonical identities are added. Protocol code hashes and
pre/post hashes for previous files are stored in the results. Future changes to
metric definitions require a new version and a separate output directory.

Reference design: [ProRL evaluator.py](https://github.com/hongruhou89/ProRL/blob/506e91355377f546dcf51f684fc871fe57a9d5c8/evaluator.py).
Existing independently written SASRec adapter is reused; no external source is
copied/edited. Public availability is not a redistribution license. Mapping is
LIKELY_COMPATIBLE, not independently verified against a training mapping artifact.
This is not a strict reproduction of the original LLM-IPP evaluator.

Target probability is a full-vocabulary softmax; Proxy Acceptability instead uses
the existing ProRL-style **sigmoid dot-product score** at each prefix. It is neither
the same softmax distribution nor true CTR. Rank ties follow the unchanged existing
PyTorch argsort implementation, with no new stable-sort or ID-based tiebreaker.

Invalid paths stay in the record. Diagnostic-drop values never enter formal means.
SR and TargetLastRate use all successfully parsed nonempty string-list paths as
their denominator, including paths invalid for SASRec metrics. Defined-metric
counts accompany valid-only means; empty denominators yield null, never zero.

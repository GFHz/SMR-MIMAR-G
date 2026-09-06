# ProRL-style SASRec minimal validation

ProRL is used as an evaluation reference. This is an independently written
single-user adapter, not a copy of `PRAEvaluator`, not a training pipeline, and
not a strict reproduction of the LLM-IPP paper evaluator.

Reference: [ProRL evaluator.py](https://github.com/hongruhou89/ProRL/blob/506e91355377f546dcf51f684fc871fe57a9d5c8/evaluator.py),
observed snapshot `506e91355377f546dcf51f684fc871fe57a9d5c8`.
No LICENSE was found in the referenced ProRL snapshot. Source attribution is
not a redistribution license; do not publish its source/checkpoint/data as ours
or assume permission. Download from the original source instead. The reference
directories and their provenance README remain unchanged.

## Scope and artifacts

- Official checkpoint, strict loading, CPU, no optimizer or training invocation.
- Published `external/ProRL/datasets/ml-1m-sas` `.inter`, `.user`, `.item` files.
  Missing any of them is fatal. No ratings conversion or new train/test split.
- `DataAdapter` builds exact title -> raw MovieLens ID -> `field2token_id`.
  It does not read ProRL policy `item2id`, use semantic IDs, or correct titles.
- `SASRecEvaluator` scores one sequence at a time; preserves all 20 history items.
  Any sequence longer than checkpoint max 50 raises an error, never truncates.
- Frozen four paths only. All target occurrences are excluded from extensions,
  preserving the other items' order and target's original 1-based positions.
- Unknown titles fail strict mode. An explicitly separate drop mode is diagnostic
  only, never a substitute formal result. No LLM repairs, generation or judging.

Checkpoint downloaded to `repro/results/evaluator_validation/checkpoints/`,
because no local checkpoint existed and `external/` is read-only in this task.
Acquisition is pinned by commit, recorded URL/bytes/SHA-256. `torch.load` uses
`weights_only=False` because the official file includes a RecBole Config pickle;
only this explicitly trusted official file is accepted. Never use this loader
with an arbitrary untrusted pickle/checkpoint. This is not a generic safe loader.

## Verified environment and minimal installation

Windows, `.venv-repro`, Python 3.10.20, torch 2.8.0+cpu, RecBole 1.2.1,
numpy 1.26.4, pandas 2.2.3. CPU-only CUDA availability is intentionally False.
Ollama and its GPU/model configuration were not changed.

Commands used (from repository root, only in isolated environment):

```powershell
.\.venv-repro\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-repro\Scripts\python.exe -m pip install recbole==1.2.1 --no-deps
.\.venv-repro\Scripts\python.exe -m pip install PyYAML==6.0.3 colorlog==4.7.2 scipy==1.15.3 scikit-learn==1.7.2 tensorboard==2.20.0 texttable==1.7.0
```

`environment.json` records all actually installed versions, including transitive
dependencies. No pre-existing package was upgraded. `packages_before.json` was
captured during preparation after installation had already begun; it is not a
complete pre-turn environment snapshot. The pre-install package listing in
the execution record established numpy/pandas and existing packages were kept.

This intentionally minimal installation is **not a full dependency-clean RecBole
installation**: `pip check` reports missing plotly, psutil, ray, tabulate, thop,
and RecBole's exact colorama 0.4.4 requirement versus existing 0.4.6. These are
not needed on the tested inference import path. No stubs were injected to bypass
imports. RecBole's own pandas FutureWarnings are visible; no package/source patch
or pandas upgrade was applied. Installing the full training stack is out of scope.

## Run and test

```powershell
.\.venv-repro\Scripts\python.exe -B -X utf8 -m repro.evaluators.sasrec.checkpoint_loader
.\.venv-repro\Scripts\python.exe -B -X utf8 -m repro.evaluators.sasrec.evaluator
.\.venv-repro\Scripts\python.exe -B -X utf8 -m repro.evaluators.sasrec.evaluator --finalize
```

Download preparation is idempotent and verifies the existing checkpoint hash.
The generation of numerical artifacts/report refuses to overwrite saved outputs.
After completion, verify again without overwriting results using:

```powershell
.\.venv-repro\Scripts\python.exe -B -X utf8 -m unittest discover -s repro/evaluators/sasrec/tests -v
```

## Primitive definitions

`get_target_score`: softmax probability from full-sort logits.
`get_target_rank`: 1-based descending probability rank, using PyTorch argsort.
Both include padding row 0 and do not exclude history items, matching the released
ProRL score/rank computation. Vocabulary size 3884 = 3883 movies + padding.
This is not an unseen-only recommendation rank.

`get_item_score`: sigmoid of sequence representation dot item embedding.
`get_path_item_scores`: score each item **before** appending it, then return scores
and their arithmetic mean as `SASREC_PROXY_ACCEPTABILITY`. Not empirical CTR.
No final IoI/IoR, coherence, click simulation, beam-max or multi-user aggregation.

## Mapping provenance limit

Strict parameter loading and all architecture keys match. Current vocabulary is
the released evaluator's RecBole mapping before any split. RecBole token remapping
occurs in dataset construction, before `build()`/`data_preparation()`; no split is
needed for inference. Target raw ID 2620 maps to internal 2844, not raw 2620 nor
ProRL policy ID 2898.

The checkpoint Config records `ml-1m` / `./dataset/ml-1m`, while its published
evaluator uses `ml-1m-sas`. No independent training-time token map was provided
inside the checkpoint. Matching vocabulary size, round trips and finite scores
does **not** prove training row semantics. Thus this validates execution of the
released evaluator mapping, not independent training/evaluator alignment. Obtain
the authors' original mapping or confirmation before making paper-level claims.

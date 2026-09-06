# Final controlled experiment

## Cohort and targets

| User | Target | Genres |
|---:|---|---|
| 419 | Clean Slate (Coup de Torchon) (1981) | Crime |
| 5021 | Drunken Master (Zui quan) (1979) | Action, Comedy |
| 2677 | Jingle All the Way (1996) | Adventure, Children's, Comedy |
| 3113 | League of Their Own, A (1992) | Comedy, Drama |
| 2249 | Timecop (1994) | Action, Sci-Fi |

Two paths were generated per user, with at most six intermediates. Every parser-valid, catalog-valid, guard-valid intermediate was treated as accepted. There was no real or simulated user rejection.

## Controlled 100-item pools

Each pool was frozen by adding the user’s Last-20 history, target, and saved Static MI-Bridge direct candidates without duplicates, then shuffling remaining catalog items with `Random(20260905 + user_id)` and filling to 100 unique items. Last-20 is used only in this fairness-pool construction. SMR-MIMAR-G computes interests from the complete positive history and excludes that complete history from intermediate selection.

At each step, the method retains only pool movies that directly support at least one route, removes full-history items, used intermediates and target, sorts by MovieLens ID, and exposes at most 20 candidates.

## Local LLM

Ollama 0.33.3 served `qwen3:4b-q4_K_M` (`Q4_K_M`, frozen digest in the example config), with temperature 0.1, seed 42, thinking disabled, context 4096 and output budget 2048. Model binaries are not distributed.

## Formal evaluator

Formal-Evaluation-v1 uses the public ProRL MovieLens-1M SASRec checkpoint post-hoc. Its reconstructed token mapping is marked `LIKELY_COMPATIBLE`, not independently proven identical to an unavailable training-time mapping artifact. SASRec never participates in planning.

## Reproduction command

From the repository root, after installing dependencies, placing external assets, copying `configs/local_config.example.json` to `repro/local_config.json`, starting Ollama, and pulling the exact model:

```powershell
python -m repro.experiments.smr_mimar_g.run_controlled_positive_5users
```

The runner intentionally refuses to overwrite a completed output directory. This release includes frozen results for review; run in a clean copy where `repro/results/smr_mimar_g/controlled_positive_5users/summary.json` is absent, while preserving the supplied results elsewhere.

## Result scope

The experiment is an exploratory five-user pre-study. The fixed request seed made each user’s two paths identical. No significance test was performed. Full paths and metrics are in `docs/FINAL_RESULTS.md`.


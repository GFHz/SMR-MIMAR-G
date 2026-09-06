# Final experimental protocol

The final comparison is `LLM-IPP-style` versus `SSD-PR`. Both local planners receive the same complete chronological MovieLens-1M positive history (`rating >= 4`). This does not assert that the original LLM-IPP paper uses full history.

| User | Target | Target genres | History length |
|---:|---|---|---:|
| 419 | Clean Slate (Coup de Torchon) (1981) | Crime | 77 |
| 5021 | Drunken Master (Zui quan) (1979) | Action, Comedy | 95 |
| 2677 | Jingle All the Way (1996) | Adventure, Children's, Comedy | 22 |
| 3113 | League of Their Own, A (1992) | Comedy, Drama | 27 |
| 2249 | Timecop (1994) | Action, Sci-Fi | 48 |

Two paths are generated per user. Frozen `qwen3:4b-q4_K_M` runs through Ollama with temperature 0.1, seed 42, thinking disabled, context 4096, and output budget 2048. Neither planner trains or fine-tunes the LLM.

LLM-IPP-style receives demographics, all chronological positive movie titles/genres, and target, then implicitly plans a raw path without post-generation repair. SSD-PR computes its profile from the same history, constructs static routes, filters the frozen 100-item pool to DIRECT-support candidates, and appends the predefined target by protocol.

Formal-Evaluation-v1 uses the frozen ProRL-style SASRec checkpoint after generation only. It retains its frozen 20-item SASRec evaluation-history protocol, full-vocabulary softmax, 1-based rank, IoI/IoR definitions, Proxy Acceptability, and genre-overlap Coherence. Mapping status is `LIKELY_COMPATIBLE`.

After preparing `data/README.md`, dependencies, local config, and Ollama:

```powershell
python -m repro.experiments.main_comparison.run_main_comparison
```

The runner refuses to overwrite included frozen results. No significance or general-superiority claim is made.

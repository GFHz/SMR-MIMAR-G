# LLM-IPP-style vs SMR-MIMAR-G: same-protocol comparison

This is a five-user local Qwen case comparison, not a reproduction of the original GPT-based paper result.

| Method | Valid | IoI | IoR | Proxy | Coherence | History reuse | Target present | Target last | Duplicate rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLM-IPP-style same-protocol | 7/10 | -0.9899796470815401 | 26.428571428571427 | 0.9499672182968685 | 0.8785714285714286 | 0.675 | 0.9 | 0.8 | 0.0 |
| SMR-MIMAR-G | 10/10 | 2.3745645592687077 | 416 | 0.6067911714418586 | 0.9333333333333333 | 0.0 | 1.0 | 1.0 | 0.0 |

SMR-MIMAR-G target presence/last are guaranteed by endpoint normalization. Formal metrics use only strict evaluator-valid paths. No significance or general superiority is claimed.

# Final results

Both planners use the same complete chronological positive history and frozen Qwen3-4B configuration.

| Metric | LLM-IPP-style | SSD-PR | SSD-PR minus baseline |
|---|---:|---:|---:|
| IoI | 0.1027 | 2.3746 | +2.271851 |
| IoR | 313.125 | 416.000 | +102.875 |
| Proxy Acceptability | 0.9296 | 0.6068 | -0.322781 |
| Coherence | 0.8500 | 0.9333 | +0.083333 |

The IoR relative change is approximately `+32.85%`. In this five-user descriptive pre-study, SSD-PR shows stronger target guidance and higher Coherence but lower Proxy Acceptability, indicating a trade-off between target guidance and short-term compatibility.

Additional structural and validity diagnostics are retained in the raw experiment outputs and are not treated as primary recommendation-quality metrics.

The fixed seed produced identical paired paths per user. There is no significance test or claim of general superiority. The baseline is a local implementation, not the original GPT paper result.

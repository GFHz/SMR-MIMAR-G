# Final results

Both planners use the same complete chronological positive history and frozen Qwen3-4B configuration.

| Metric | LLM-IPP-style | SMR-MIMAR-G | SMR minus baseline |
|---|---:|---:|---:|
| Evaluator Valid Paths | 8/10 | 10/10 | +2 paths |
| IoI | 0.102713 | 2.374565 | +2.271851 |
| IoR | 313.125000 | 416.000000 | +102.875000 |
| Proxy Acceptability | 0.929572 | 0.606791 | -0.322781 |
| Coherence | 0.850000 | 0.933333 | +0.083333 |
| HistoryReuseRate | 46.5% | 0% | -46.5 pp |
| TargetPresenceRate | 90% | 100% | +10 pp |
| TargetLastRate | 90% | 100% | +10 pp |

The IoR relative change is `+32.854291%`. In this five-user descriptive pre-study, SMR-MIMAR-G shows stronger target guidance and constraint consistency but lower Proxy Acceptability.

SMR-MIMAR-G target presence and target-last are guaranteed by endpoint protocol, not learned successes. Raw outputs, strict validity diagnostics, per-path metrics, history/context audits, and frozen SMR references are in `repro/results/main_comparison/`.

The fixed seed produced identical paired paths per user. There is no significance test or claim of general superiority. The baseline is a local implementation, not the original GPT paper result.

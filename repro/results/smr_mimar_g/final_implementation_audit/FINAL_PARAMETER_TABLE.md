# SMR-MIMAR-G 最终参数表

本表以当前代码为唯一事实来源。行号对应审计时的工作区文件。

| NAME | VALUE | FILE | LINE / FUNCTION | PURPOSE |
|---|---:|---|---|---|
| frequency weight | 0.5 | `repro/methods/mimar_v01/interest_profile.py` | 4, `long_term_interests` | 长期兴趣的频率分量 |
| persistence weight | 0.3 | 同上 | 4, `long_term_interests` | 长期兴趣的四时间段持续性分量 |
| recency weight | 0.2 | 同上 | 4, `long_term_interests` | 长期兴趣的平均归一化位置分量 |
| temporal bins | 4 | 同上 | 31–32 | persistence 分母与时间分段数 |
| TOP_K_USER_INTERESTS | 5 | `repro/methods/smr_mimar_v01/routes.py` | 7, `ranked_long_term_interests` | 进入路由构造的最高长期兴趣数 |
| ROUTE_SET_SIZE | 4 | 同上 | 8, `select_diverse_routes` | 静态路由集最大规模 |
| alpha | 0.35 | 同上 | 5, `feasible_routes` | RoutePrior 中 L(i) 权重 |
| gamma | 0.30 | 同上 | 6, `feasible_routes` | RoutePrior 中 C(i,g) 权重 |
| minimum co-occurrence | `N(i,g)>0` when `i!=g`; `N(i)>0` when `i==g` | 同上 | 21 | 路由可行性门控；无额外数值阈值 |
| TOP_K_CANDIDATES | 20 | `repro/methods/smr_mimar_v01/candidates.py` | 2, `legal_candidates` | 每步提供给 LLM 的最大候选数 |
| candidate tiers | one tier: `DIRECT` | 同上 | 4–14 | 仅保留同时覆盖某条路由两端 genre 的电影 |
| MAX_INTERMEDIATE_STEPS | 6 | `repro/methods/smr_mimar_v01/planner.py` | 8, `step` | 最大中间项数 |
| MAX_LLM_RETRIES | 2 | `repro/methods/smr_mimar_v01/prompt.py` | 3 | 首次请求后最多两次纠正请求；总生成最多 3 次/步 |
| overlap threshold | 0 (strict decrease) | `repro/methods/smr_mimar_g/guard.py` | 7–10 | 仅当 `O_new-O_prev<0` 触发停止；不是可调阈值 |
| model | `qwen3:4b-q4_K_M` | `repro/local_config.json` | 2 | 本地 Ollama 模型 |
| model digest | `2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0` | 同上 | 3 | 固定模型版本 |
| quantization | `Q4_K_M` | 同上 | 5 | 模型量化 |
| temperature | 0.1 | 同上 | 6 | 采样温度 |
| context length | 4096 | 同上 | 7 | Ollama `num_ctx` |
| thinking mode | false | 同上 | 8 | Ollama `think` |
| random seed | 42 | 同上 | 9 | 每个模型请求的 Ollama seed |
| max output tokens | 2048 | 同上 | 10 | Ollama `num_predict` |
| request timeout | 180 s | 同上 | 16 | 本地请求超时配置 |
| keep_alive | `5m` | `repro/experiments/mimar_v02/run_controlled_positive_5users.py` | 41, `llm_select` | Ollama 模型驻留时间 |
| pilot cohort seed | 20260905 | `repro/experiments/pilot/freeze_pilot_manifest.py` | 28 | 10-user manifest 抽样 |
| controlled pool seed | 20260905 | `repro/experiments/pilot/run_catalog_grounded_ablation.py` | module constant / `candidate_pool` | 5-user选择与补足 100 项候选池 |
| paths per user | 2 | `repro/experiments/smr_mimar_g/run_controlled_positive_5users.py` | 53–56 | 最终受控实验重复数 |
| users | 419, 5021, 2677, 3113, 2249 | 同上 | 13 | 最终受控实验用户 |
| rating threshold | rating ≥ 4 | `repro/experiments/pilot/freeze_pilot_manifest.py` | 77–79, 174 | 正反馈定义 |
| eligibility | positive count > 20 | 同上 | 77–79 | 用户资格 |
| history ordering | pandas `sort_values(timestamp)` default sort | 同上 | 172–176 | 完整正反馈序列与 Last-20 的时序 |
| target sampling | `Random(20260905+user_id).randint(1,3952)` until valid and outside all positives | 同上 | 88–101 | 固定未正反馈 target |

## Deterministic ordering keys

- 长期兴趣：`(-L, genre)`，见 `routes.py:12`。
- target genre：先 `sorted(set(target_genres))`，见 `routes.py:19`。
- 全部可行路由：`(-RoutePrior, -L, interest, target_genre)`，见 `routes.py:28`。
- 多样性贪心：先最大 novelty（BOTH=2、单新维度=1、SCORE_FILL=0），同类沿用上述全局路由顺序，见 `routes.py:31–41`。
- 候选电影：MovieLens item ID 升序后截取前 20，见 `candidates.py:14–15`。

## 未使用的继承常量

`RECENT_WINDOW=20`、`ACTIVE_DECAY=0.95`、`ACCEPT_BOOST=0.15`、`REJECT_GENRE_DROP=0.05`、`ACTIVATION_THRESHOLD=0.05` 存在于共享的 MIMAR-v0.1 兴趣模块，但 SMR-MIMAR-G 只调用 `long_term_interests`，因此这些动态兴趣常量不参与本方法执行。

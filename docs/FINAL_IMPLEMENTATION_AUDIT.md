# SMR-MIMAR-G 最终实现规格

## 1. 数据与长期兴趣

最终受控实验从冻结 manifest 读取每位用户的**完整正反馈序列** `positive_movie_ids`，正反馈为 rating ≥ 4，并按 timestamp 升序排列。用户资格要求正反馈数 >20。用于早期 Static MI-Bridge 的 Last-20 仍保存在 manifest 中，但 SMR-MIMAR-G 不使用 Last-20 计算兴趣，也不存在滑动窗口。

目标由 `Random(20260905+user_id)` 在 MovieLens 合法 ID 中抽取，并排除该用户的**全部**正反馈，因此目标不会进入长期兴趣历史。时间截止点是数据中该用户最后一条已观察正反馈；代码没有使用截止点之后的额外交互。这里的 target 是合成的未正反馈目标，不是从未来交互切出的 held-out item。

设完整正反馈序列长度为 n，genre i 的出现位置集合为 `P_i`（1-based）：

- `Frequency(i)=|P_i|/n`。
- 把序列按 `min(3, floor(4(pos-1)/n))` 分到 4 个时间段；`Persistence(i)=出现过 i 的不同时间段数/4`。
- `Recency(i)=mean(pos/n, pos∈P_i)`；它是出现位置的平均归一化值，不是距当前时刻的指数衰减。
- `L(i)=clip_[0,1](0.5 Frequency + 0.3 Persistence + 0.2 Recency)`。

仅对历史中实际出现的 genre 建表；空历史直接抛出 `ValueError`，没有缺失值插补。genre 先字典序枚举，Top-5 再按 `(-L, genre)` 排序。

代码依据：`repro/methods/mimar_v01/interest_profile.py:12–40`、`repro/experiments/pilot/freeze_pilot_manifest.py:77–101,172–190`、`repro/experiments/smr_mimar_g/run_controlled_positive_5users.py:21–23`。

## 2. Genre co-occurrence

在完整 MovieLens `movies.dat` catalog 上，对每部电影的 genre 集合计数：

- `N(i)`：含 i 的电影数。
- `N(g)`：含 g 的电影数。
- `N(i,g)`：同时含 i 与 g 的电影数。
- `C(i,g)=N(i,g)/sqrt(N(i)N(g))`；任一边计数为 0 时为 0。

当 i=g 时，joint 循环给出 `N(i,i)=N(i)`，故 `C(i,i)=1`（只要 N(i)>0）。历史电影和 target 都属于 catalog，因此都参与全局统计；没有为用户进行排除。运行时在 planner 初始化时构建内存矩阵；实现另有 `save_cache_new()`，以独占创建方式写 vocabulary、边际计数和归一化矩阵，但最终 runner 未调用持久缓存。

代码依据：`repro/methods/mimar_v01/cooccurrence.py:7–36`。

## 3. RoutePrior 与可行性

对 Top-5 长期兴趣 i 和所有去重、字典序 target genre g 枚举路由：

`RoutePrior(i,g)=0.35 L(i)+0.30 C(i,g)`。

不再归一化或裁剪 RoutePrior。i≠g 时要求 `N(i,g)>0`；i=g 时只要求 `N(i)>0`。全局排序键为 `(-RoutePrior,-L,interest,target_genre)`。

实现中明确不存在 `S_t`、Need、Coverage、Commitment、route switching、SASRec score、IoI 或 IoR。SASRec 只在所有路径生成后运行。

代码依据：`repro/methods/smr_mimar_v01/routes.py:15–28`。

## 4. 静态多样性路由集 R*

忠实伪代码：

```text
remaining = globally_sorted_feasible_routes
chosen = []; seen_starts = {}; seen_targets = {}
repeat min(4, len(remaining)) times:
    for each route in remaining (preserving global rank):
        novelty = int(start unseen) + int(target unseen)
    choose maximum novelty; ties choose earliest global rank
    label BOTH / NEW_START_INTEREST / NEW_TARGET_ATTRIBUTE / SCORE_FILL
    append route; mark its start and target seen; remove it from remaining
```

因此精确优先级是 BOTH，然后“任一新维度”；NEW_START 与 NEW_TARGET 没有彼此固定优先级，而由 RoutePrior 全局顺序决定；最后 SCORE_FILL。RoutePrior 在多样性类别内决定先后，而非压过类别。可行路由不足 4 条时全部选取。多个路由可以共享 start，也可以共享 target。

真实例子：User 2677 的 R* 依次为 Action→Adventure（BOTH）、Thriller→Comedy（BOTH）、Drama→Children's（BOTH）、Romance→Comedy（NEW_START_INTEREST）。第四条允许复用 target genre Comedy。

代码依据：`repro/methods/smr_mimar_v01/routes.py:30–42`；实例来自冻结 generation JSON。

## 5. 候选支持、排序与硬排除

唯一候选层级为 `DIRECT`：

`SupportedRoutes(x)={(i,g)∈R*: i∈Genres(x) AND g∈Genres(x)}`。

只有 `SupportedRoutes(x)` 非空才进入候选；`RouteSupportCount=|SupportedRoutes(x)|`。该数量仅作为 prompt metadata，不参与排序或额外过滤。候选按原始 MovieLens item ID 升序，截取前 20；不足 20 时全部提供。一个多标签电影可支持多条路由。

每一步的排除集合精确为：完整正反馈历史 H0 的所有 item IDs ∪ 已接受中间项 IDs ∪ target ID。原历史项永不重新进入；已接受中间项永久排除；没有 Last-20。Guard 触发项在父 planner 中先临时加入，随后被 pop 且从 `used_ids` 移除；由于 planner 同步终止，它实际不会被再次选择。如果未来绕过停止状态继续调用，单看排除集合它可重新进入。

代码依据：`repro/methods/smr_mimar_v01/candidates.py:4–15`、`planner.py:10–17,29`、`repro/methods/smr_mimar_g/planner.py:19–23`。

## 6. LLM prompt 与配置

模型为 Ollama `qwen3:4b-q4_K_M`，digest `2bfd38a7...a114e0`，Q4_K_M，temperature 0.1，seed 42，`num_ctx=4096`，`num_predict=2048`，thinking=false，keep_alive=5m。

System prompt（逐字）：

```text
Select exactly one next movie from the supplied catalog-grounded candidates.
The routes are simultaneous migration directions, not a single route to follow.
A candidate may support multiple routes at the same time.
You do not need to select one route before selecting the movie.
Choose the candidate that forms the most natural next step in the overall migration toward the target.
ROUTE LABELS ARE GENRE CONCEPTS, NOT MOVIE TITLES.
The target item is reference only and must not be selected.
Output exactly one title copied verbatim from ALLOWED CANDIDATES.
Do not explain, number, rewrite, abbreviate, or output multiple titles.
```

User prompt 模板：

```text
{
  "demographics": <gender/age/occupation>,
  "frozen_long_term_multi_interest_profile": <Top-5 rows with L, score, frequency, persistence, recency, count>,
  "target_reference_only": <id/title/genres>,
  "all_target_genres": <target genres>,
  "static_selected_route_set": [<start_interest,target_attribute,route_prior>, ...],
  "current_generated_path": <accepted titles>
}

ALLOWED CANDIDATES:
[
  {"title": ..., "genres": ..., "supported_routes": ..., "route_support_count": ...}, ...
]

Your entire response must be exactly one title copied verbatim from the list above.
```

注意：planner 内部保存完整 L(i) 表，但 prompt 只传 Top-5 行；字段名“multi_interest_profile”不代表全 genre 表。

代码依据：`repro/methods/smr_mimar_v01/prompt.py:3–24`、`planner.py:23–25`、`repro/local_config.json:2–16`、runner 的 `llm_select`。

## 7. Parser 与 retry

先要求 raw 必须是非空字符串，再 `strip()` 首尾空白。若 stripped text 与 allowed title 完全相等，直接接受；否则依次尝试 `json.loads` 与 `ast.literal_eval`，只接受长度恰为 1、元素为字符串且精确属于 allowed set 的 list。tuple/dict/set/list 等其他容器拒绝；多项拒绝。没有大小写折叠、文章前后缀处理、模糊匹配、标题标准化或语义修复。裸标题可包含引号或换行，但只有其整体 stripped 文本恰好等于 catalog title 才能通过；带解释的多行输出不会通过。

每步最多总计 3 次生成（首次 + 2 retry）。retry 保持 planner state、候选集和基础 prompt 不变，只在 user prompt 末尾增加：

`INVALID OUTPUT. Copy exactly one movie title verbatim from the unchanged ALLOWED CANDIDATES.`

耗尽后状态为 `invalid_llm_selection_after_retries`，不加入任何新项。

代码依据：`repro/methods/smr_mimar_v01/prompt.py:13,24–34`、`planner.py:23–29`。

## 8. TargetOverlap guard

`TargetOverlap(x)=|set(Genres(x)) ∩ set(TargetGenres)| / |set(TargetGenres)|`。

分母是去重后的 target genre 数。单 genre target 的结果只能是 0 或 1；多 genre target 为覆盖比例。空 target genre 直接 `ValueError`，没有缺失值补救。第一项无比较直接通过；后续只在 `O_new < O_prev`（严格小于）时触发，所以相等允许。无阈值、无 SASRec、无 retry、不是用户拒绝。

```text
record = frozen_SMR.step(LLM)       # 暂时选择并加入
if no selected item: return
O_new = overlap(selected, target)
if first item: accept; O_prev = O_new
elif O_new < O_prev:
    remove selected item from accepted path and used IDs
    stop with target_overlap_decrease
else:
    accept; O_prev = O_new
```

触发项不在最终路径中；预定义 target 由 snapshot/runner 正规化追加。

代码依据：`repro/methods/smr_mimar_g/guard.py:2–10`、`planner.py:10–29`。

## 9. 停止条件与顺序

一次 `step()` 内顺序：

1. planner 已停止时抛出 `RuntimeError`（不是正常 stop reason）。
2. 已接受项数 ≥6：`max_intermediate_steps`。
3. 无合法候选：`no_legal_candidate`。
4. 三次输出均解析失败：`invalid_llm_selection_after_retries`。
5. 解析成功后，若 overlap 严格下降：回滚该项并设 `target_overlap_decrease`。
6. 否则接受；若此后达到 6 项，设 `max_intermediate_steps`，否则 `running`。

所有正常停止状态下，最终输出都是“已接受 intermediates + 预定义 target”。没有让 LLM 选择 target。

## 10. Formal-Evaluation-v1

- IoI：`ln(P_after+1e-12)-ln(P_before+1e-12)`。
- IoR：`R_before-R_after`。rank 是对 SASRec 全 3884 行（含 PAD、未 mask 历史）softmax 后降序 argsort 的 1-based 位置。
- target representation：target 不进入 extension；extension 是按原顺序解析出的全部非-target 中间项；target 的 raw MovieLens ID 再映射为 RecBole internal token ID。
- Proxy Acceptability：逐步对每个中间项，在 append 前计算 `sigmoid(sequence_embedding·item_embedding)`，再取均值。它不是真实点击、不是 softmax CTR，也没有用户反馈标签，因此不能称 CTR。
- Coherence：`intermediates + target` 的相邻对中，genre 交集非空记 1，否则 0，再取均值；不含“历史末项→路径首项”边。少于两项时 undefined/null。
- TargetLast：解析成功路径中，最后一个解析 item 是否为 target；聚合分母为 parsed paths，不限 formal-valid。
- HistoryReuse：已解析非-target 历史 item 出现次数 / `max(全部非-target path occurrences（含 unresolved）,1)`。
- PHRR：Formal protocol 没有独立 PHRR 公式；最终实验 runner 将其直接赋为 `HISTORY_REUSE_RATE`，故当前结果中 PHRR 是 HistoryReuseRate 的别名。

代码依据：`repro/evaluators/formal/metrics.py:8–39,58–74`、`protocol.py:14–50,56–91`、`coherence.py:8–20`、`repro/evaluators/sasrec/evaluator.py:39–78`、`repro/experiments/dynamic_no_bridge/run_controlled_5user.py:80–110`。

## 11. 最终实验配置

- 用户：419、5021、2677、3113、2249；每用户 2 路径；最多 6 intermediates。
- targets：419→Clean Slate (Coup de Torchon) (1981)；5021→Drunken Master (Zui quan) (1979)；2677→Jingle All the Way (1996)；3113→League of Their Own, A (1992)；2249→Timecop (1994)。
- 冻结 100-item pool 构造：依次加入固定 Last-20 history、target、Static MI-Bridge direct candidates（去重）；其余 catalog 用 `Random(20260905+user_id).shuffle` 后补足至 100。SMR 的每步候选仅从该池中、再按自身 direct-route 支持和硬排除筛选。
- Offline-positive：所有通过 parser、catalog 与 overlap guard 的项视为接受；guard 停止不是模拟拒绝。
- 模型配置见参数表。seed 42 在每次请求中重复传入，因此同样 prompt 产生了高度重复的两条路径；这不是独立随机 seed。
- SASRec 严格后验运行，不参与规划。

## 12. 实现与设计/README 差异

共记录 5 项：

1. **MINOR**：prompt 字段名暗示“完整长期多兴趣表”，实际只传 Top-5；完整 profile 仅保存在日志。
2. **MATERIAL**：方法描述曾允许“direct pool 太小时使用已定义的 broader support tiers”，但当前代码只有 DIRECT tier，没有阈值或 fallback tier。最终 README 明确写 direct-only，因此是“早期意图 vs 冻结实现”差异，而非 README/code 冲突。
3. **MINOR**：guard 文档描述为“加入前停止”，代码通过父类先加入、随后 pop/移除 used ID；最终路径与外部状态等价。
4. **MINOR**：guard-blocked item 从 `used_ids` 移除。由于状态立即终止，它不会在当前运行重入；若未来复用 planner 解除停止，该 item 不再受 duplicate exclusion。
5. **MINOR**：实验报告字段 PHRR 没有独立实现，实际直接复制 HistoryReuseRate；口头/书面报告应明确二者同值同定义。

README 与当前核心公式、静态路由、direct support 和 overlap guard 没有其他冲突。上述差异未在本审计中修复。

## 13. 结果边界

这是 5 用户、每人 2 路径的描述性受控预实验。固定 seed 导致每位用户两次路径相同，实际独立输出多样性有限；没有显著性检验，不能外推总体优势。最终均值：IoI 2.374565、IoR 416、Proxy 0.606791、Coherence 0.933333、mean path length 4.6（**包含 target**）、10/10 generation success。完整逐路径证据见 `FINAL_10_PATHS.md`。

## 审计附件

完整、冻结的审计材料同时保存在：

- `repro/results/smr_mimar_g/final_implementation_audit/FINAL_METHOD_SPEC.md`
- `repro/results/smr_mimar_g/final_implementation_audit/FINAL_PARAMETER_TABLE.md`
- `repro/results/smr_mimar_g/final_implementation_audit/FINAL_10_PATHS.md`
- `repro/results/smr_mimar_g/final_implementation_audit/FINAL_DEFENSE_CHEATSHEET.md`
- `repro/results/smr_mimar_g/final_implementation_audit/final_audit.json`

本发布过程未重新生成路径、未运行或修改 Formal-Evaluation-v1，也未修改冻结方法。

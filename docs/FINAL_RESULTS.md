# SMR-MIMAR-G：最终 10 条路径

所有内容直接摘自冻结 generation 与 Formal-Evaluation-v1 输出。每位用户的两次路径完全相同；仍分别列出，以保留 10-path 实验单位。`R*` 格式为 `start→target (RoutePrior, diversity reason)`；item 后列 `genres | SupportedRoutes | TargetOverlap`。

## User 419 — target: Clean Slate (Coup de Torchon) (1981) [Crime]

R*: Drama→Crime (0.252413, BOTH); Action→Crime (0.236622, NEW_START_INTEREST); Thriller→Crime (0.231861, NEW_START_INTEREST); Comedy→Crime (0.205327, NEW_START_INTEREST).

### Path 1

1. From Dusk Till Dawn (1996) — Action/Comedy/Crime/Horror/Thriller | Action→Crime, Thriller→Crime, Comedy→Crime | 1.0
2. Batman Forever (1995) — Action/Adventure/Comedy/Crime | Action→Crime, Comedy→Crime | 1.0
3. Best Men (1997) — Action/Comedy/Crime/Drama | Drama→Crime, Action→Crime, Comedy→Crime | 1.0
4. Twin Town (1997) — Comedy/Crime | Comedy→Crime | 1.0
5. Midnight in the Garden of Good and Evil (1997) — Comedy/Crime/Drama/Mystery | Drama→Crime, Comedy→Crime | 1.0
6. Hana-bi (1997) — Comedy/Crime/Drama | Drama→Crime, Comedy→Crime | 1.0
7. **Target:** Clean Slate (Coup de Torchon) (1981)

Guard: no. Stop: `max_intermediate_steps`. IoI=4.414846; IoR=413.

### Path 2

与 Path 1 完全相同。Guard: no. Stop: `max_intermediate_steps`. IoI=4.414846; IoR=413.

## User 5021 — target: Drunken Master (Zui quan) (1979) [Action, Comedy]

R*: Action→Action (0.476637, BOTH); Drama→Comedy (0.265113, BOTH); Sci-Fi→Action (0.287120, NEW_START_INTEREST); Thriller→Action (0.271069, NEW_START_INTEREST).

### Path 1

1. Star Wars: Episode V - The Empire Strikes Back (1980) — Action/Adventure/Drama/Sci-Fi/War | Action→Action, Sci-Fi→Action | 0.5
2. Star Wars: Episode VI - Return of the Jedi (1983) — Action/Adventure/Romance/Sci-Fi/War | Action→Action, Sci-Fi→Action | 0.5
3. Transformers: The Movie, The (1986) — Action/Animation/Children's/Sci-Fi/Thriller/War | Action→Action, Sci-Fi→Action, Thriller→Action | 0.5
4. Saving Private Ryan (1998) — Action/Drama/War | Action→Action | 0.5
5. Soldier (1998) — Action/Adventure/Sci-Fi/Thriller/War | Action→Action, Sci-Fi→Action, Thriller→Action | 0.5
6. Mars Attacks! (1996) — Action/Comedy/Sci-Fi/War | Action→Action, Sci-Fi→Action | 1.0
7. **Target:** Drunken Master (Zui quan) (1979)

Guard: no. Stop: `max_intermediate_steps`. IoI=-0.425712; IoR=-508.

### Path 2

与 Path 1 完全相同。Guard: no. Stop: `max_intermediate_steps`. IoI=-0.425712; IoR=-508.

## User 2677 — target: Jingle All the Way (1996) [Adventure, Children's, Comedy]

R*: Action→Adventure (0.319096, BOTH); Thriller→Comedy (0.285541, BOTH); Drama→Children's (0.208054, BOTH); Romance→Comedy (0.261814, NEW_START_INTEREST).

### Path 1

1. Charade (1963) — Comedy/Mystery/Romance/Thriller | Thriller→Comedy, Romance→Comedy | 0.333333
2. Pyromaniac's Love Story, A (1995) — Comedy/Romance | Romance→Comedy | 0.333333
3. Free Willy 2: The Adventure Home (1995) — Adventure/Children's/Drama | Drama→Children's | 0.666667
4. **Target:** Jingle All the Way (1996)

Guard at attempted step 4: Runaway Train (1985), Action/Adventure/Drama/Thriller, overlap 0.333333 < 0.666667；该项未进入路径。Stop: `target_overlap_decrease`. IoI=4.377270; IoR=1450.

### Path 2

与 Path 1 完全相同，包括同一 guard event。IoI=4.377270; IoR=1450.

## User 3113 — target: League of Their Own, A (1992) [Comedy, Drama]

R*: Drama→Drama (0.508444, BOTH); Comedy→Comedy (0.490531, BOTH); Romance→Comedy (0.289100, NEW_START_INTEREST); Action→Drama (0.232067, NEW_START_INTEREST).

### Path 1

1. Waiting to Exhale (1995) — Comedy/Drama | Drama→Drama, Comedy→Comedy | 1.0
2. **Target:** League of Their Own, A (1992)

Guard: no. Stop: `invalid_llm_selection_after_retries` at next step. IoI=1.551748; IoR=93.

### Path 2

与 Path 1 完全相同。Guard: no. Stop: `invalid_llm_selection_after_retries`. IoI=1.551748; IoR=93.

## User 2249 — target: Timecop (1994) [Action, Sci-Fi]

R*: Action→Action (0.484255, BOTH); Thriller→Sci-Fi (0.241916, BOTH); Comedy→Action (0.236720, NEW_START_INTEREST); Drama→Action (0.221097, NEW_START_INTEREST).

### Path 1

1. Get Shorty (1995) — Action/Comedy/Drama | Action→Action, Comedy→Action, Drama→Action | 0.5
2. Tank Girl (1995) — Action/Comedy/Musical/Sci-Fi | Action→Action, Comedy→Action | 1.0
3. **Target:** Timecop (1994)

Guard at attempted step 3: Batman Forever (1995), Action/Adventure/Comedy/Crime, overlap 0.5 < 1.0；该项未进入路径。Stop: `target_overlap_decrease`. IoI=1.954671; IoR=632.

### Path 2

与 Path 1 完全相同，包括同一 guard event。IoI=1.954671; IoR=632.

## 方法对照表

“Mean Path Length”按各方法保存的 normalized path 长度统计，包含 target。generation success 表示至少生成一个 intermediate；数据缺失时不推断。

| Method | IoI | IoR | Proxy | Coherence | Mean Path Length | Generation Success |
|---|---:|---:|---:|---:|---:|---:|
| Static MI-Bridge | 0.657813 | 25.5 | 0.884113 | 0.965278 | 9.5 | 10/10 |
| Dynamic-No-Bridge | 0.654154 | 267.6 | 0.866865 | 0.850000 | 9.0 | 10/10 |
| MIMAR-v0.1 | 1.882053 | 191.6 | 0.805558 | 1.000000 | 6.4 | 8/10 |
| MIMAR-v0.2 | 0.676167 | 90.9 | 0.766642 | 0.920000 | 5.6 | 10/10 |
| MIMAR-v0.3 | 0.231917 | -88.0 | 0.783643 | 0.966667 | 5.8 | 10/10 |
| SMR-MIMAR-v0.1 | 1.349992 | -166.9 | 0.592039 | 0.933333 | 6.0 | 10/10 |
| **SMR-MIMAR-G** | **2.374565** | **416.0** | **0.606791** | **0.933333** | **4.6** | **10/10** |

来源：各冻结 `summary.json` 与 generation JSON。表仅作 5-user 描述性比较；不同方法的 stopping、候选筛选和路径长度不同，不构成统计显著性或总体优越性证明。

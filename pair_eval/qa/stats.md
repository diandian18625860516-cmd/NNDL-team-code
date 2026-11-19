# D 交付统计 (pair generation v1.1.0 · 用途划分 = B 官方清单 v2)

## 配对 (pair) 数量

| 类别 | 用途 | A 对 (重摆/移格 vs 违规) | B 对 (替换/缺件) | 合计 |
|---|---|---|---|---|
| pushpins | development | 50 | 0 | 50 
| pushpins | final_diagnostic | 46 | 0 | 46 
| pushpins | train | 213 | 81 | 294 
| screw_bag | development | 10 | 0 | 10 
| screw_bag | final_diagnostic | 15 | 0 | 15 
| screw_bag | train | 56 | 14 | 70 
| **合计** | | | | **485** |

## 源图 (source) 状态

| 类别 | 用途 | 可用 | 失败/未标注 |
|---|---|---|---|
| pushpins | development | 52 | 4 |
| pushpins | final_diagnostic | 50 | 6 |
| pushpins | train | 230 | 30 |
| screw_bag | development | 10 | 45 |
| screw_bag | final_diagnostic | 15 | 39 |
| screw_bag | train | 56 | 195 |

- 自动校验 (QA)：485 对中 485 对全部通过，0 对存在未通过项（见 qa_report.json）。
- 审核状态：全部为 pending，等待 B 验收后更新为 approved/rejected。
- 统计单位：同时报告源图数与编辑对数；一张源图最多派生 2 对（A/B），不把变体当作独立源图计数。
- 可用源图与实际对数差额：pushpins 332 张可用源图中 23 张 A 对生成失败
  （13 crosscell_infeasible + 10 incell_infeasible，重摆无可行解）；screw_bag train
  可用源图中 5 张 B 对生成失败（substitution_infeasible）。逐张明细见
  `source_status.json` 的 `pair_failures` 字段（对应 `stats.json` 的 `pairs_failed_A/B`）。
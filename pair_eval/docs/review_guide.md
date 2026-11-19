# B 审核指南（验收流程与判定要点）

本包 {TOTAL_PAIRS} 对的 `review_status` 全部为 **pending**。按约定：审核未通过对计数但不进入
评分；全部未通过时不输出性能指标。以下流程让验收尽量快、且可直接落到机器可读字段。

## 1. 工具

- `qa/contact_sheets/{category}_{purpose}_pageNN.jpg`：每页 6 对，
  行=pair，列=original / VALID / INVALID（彩色标签：红=长螺丝、橙=短螺丝、蓝=螺母、
  绿=垫圈；pushpins 红=图钉、紫=隔板）。
- `review_workbook.csv`：每对一行，含 self_check 结果与待填的
  `review_status / review_reason_B / reviewer / date`。
- `qa/qa_report.json`：程序校验明细（当前 0 失败）。

## 2. 建议顺序（先诊断、后训练）

1. `pushpins_final_diagnostic`（{n_pairs("pushpins", "final_diagnostic")} 对，配对诊断主力，重点看跨格侧）
2. `pushpins_train/development` + `screw_bag`（先 A 对后 B 对）
3. 失败/待补清单（`source_status.json`）最后统一处理。

## 3. 每种编辑的验收要点

| 编辑 | 必须满足 | 常见误检 |
|---|---|---|
| SB 合法重摆 | 移动的件仍完整（形状无残缺）、未叠到其他件上、未越出袋内界 | 件被"剪切"出画布/压印 |
| SB 删除（A 违规） | 恰好少一个必需件、其余件像素未被连带删除（尤其叠压处） | 删除时误删叠压伙伴 |
| SB 替换（B 违规） | 总数仍 6、出现两根长螺丝、无短螺丝；原长螺丝未移动 | 副本与原长螺丝重叠/出界 |
| PP 格内移动 | 钉完整、质心仍在原格 | 钉被格壁"切掉"一角（允许 ≤10% 覆盖，见 wall_overlap_px） |
| PP 跨格移动 | 一格空、相邻格两钉、总数 15 | 两钉重叠、或钉未真正进入目标格 |
| PP 删除（B 违规） | 总数 14、一格空 | — |

逐对的平移/旋转/距离/面积参数都在 `pair_manifest.json` 的 `transform_parameters`
中，可抽查与缩略图相互印证；`edit_region` 给出全分辨率坐标下的编辑区域。

## 4. 状态回写

- 对通过的对：在 `review_workbook.csv` 填 `approved`（可另存机器可读版本：
  pair_id→approved/rejected 的 JSON 由 C/E 按需读取；本包不改写 `pair_manifest.json`
  的 review_status，保持单一事实来源待 B 定稿后统一更新）。
- 拒绝时务必写 `review_reason_B`（如"垫圈被剪切"），D 会按原因修复并重出该子集。
- 统计口径：报告通过/修改/淘汰数量与原因分布（`qa/stats.md` 已给当前基线）。

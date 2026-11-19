# D 部分交付包 —— 合法/违规配对总清单 v2.1（按 B 官方划分对齐）

按 C 负责人发布的《B/D/E → C 最小交付约定 v1》整理。v2 的唯一驱动变更：**用途分配改
为逐图采用 B 交付的官方划分清单**（`generator/official_split/*.csv`，逐图交错、覆盖全部
train/good），替代 v1 自定的连续区段方案；本包其余口径（表示/规则/编辑操作/QA）与 v1 一致。
**配对总清单 + 实际生成的编辑文件 + 可复现的生成工具 + QA 证据**，供 B 验收、E 接入、C 复核。

## 1. 交付物一览

| 文件 | 说明 |
|---|---|
| `pair_manifest.json` | **核心交付**：合法/违规配对总清单（对象数组，含全部用途 train/development/final_diagnostic） |
| `source_status.json` | 源图用途分配表（先固定用途、后生成变体；含失败/待补源图及原因） |
| `data/composition_maps/` | 各源图的组成图（256×256，单通道整数标签 PNG） |
| `data/edits/pushpins/valid`、`data/edits/pushpins/invalid` | 每对编辑产物（合法侧/违规侧，256×256） |
| `qa/qa_report.json` | 每对自动校验明细（尺寸/标签域/类计数/文件存在性，全部通过=0 失败） |
| `qa/stats.md`、`qa/stats.json` | 源图数与编辑对数统计（同时报告两个单位，不把变体当独立源图） |
| `qa/contact_sheets/` | B 验收用的分页缩略图（每页 6 对：original/VALID/INVALID 三联） |
| `qa/triplets/` | 方法文档用三联/四联图（RGB 原图 + 组成图 original/VALID/INVALID） |
| `review_workbook.csv` | B 的审核工作簿（每对一行，含 self_check 结果与待填字段） |
| `r1_r2_sample_list.json` | 排序控制实验（R1 vs R2）的**固定违规样本清单**（v2 按新清单重生成） |
| `integration/run_template_pair_eval.json` | E 的配对运行 JSON 模板（含 model_seen_source_paths 等必填项） |
| `generator/` | 可复现生成工具（组成图提取 + 配对生成 + 校验 + VLM 标注脚本） |
| `generator/official_split/` | **B 交付的官方划分清单副本**（本包用途分配的唯一依据） |
| `docs/`（本目录） | 方法说明、规则依据、目标定义、E 接入指南、B 审核指南 |

## 2. 数量总览

配对数量（A 对 = 基本合法/违规对照；B 对 = 训练用途的扩展违规类型）：

| 类别 | 用途 | A 对 | B 对 | 合计 |
|---|---|---|---|---|
| pushpins | development | 50 | 0 | 50 |
| pushpins | final_diagnostic | 46 | 0 | 46 |
| pushpins | train | 213 | 81 | 294 |
| screw_bag | development | 10 | 0 | 10 |
| screw_bag | final_diagnostic | 15 | 0 | 15 |
| screw_bag | train | 56 | 14 | 70 |
| **合计** | | | | **485** |

源图状态：

| 类别 | 用途 | 可用 | 失败/待补 |
|---|---|---|---|
| pushpins | development | 52 | 4 |
| pushpins | final_diagnostic | 50 | 6 |
| pushpins | train | 230 | 30 |
| screw_bag | development | 10 | 45 |
| screw_bag | final_diagnostic | 15 | 39 |
| screw_bag | train | 56 | 195 |

自动校验：485 对中 **485 对全部通过**（0 项失败）。
审核状态：**全部 `pending`，等待 B 验收**；审核未通过对计数但不进入评分（按约定）。

可用源图与实际对数的差额（如实披露）：pushpins 332 张可用源图中 **23 张的 A 对生成失败**
（13 张 crosscell_infeasible、10 张 incell_infeasible，即跨格/格内重摆无可行解），
screw_bag train 可用源图中 **5 张的 B 对生成失败**（substitution_infeasible，替换无可行解）。
这些源图本身可用、用途不变，仅未产出对应编辑对；逐张明细见 `source_status.json` 的
`pair_failures` 字段与 `qa/stats.json` 的 `pairs_failed_A/B` 计数。

## 3. 表示与标签约定

- `representation = composition_map`（全包一致，同一评价不混用表示）。
- 文件格式：256×256、PIL 'L' 模式（单通道）、整数标签，取值 0..5，与 SALAD 组成分支
  `read_mask(num_cls=6)` 的输入约定兼容。
- **标签方案（D 自建，随包文档化）**：
  - screw_bag：`0=背景/袋体`、`1=长螺丝`、`2=短螺丝`、`3=螺母`、`4=垫圈`
  - pushpins：`0=背景`、`1=图钉`、`2=隔板/格壁`
  - 与 SALAD 作者组成图的类别索引**不保证一致**（作者图为聚类索引，如 screw_bag 的 {0,1,2,5}）。
    配对诊断在**同一清单、同一标签方案**内进行内部对照，因此不影响配对评价的内部有效性；
    但不得把本包组成图与作者组成图混入同一次评价。
- 路径约定：`source_path` 相对 LOCO 数据根；`original_path`/`valid_path`/`invalid_path`
  相对本交付包根（即 `pair_manifest.json` 所在目录）。
- 每对在 `file_sha256` 中记录三个文件的 SHA-256，C 可直接核对。

## 4. 用途分配（先固定、后生成；v2 = B 官方划分）

分配策略在生成任何变体之前固定（`generator/generate_deliverable.py` 的
`load_official_split()`），**逐图采用 B 交付的官方划分清单**（副本随包：
`generator/official_split/pushpins_split_manifest.csv`、`screw_bag_split_manifest.csv`，
每图一列，标记 train_edit_source / dev_source / final_diag_source，覆盖全部
train/good 图；同一源图及其所有变体只留在同一用途）：

| 类别 | train（train_edit_source） | development（dev_source） | final_diagnostic（final_diag_source） |
|---|---|---|---|
| screw_bag | train/good 逐图交错，共 251 张 | 55 张 | 54 张 |
| pushpins | train/good 逐图交错，共 260 张 | 56 张 | 56 张 |

- v1 自定的连续区段方案（train 000–239 / dev 240–299 / FD validation 000–059）已废弃；
  v1 基于 validation/good 的 45 个 FD 对不再保留（B 清单未给 validation/good 任何角色）。
- 全部配对源图都在 train/good（即 SALAD 官方 train 划分）内：E 仍需按
  `docs/integration_E.md` 披露起点检查点见过的源图（官方训练同时使用 validation 划分）。
- 诊断用途对不参与任何训练/调参（排序参数 m 只在 development 对上选定）。

## 5. 清单字段（与 C 的接口逐项对应）

每个对象含：`pair_id`、`source_id`、`source_path`、`category`、`source_split`、
`original_path`、`valid_path`、`invalid_path`、`transform_parameters`（非空对象，
含两侧实际操作、平移/旋转参数、距离、面积、编辑前后部件计数等控制量）、
`edit_region`（编辑部件/区域，结构化）、`rule_reference`（两侧各自的规则依据，
指向官方缺陷类型与正常材料依据，不用"分数低"作依据）、`review_status`（当前全部
`pending`，附 `review_reason=null`）、`representation`、`purpose`，另附
`generator`（版本与确定性说明）与 `file_sha256`（三文件哈希）。

## 6. 编辑产物唯一性

- 每个编辑文件以 `pair_id` 命名，一对一归属，**没有多对复用同一编辑产物、也没有编辑
  产物冒作原图**的情况；原图文件（`composition_maps/`）即源图组成图本身。
- 变体不冒充 LOCO 官方异常：E 的预测 CSV 中这些输入只按本清单的 `pair_id`/路径出现，
  不标成官方 structural/logical 异常。

## 7. 已知缺口与补齐方式（如实披露）

1. **screw_bag 覆盖受 VLM 标注进度限制**：B 清单要求 360 张 train/good 全部有角色，
   当前已标注 96 张（通过组成门限 81 张，通过率
   84%），其余 264 张标注待续（接口限流，
   已在 `source_status.json` 标为 `vlm_annotation_pending`）。补齐步骤（可断点续跑）：
   ① 运行 `generator/vlm_annotate.py --budget 3600`（v2 起只标注 train/good）；
   ② 删除生成脚本的 `records_cache.pkl` 缓存（`generate_deliverable.py` 内 `CACHE`
   常量路径）后重跑 `generator/generate_deliverable.py`；
   ③ 重跑 `generator/build_auxiliary.py` 与 `generator/write_docs.py`。清单与统计自动更新。
2. **screw_bag 组成图为 D 自建**（规则+VLM 辅助定位，见 `docs/method.md`），非 SALAD 作者图。
   这一点已按"同一表示内对照"原则处理并在 §3 披露。
3. pushpins 有 40 张源图提取失败（网格粘连/亮度异常），已列在 `source_status.json`
   并附原因；不影响任何已通过对的合法性。validation/good 在 B 清单下无角色，不参与本包。
4. 视觉抽验：本包的自动校验为程序级（计数/标签/哈希）；**图像语义层面的最终把关就是
   B 的验收流程本身**（`docs/review_guide.md`）。分批补齐与复验均按同一流程执行。

## 8. 变更记录（由 B 官方划分清单驱动）

### v2 → v2.1（本次，纯披露补充，数据与清单零变更）

- §2 新增"可用源图与实际对数差额"披露：pushpins 23 张 A 对、screw_bag 5 张 B 对
  生成约束不可行的明细（数据本就记录在 `source_status.json` 的 `pair_failures` 字段
  与 `qa/stats.json` 的 `pairs_failed_A/B`，本次补入文字说明以便审阅者复算）。
- §7.1 补齐步骤中 `records_cache.pkl` 的路径口径更正为"脚本内 `CACHE` 常量路径"。
- 配对清单、编辑产物、QA、哈希均与 v2 逐字节一致（无任何数据变更）。

### v1 → v2

1. **用途分配改源**：v1 按 D 自定连续区段 + validation/good 做 FD；v2 逐图采用 B 的
   官方划分清单（交错式，FD 源图也在 train/good 内），两份清单副本随包发布，
   `source_status.json` 覆盖 B 清单的全部源图（含待标注/失败态）。
2. **移除项**：v1 基于 validation/good 的 45 个 pushpins FD 对不再保留（B 清单未给
   validation/good 任何角色）；组成图目录中不再出现 validation/ 子目录。
3. **重生成**：全部 pair_id 按新用途重新分配（如 pushpins 001 现为 FD 源图，其 A 对为
   PP-FD-001-A）；编辑内容（操作/参数/规则）由同一生成器 v1.1.0 重出，确定性与哈希
   校验口径不变。R1/R2 清单、审核工作簿、contact sheets 全部重生成。
4. **未变更项**：表示类型（composition_map）、标签方案、编辑操作定义、规则依据、
   QA 校验口径、审核状态（全部 pending）与《B/D/E → C 交付约定》的字段契约。

## 9. 下一步（对接）

- **B**：按 `docs/review_guide.md` 验收，更新 `review_workbook.csv` 与清单内 `review_status`。
- **E**：按 `docs/integration_E.md` 接入；R1/R2 使用 `r1_r2_sample_list.json` 的固定样本。
- **C**：按清单核对源图隔离（train/dev/FD 无交叉、无编辑产物复用）后接收 E 的同输入分数。

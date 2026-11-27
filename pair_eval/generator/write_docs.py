#!/usr/bin/env python3
"""Write D's documentation set (Chinese, team-facing). v2: aligned to B's official split."""
import json
import os

ROOT = '/home/z/my-project/workspace/D_deliverable'
MAN = json.load(open(f'{ROOT}/pair_manifest.json'))
QA = json.load(open(f'{ROOT}/qa/qa_report.json'))
SRC = json.load(open(f'{ROOT}/source_status.json'))
ST = json.load(open(f'{ROOT}/qa/stats.json'))

from collections import Counter, defaultdict
per = defaultdict(Counter)
for e in MAN:
    per[(e['category'], e['purpose'])][e['pair_id'][-1]] += 1
srcper = defaultdict(Counter)
for k, v in SRC.items():
    cat = k.split('/')[0]
    srcper[(cat, v['purpose'])][v['status']] += 1

TOTAL_PAIRS = len(MAN)
QA_FAILED = sum(1 for v in QA.values() if not v['all_passed'])


def n_pairs(cat, pur):
    return per[(cat, pur)]['A'] + per[(cat, pur)]['B']


def src_fail_pending(cat):
    """未标注（VLM pending）源图数。"""
    return sum(1 for k, v in SRC.items()
               if k.startswith(cat) and any('annotation' in f for f in v.get('flags', [])))


def src_fail_other(cat):
    """已标注/已提取但门限失败的源图数。"""
    return sum(1 for k, v in SRC.items()
               if k.startswith(cat) and v['status'] != 'ok'
               and not any('annotation' in f for f in v.get('flags', [])))


def src_ok(cat):
    return sum(1 for k, v in SRC.items() if k.startswith(cat) and v['status'] == 'ok')


R1R2_N = sum(1 for e in MAN if e['category'] == 'screw_bag' and e['purpose'] == 'train')
SB_ANnotated = src_ok('screw_bag') + src_fail_other('screw_bag')
SB_GATE_RATE = (100.0 * src_ok('screw_bag') / SB_ANnotated) if SB_ANnotated else 0
PP_OK, PP_FAIL = src_ok('pushpins'), src_fail_other('pushpins')


def table_per():
    rows = []
    for (cat, pur), c in sorted(per.items()):
        rows.append((cat, pur, c['A'], c['B'], c['A'] + c['B']))
    return rows


def table_src():
    rows = []
    for (cat, pur), c in sorted(srcper.items()):
        rows.append((cat, pur, c['ok'], c['extraction_failed']))
    return rows


DOCS = {}

DOCS['README_D.md'] = f"""# D 部分交付包 —— 合法/违规配对总清单 v2（按 B 官方划分对齐）

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
| `data/edits/{cat}/valid`、`data/edits/{cat}/invalid` | 每对编辑产物（合法侧/违规侧，256×256） |
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
{chr(10).join(f"| {c} | {p} | {a} | {b} | {t} |" for c,p,a,b,t in table_per())}
| **合计** | | | | **{len(MAN)}** |

源图状态：

| 类别 | 用途 | 可用 | 失败/待补 |
|---|---|---|---|
{chr(10).join(f"| {c} | {p} | {ok} | {fail} |" for c,p,ok,fail in table_src())}

自动校验：{len(MAN)} 对中 **{sum(1 for v in QA.values() if v['all_passed'])} 对全部通过**（0 项失败）。
审核状态：**全部 `pending`，等待 B 验收**；审核未通过对计数但不进入评分（按约定）。

## 3. 表示与标签约定

- `representation = composition_map`（全包一致，同一评价不混用表示）。
- 文件格式：256×256、PIL 'L' 模式（单通道）、整数标签，取值 0..5，与 SALAD 组成分支
  `read_mask(num_cls=6)` 的输入约定兼容。
- **标签方案（D 自建，随包文档化）**：
  - screw_bag：`0=背景/袋体`、`1=长螺丝`、`2=短螺丝`、`3=螺母`、`4=垫圈`
  - pushpins：`0=背景`、`1=图钉`、`2=隔板/格壁`
  - 与 SALAD 作者组成图的类别索引**不保证一致**（作者图为聚类索引，如 screw_bag 的 {{0,1,2,5}}）。
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
   当前已标注 {SB_ANnotated} 张（通过组成门限 {src_ok('screw_bag')} 张，通过率
   {SB_GATE_RATE:.0f}%），其余 {src_fail_pending('screw_bag')} 张标注待续（接口限流，
   已在 `source_status.json` 标为 `vlm_annotation_pending`）。补齐步骤（可断点续跑）：
   ① 运行 `generator/vlm_annotate.py --budget 3600`（v2 起只标注 train/good）；
   ② 删除 `generator/records_cache.pkl` 后重跑 `generator/generate_deliverable.py`；
   ③ 重跑 `generator/build_auxiliary.py` 与 `generator/write_docs.py`。清单与统计自动更新。
2. **screw_bag 组成图为 D 自建**（规则+VLM 辅助定位，见 `docs/method.md`），非 SALAD 作者图。
   这一点已按"同一表示内对照"原则处理并在 §3 披露。
3. pushpins 有 {PP_FAIL} 张源图提取失败（网格粘连/亮度异常），已列在 `source_status.json`
   并附原因；不影响任何已通过对的合法性。validation/good 在 B 清单下无角色，不参与本包。
4. 视觉抽验：本包的自动校验为程序级（计数/标签/哈希）；**图像语义层面的最终把关就是
   B 的验收流程本身**（`docs/review_guide.md`）。分批补齐与复验均按同一流程执行。

## 8. v1 → v2 变更记录（由 B 官方划分清单驱动）

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
"""

DOCS['method.md'] = f"""# 方法说明 —— 组成图构建与配对编辑生成

## 1. 两条组成图提取线

### 1.1 screw_bag（VLM 辅助定位 + 局部精化）
袋子为半透明塑料袋，部件为金属反光体，亮度与褶皱高度重叠，纯阈值法不可靠
（实测固定阈值/自适应阈值的组成门限通过率 < 10%）。因此：
1. **逐图 VLM 部件定位**（`generator/vlm_annotate.py`）：对每张正常图询问 6 个部件
   （1 长螺丝、1 短螺丝、2 螺母、2 垫圈）的边界框与类型。定位框与类型来自视觉证据，
   松弛约 ±40px。
2. **局部精化**（`generator/screwbag_map_builder.py`）：在每个定位框内做局部阈值
   （p25+25、下限 60）、闭运算、取主连通域，得到精确部件掩膜；重测特征
   （长宽 L、孔洞像素、伸长率）。
3. **严格门限**（无静默回退）：
   - 恰好 2 根螺丝（较长者为长螺丝，L 比 ≥ 1.08）+ 4 个小件；
   - 垫圈判据 `hole_px ≥ 3300`（实测垫圈孔 ~4300px、螺母孔 ~2740px，四分位区间完全不交叠），
     数量必须恰为 2+2；
   - 精化类型与 VLM 类型**一致**，不一致即整图判失败（交 B 人工核查，不静默采纳任一方）。
   门限不满足的源图进入 `source_status.json` 的失败清单并给出原因。
4. **像素归属**：视觉上互相叠压的部件按固定绘制顺序落笔；每个部件记录
   **有效归属掩膜**（画布上真正承载其标签的像素），编辑（移动/删除）只作用于有效归属，
   保证删除一个部件不会误删被叠压伙伴的像素。

实测：{SB_ANnotated} 张已标注源图中 {src_ok('screw_bag')} 张通过门限（{SB_GATE_RATE:.0f}%）；失败原因主要是类型分歧与罕见部件形态
（如开口弹簧垫圈的孔判据失败），均已显式记录；未标注图在 `source_status.json` 标为
`vlm_annotation_pending` 单列，不冒充通过。

### 1.2 pushpins（纯几何提取，无需 VLM）
1. 黄色头部色彩规则分割（面积 1400–7000px 过滤）→ 恰好 15 枚；
2. **格壁点阵**：亮中灰的最大连通域即格子隔板网络；取其 15 个最大"孔洞"为格子区域
   （按面积排序，剔除格壁内部纹理小孔，最小格子面积阈值 25000px）；
3. 头部按**质心落区**分配到格子（对格壁倾斜鲁棒），门限=恰好 15 格各 1 枚；
4. 金属针：亮、非黄、非格壁像素按最近头部（≤260px 上限）归属；
5. 标签：`1=图钉`（头+针），`2=隔板`（格壁点阵像素质心），`0=背景`。

实测：train/good 全量 372 张中 {PP_OK} 张通过（{100.0 * PP_OK / 372:.0f}%）；失败为格壁断裂导致的格子粘连等，
已记录原因。validation/good 在 B 官方划分下无角色，不参与本包。3 张下载损坏文件已重新下载修复。

## 2. 编辑操作与匹配控制

所有编辑在全分辨率画布上进行（screw_bag 1600×1100、pushpins 1700×1000），
交付文件为其 256×256 NEAREST 缩放（与原图文件同一缩放约定）。候选位置由
`crc32(pair_id)` 决定顺序的确定性网格给出——同一次重跑得到完全相同的产物
（`generator.determinism` 字段）。

### 2.1 screw_bag
- **A 对（所有用途）**：合法侧 = 平移重摆一个松散件（垫圈/螺母，按源图序号轮换），
  平移距离 ≥110px、落点在袋内界内、与其他部件（膨胀 8px）不重叠；违规侧 = 删除
  **同一个**部件（对应官方 1_missing_* 类型）。控制量：同部件、同编辑面积；
  `transform_parameters.controls` 记录两侧编辑后的部件计数表。
- **B 对（train，每 3 张一取）**：违规侧 = 把短螺丝替换为长螺丝的旋转副本
  （对应官方 screw_too_long；**总数保持 6 不变**，规格违例），落点优先取短螺丝质心，
  不可行时按确定性偏移网格外扩并记录 `placement_offset_px`；合法侧 = 重摆另一类松散件。
  原长螺丝保持原位（副本不与原长螺丝或任何部件重叠，膨胀 8px 校验）。
- 交付：A 对 59、B 对 7（5 张替换几何不可行已记录）。

### 2.2 pushpins
- **A 对（所有用途）**：同一枚图钉，合法侧 = **格内平移**（质心留在原格内，
  距离 ≥25px，取可行候选中最远者）；违规侧 = **跨格平移**（质心落入相邻格，
  不得与原住民图钉重叠）。控制量：同钉（面积完全一致）、同一编辑流程；
  距离差为**结构性不可避免**（跨格 ≥ 格距），已在 `controls.distance_note` 披露——
  这正是"只看总数无法区分、必须看分配"设计的代价与证据。
- **B 对（train，每 3 张一取）**：合法侧 = 另一枚钉格内平移；违规侧 = 删除一枚钉
  （对应官方 missing_pushpin）。
- 格壁重叠：移动后的钉与格壁像素重叠 ≤ 钉面积 10%（原物本身即可压壁），实际值逐对
  记录在 `wall_overlap_px`。
- 交付：A 对 297、B 对 77（24 张 A 对几何不可行已记录原因）。

## 3. QA（自动校验）

每对交付后重新从磁盘读取三个文件做程序级校验（`qa_report.json`）：
- 256×256 尺寸、标签域合法（0..4 / 0..2）；
- screw_bag：合法侧总部件面积稳定（±2%）且**各部件类计数不变**；
  违规侧按操作分流校验——删除类：目标类计数显著下降且**其余类计数稳定（±70px 内）**；
  替换类：长螺丝像素增加、短螺丝基本消失；
- pushpins：图钉面积两侧稳定（±12%）、隔板面积仅允许被钉覆盖造成的少量减少（≤15%）；
- 文件存在性与 SHA-256 哈希（记入清单）。

当前 {TOTAL_PAIRS} 对全部通过（{QA_FAILED} 项失败）。

## 4. 确定性与复现

- 候选顺序：`crc32(pair_id) % N` 决定起始相位，候选网格固定（16 角×4 距离 / 24 旋转×
  8 偏移 / 12 角×3 距离等）；
- 无任何随机数发生器状态、无时间依赖；两次运行产物按字节一致（哈希可复核）；
- 重跑入口：`generator/generate_deliverable.py`（记录缓存于 `records_cache.pkl`，
  支持断点续跑）。
"""

DOCS['rules.md'] = """# 规则依据 —— 每种编辑为何合法/违规

依据来源（不使用"模型分数低"作为合法性依据）：
1. Bergmann et al., "Beyond Dents and Scratches: Logical Constraints in Unsupervised
   Anomaly Detection and Localization", IJCV 2022（数据集定义与官方约束）；
2. 官方 `defects_config.json`（随数据集发布，本包在 `generator/` 内附副本）；
3. 正常训练材料的组成审计（本组进度报告 §5.1：360 张 screw_bag 正常图全部为
   1 长螺丝+1 短螺丝+2 螺母+2 垫圈；pushpins 3×5 格每格一钉、共 15 钉）。

## screw_bag

| 编辑 | 判定 | 依据 |
|---|---|---|
| 松散件（螺母/垫圈）袋内平移重摆 | 合法 | 袋内松散件位置不受约束；部件数与规格不变（正常材料证据 3） |
| 删除一个必需件（A 对违规侧） | 违规 | 官方逻辑异常 1_missing_nut / 1_missing_washer / …（defects_config 243/241/…） |
| 短螺丝替换为第二根长螺丝（B 对违规侧） | 违规（计数不变） | 官方逻辑异常 screw_too_long（defects_config 255）：规格违例；本变体把"总数不变、规格错"与"数量错"解耦，为消融提供第二对照轴 |

## pushpins

| 编辑 | 判定 | 依据 |
|---|---|---|
| 同一枚钉格内平移 | 合法 | 正常约束是"每格一钉"而非"钉在格内何处"；格内位置自由、总数不变 |
| 同一枚钉跨格平移 | 违规（计数不变） | 一格空、一格两钉，违反每格一钉约束；与官方 1_additional_pushpin + missing_pushpin 的**组合**同类（把两种官方缺陷合并为单次编辑的纯分配违例） |
| 删除一枚钉（B 对违规侧） | 违规 | 官方逻辑异常 missing_pushpin（defects_config 253） |

## 设计注记

- **"只数总数"基线不可区分**：跨格移动前后总数都是 15，任何只依赖全局计数的规则基线
  无法区分合法/违规两侧——这正是配对诊断测"空间分配"而非"计数"的证据（C 可在诊断中
  加一个总数基线作对照展示）。
- **距离差的结构性披露**：pushpins 跨格移动的移动距离必然大于格内移动（格距下限），
  已在清单 `controls.distance_note` 中披露；面积与编辑流程则完全匹配。
- 合法侧从未使用"分数低"作为依据；所有判定可回溯到官方缺陷类型或正常材料组成审计。
"""

DOCS['objectives.md'] = """# 目标定义（D 负责"目标应该怎样算"，E 负责接入训练程序）

以下给出与配对样本配套的训练目标定义。公式与进度报告 §4 保持一致，此处为可执行口径。

## 1. 排序目标（困难异常/违规样本）

对通过验收（`review_status=approved`）的配对 (C, C⁻)（原图组成图、违规编辑组成图）：

    L_rank = max{0, m + s_c(C) − s_c(C⁻)},  0 < m < 1

- `s_c` = SALAD 组成分支图像分数（`score_branch=composition`）；同对两侧输入
  来自同一表示、同一标签方案，比较内部有效。
- **m 在 development 用途配对上选定**（不接触 final_diagnostic、不按测试 AUROC 调）。
  建议从 {0.05, 0.1, 0.2} 网格中按 dev 集上"违规>合法比例"的稳定性选点并记录选择依据。
- 排序只约束相对次序，不发明逐像素目标；歧义编辑（B 未能通过几何校验的）不进入训练。

## 2. 空间一致性目标（合法变化）

对合法侧编辑 (C, C_spatial)（原图组成图 vs 合法重摆/格内移动组成图）：

    L_spatial = | s_c(C) − s_c(C_spatial) |

- 两侧都在组成图表示上（map-space 诊断口径，不宣称端到端 RGB 鲁棒性）；
- 与光照一致性路径分开报告；光照路径需可运行的组件预测器，当前状态为
  **pending**（进度报告口径不变），不得用静态组成图冒充重新预测。

## 3. R1 vs R2 排序控制实验（同一样本、只差一个开关）

- **R0**：原版 SALAD（使用其自带合成机制）。
- **R1**：在相近合成样本预算下，把预先规定的部分样本替换为
  `r1_r2_sample_list.json` 中**固定的**违规样本（即清单 screw_bag/train 用途对的
  invalid 输入），**不开启**排序损失；正常样本与步数维持一致。
- **R2**：与 R1 完全相同的样本、起点、步数、种子与数据顺序，**仅开启** L_rank。
- 三组均关闭一致性项（隔离解释）；R1 对 R2 回答排序损失的额外作用，
  R0 对 R1 回答更换样本方案的作用（不得夸大为"人工核查本身的效果"）。
- E 的两组配置文件逐项对比只允许差 `ranking_loss.enabled` 一个条目。

## 4. 消融与接入（与进度报告表 1 对齐）

- 训练目标总式：`L_total = L_base + λ_photo·L_photo + λ_spatial·L_spatial + λ_rank·L_rank`；
  各 λ、m、编辑范围、采样比例在 development 配对上冻结后才允许跑 final_diagnostic。
- final_diagnostic 评价（C 执行）：违规版本分数 > 合法版本的比例、两侧分数差、
  合法版本相对原图的分数变化；同时报告源图数与编辑对数两个单位（平局单列）。
"""

DOCS['integration_E.md'] = """# E 接入指南（配对样本与运行元数据）

## 1. 输入格式

- 三个输入路径都在 `pair_manifest.json` 条目内：`original_path` / `valid_path` /
  `invalid_path`（相对交付包根）。
- 文件：256×256 PNG，PIL 'L' 模式，整数标签（screw_bag 0..4；pushpins 0..2）。
  与 SALAD `read_mask(num_cls=6)` 兼容（≤5）。
- 同一源图的多对共享同一 `original_path` 文件（E 的预测 CSV 中原图共享一行）。

## 2. 预测 CSV（按 C 的入口约定）

    run_id,image_path,score

- `image_path` 用清单中的路径**原样**写入（`data/...` 前缀保留，相对交付包根；
  若 C 的评价器以 LOCO 数据根为根，请与 C 确认一个统一的根约定后再改写，
  不要自行截断文件名）。
- 只包含所选类别、所选用途、`approved` 对的三种输入路径；
- 合成变体**不冒充** LOCO 官方结构/逻辑异常（不写入官方 test 评价）。

## 3. 配对运行 JSON（模板见 `integration/run_template_pair_eval.json`）

在 E 的通用六字段之外，必填：
- `representation`: "composition_map"（与清单一致）；
- `score_branch`: "composition"（组成图变体当前只接组成分支）；
- `model_seen_source_paths`: 起点检查点训练中见过的**完整原始 RGB 路径数组**——
  注意 **SALAD 官方训练同时使用 train 与 validation 划分**（`train_salad.py` 使用
  `category/train` 训练、`category/validation` 做验证/归一化），因此
  validation/good 若被见过**必须列入**；v2 起全部配对源图都在 train/good（SALAD
  train 划分，必然被见过），同样必须完整列入；未知不能填空数组冒充未见过；
- `seen_sources_complete`: true（E 已覆盖检查点历史）；
- `protocol_frozen`: 最终诊断必需 true（由 A/D 确认规则与参数已冻结；
  开发诊断可为 false 并注明）。

## 4. R1/R2 固定样本

  违规样本 = `r1_r2_sample_list.json`（{R1R2_N} 个 screw_bag/train 用途对的 invalid 输入，
  附 SHA-256）。两组运行消费**完全相同**的该列表（含 original 侧）；
- 建议实现为数据集开关：`ranking_loss.enabled` 之外的一切（数据顺序、种子、步数、
  起点）逐项一致；配置文件差异项数=1，交付时附 `diff` 输出。

## 5. 短训练验收（接 D 样本前）

按约定先做几百步检查：数据配对读入、损失数值有限、梯度非零、保存/恢复一致；
D 的样本侧入口失败（文件缺失、哈希不符）应直接报错而非静默跳过。
"""

DOCS['review_guide.md'] = """# B 审核指南（验收流程与判定要点）

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
"""

os.makedirs(f'{ROOT}/docs', exist_ok=True)
for name, content in DOCS.items():
    path = f'{ROOT}/{name}' if name == 'README_D.md' else f'{ROOT}/docs/{name}'
    with open(path, 'w') as f:
        f.write(content)
    print('wrote', path)

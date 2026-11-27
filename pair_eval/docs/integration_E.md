# E 接入指南（配对样本与运行元数据）

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

# generator/ —— D 配对生成工具（可复现）

## 用途分配依据（v2）

源图用途不再是脚本内的连续区段方案，而是逐图读取 **B 的官方划分清单**
（`official_split/pushpins_split_manifest.csv`、`official_split/screw_bag_split_manifest.csv`，
每图一行：train_edit_source / dev_source / final_diag_source，覆盖全部 train/good）。
清单副本随包发布，是用途分配的唯一事实来源；改动它再重跑即可改变分配。

## 一键重跑（延伸到全部源图 / 补齐 screw_bag 覆盖）

```bash
# 1) (可选) 补齐 screw_bag 源图部件定位——v2 只标注 train/good（B 清单角色全部在 train/good，
#    validation/good 无角色不再标注）；断点续跑，限流时多次运行即可
python3 vlm_annotate.py --budget 3600

# 2) 重建源图记录（组成图 + 部件/钉记录；含缓存与断点；
#    新增 pushpins train/good 300–371 号图的提取）
python3 -c "import generate_deliverable as gd; gd.build_records()"

# 3) 生成全部配对 + 清单 + QA（用途按 B 官方清单；幂等：
#    删除 records_cache.pkl 后重跑可全量重建）
python3 generate_deliverable.py

# 4) 文档重出（README_D.md + docs/，动态数字随清单更新）
python3 write_docs.py

# 5) 附属产物（contact sheets / triplets / workbook / r1_r2 / 模板 / 统计）
python3 build_auxiliary.py

# 6) 几何约束复核（差异像素必须落在 edit_region 内）
python3 confinement_check.py
```

依赖：numpy、scipy、opencv-python、Pillow、`z-ai` CLI（仅 VLM 标注步骤需要）。
运行目录须能访问 LOCO 数据根（脚本内 `DATA_ROOT` 变量，默认
`/home/z/my-project/workspace/data/mvtec_loco`，可按需修改）。
注意：`generate_deliverable.py` 会清空并重建交付根目录（`OUT_ROOT` 变量），
请在生成之后再放置/修改 docs、generator 等自建内容（或调整脚本保留逻辑）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `official_split/` | **B 交付的官方划分清单副本**（用途分配唯一依据） |
| `vlm_annotate.py` | screw_bag 逐图部件定位（VLM 边界框+类型，限流退避，断点续跑；v2 只标 train/good） |
| `screwbag_map_builder.py` | 边界框→精确部件掩膜→严格组成门限（无静默回退） |
| `screwbag_extract.py` | 袋/封口线/内界估计与部件特征工具（供 map_builder 与 orchestrator 调用） |
| `pushpins_extract.py` | pushpins 几何组成图提取（黄头/格壁点阵/格子区域/针归属） |
| `pair_generator.py` | 全部编辑操作（平移重摆/删除/替换副本/格内与跨格移动）与画布落笔 |
| `generate_deliverable.py` | 编排：B 官方划分→用途分配→配对生成→QA→清单/状态/统计 |
| `build_auxiliary.py` | contact sheets、三联图、审核工作簿、R1/R2 清单、运行模板、统计 |
| `write_docs.py` | 文档生成（README_D.md + docs/ 内容，关键数字动态计算） |
| `confinement_check.py` | 几何约束复核（差异像素限于 edit_region，0 违规） |
| `annotations/vlm_annotations_sb_v2.json` | screw_bag 已完成的 VLM 定位结果（审计可追溯） |
| `defects_config_screw_bag.json`、`defects_config_pushpins.json` | 官方缺陷类型定义副本（规则依据） |

## 数据来源与许可

- MVTec LOCO AD（CC BY-NC-SA 4.0，© MVTec Software GmbH），经由社区镜像获取
  （huggingface Rafa612/mvtec_loco）；引用 Bergmann et al., IJCV 2022。
- 本包只含组成图与编辑产物（标签图），不复制原始 RGB 数据；
  `source_path` 指向各成员本地 LOCO 数据根中的官方文件。

## 版本

- `D_pair_generator 1.1.0`（v2：用途分配改读 B 官方划分清单；其余与 1.0.0 同）；
  确定性 = `crc32(pair_id)` 有序候选网格（详见 docs/method.md §4）。

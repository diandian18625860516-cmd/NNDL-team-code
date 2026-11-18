# NNDL：逻辑异常检测小组项目

这是 2026-11-18 重新整理后的共同代码入口。旧 A/B/C/D/E 分工、旧日期安排和旧实验配置已停止作为执行依据；任务重新认领，见 [任务表](docs/TASKS.md)。

研究方向：以 SALAD 为主要方法，研究模型能否接受正常变化、识别违反产品规则的变化。PatchCore 用于比较；一致性、排序和样本审核是待验证的改进方向。新实验范围在原版流程跑通后确定。

**当前状态：代码准备阶段。** 已收录 SALAD 本地作者源码副本和许可，完成轻量数据检查工具；尚未在新环境跑通模型、训练新模型或产生新成绩。已有报告成绩不作为本仓库的实验结果。

## 从这里开始

1. 用 VS Code 打开本文件夹，先读 [重新开始的工作顺序](docs/01_restart.md)。
2. 在终端运行 `python3 scripts/verify_sources.py`，检查作者代码副本有没有变化。
3. 运行 `python3 scripts/check_environment.py`，查看 Python 和依赖情况。
4. 复制 `configs/paths.example.json` 为 `configs/paths.local.json`，填写自己电脑的数据目录，再运行：

```bash
python3 scripts/check_data.py --config configs/paths.local.json --category screw_bag
```

Windows 可以把 `python3` 换成 `python`。这些检查只用 Python 标准库，不安装深度学习环境、不下载数据、不训练模型。每人电脑的路径只放在 `paths.local.json`，该文件不上传。

## 文件放在哪里

| 位置 | 内容 |
| --- | --- |
| `vendor/salad/` | 导入的作者源码、README、许可和方法示意图；导入时未改动 |
| `scripts/` | 本项目的数据、环境、源码检查工具 |
| `configs/` | 共享配置样例和被忽略的个人路径配置 |
| `src/` | 后续小组训练适配、改进和推理代码 |
| `baselines/patchcore/` | PatchCore 来源和待接入说明；尚未导入实现 |
| `docs/` | 工作顺序、任务表、协作方式、验证记录 |
| `templates/` | 运行记录格式，不包含实验成绩 |
| `data/`、`weights/`、`outputs/` | 个人机器或云端的数据、权重和运行输出；默认不上传 |

[环境准备说明](docs/02_environment.md) · [GitHub 共享与日常协作](docs/03_collaboration.md) · [数据和结果约定](docs/04_data_and_results.md) · [代码讲解](docs/05_code_walkthrough.md) · [来源与许可](NOTICE.md)

先完成一个类别的“读取数据 → 原版推理 → 导出逐图结果 → 统一评价”，再进行训练和改进。云端实例、GPU 费用和正式训练步数需要根据短运行实测决定。

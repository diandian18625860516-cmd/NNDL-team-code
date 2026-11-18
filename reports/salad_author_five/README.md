# SALAD 作者权重在五类 LOCO AD 上的复评

1,568 张官方测试图，每张保留融合分数和三个分支分数。五类等权平均：逻辑 AUROC **96.42%**、结构 **95.79%**、macro **96.11%**、pooled **96.00%**。这是作者权重推理，没有重新训练模型或使用小组的合成编辑样本。

## 老师可以先复算结果（不需要 GPU、图片或权重）

在仓库根目录、有 scikit-learn 的 Python 环境中运行（实测 1.6.1）：

    python src/salad_author_eval/reproduce.py

脚本用独立的原图清单 manifest.csv 核对每类 test 路径、标签和数量；读取各类别 predictions.csv 的 score 列重算四项 AUROC，并与 summary_percent.csv 和作者程序保存的 macro 核对。分数越高越异常。逻辑、结构分别与同一类正常图比较；macro 取两项平均，pooled 使用该类全部测试图。总表最后一行是五个类别等权平均，不混合不同类别的原始分数。

## 看哪些代码

- **作者模型**：仓库已有的 `vendor/salad/test_salad.py` 加载五个模型，用正常训练集估计特征统计量、验证集做校准，调用 `train_salad.py` 的 `test` 对官方测试图打分。
- **小组推理适配**：`src/salad_author_eval/run_salad_author.py` 检查原图与组成图相对路径，在作者测试循环导出图像标识、标签、融合分数、外观/全局/组成分支分数。保留原始计算式。Windows 使用单进程读图，兼容中文路径；Breakfast Box 只在内存中重封装原检查点自己的模型条目，忽略混入的其他目录，不改变参数。
- **小组统一评价**：`image_level.py` 定义比较集合和 AUROC；`common.py` 检查分数方向、重复与缺失；`reproduce.py` 用同一套规则复算五类汇总。

## 需要重新执行 GPU 推理时

本次用小组 2026-11-18 的主分支源码快照（上游提交号未核实）。运行器直接使用带有 `NNDL-team-code-main/vendor/salad/` 前缀的仓库 ZIP，不修改厂商目录。可以使用原始小组 ZIP；如当前分支中的作者源码仍一致，也可在仓库根目录执行：

    git archive --format=zip --prefix=NNDL-team-code-main/ -o NNDL-team-code-main.zip HEAD vendor/salad

按[作者仓库](https://github.com/MaticFuc/SALAD)准备原图、作者组成图和作者权重。三个根目录下面均按类别名建子目录。每个权重子目录应有五个对应的 .pth，并放入本报告该类别的 SOURCE.json（记录本次实际使用的作者文件，仅在权重匹配时使用）。每类 run.json 记录模型来源、数据范围、软件版本和设备。原始运行环境为 Windows、Python 3.12、PyTorch 2.7.1+cu128；CPU 复算无需安装这些 GPU 依赖。

以 Pushpins 为例，将三个输入根目录换成实际位置，输出目录必须尚不存在：

    python src/salad_author_eval/run_salad_author.py --archive NNDL-team-code-main.zip --rgb data/mvtec_loco --maps data/mvtec_loco_composition_maps --weights weights/salad_author --category pushpins --out outputs/salad_pushpins_recheck

输出 predictions.csv、作者汇总 results.csv 和运行记录 run.json。其余类别只改类别名和输出目录。用于展示时，优先使用上面的 CPU 复算入口；完整 GPU 推理需要本地数据、权重及匹配环境。

# 环境与 GPU 准备

## 现在可做的检查

Python 3.10 或以上即可运行 `scripts/` 和 `tests/` 中的准备工具，无需安装 PyTorch。Mac、Windows、Linux 均可使用；路径填入个人 `configs/paths.local.json`。本机已测试的 Python 版本见验证记录。

```bash
python3 scripts/verify_sources.py
python3 scripts/check_environment.py
python3 -m unittest discover -s tests -v
```

## 模型环境仍须在目标机器验证

作者 README 以 Python 3.10 和 CUDA 12.1 的 PyTorch 安装入口为起点，但没有给出完整锁定版本。本仓库保留该说明，没有把它冒充经过验证的环境文件。当前 Mac 的 Python 环境不应直接复制为 GPU 训练环境。

在确定学校/组内机器或云端实例后，先在单独的 Python 3.10 环境安装匹配的 PyTorch/torchvision，再依据作者依赖清单逐项验证导入。检查 NumPy 与 imgaug/OpenCV 的版本兼容；现代 PyTorch 的 `torch.load` 默认行为与旧完整模型权重也可能不兼容。只加载来源确认的作者/自训练权重，不盲目修改加载参数。

确认后保存 `python -m pip freeze`、Python、PyTorch、CUDA、驱动和 GPU 型号，提交可重建的环境文件。不能仅写“装了最新版”。

源码关键入口：

- `vendor/salad/argparser.py`：类别、原图目录、组成图目录、权重、步数、随机种子。
- `vendor/salad/test_salad.py`：加载五个模型对象，做原版推理和汇总；新项目的逐图分数导出需要接入。
- `vendor/salad/train_salad.py`：原版训练；默认 70,000 步，不能把默认命令当轻量测试运行。
- 训练默认读取 ImageNet 辅助图片；传入字符串 `none` 会关闭相应项，属于实验设置改变，要记录并在对照间保持一致。
- 原版保存权重不等于保存优化器、调度器和随机状态；可靠续训需要在新代码中补齐并验证。
- 作者 README 中权重脚本名与实际文件存在差异，实际文件为 `download_pretrained_models.sh`。脚本会下载、解压并删除压缩包，使用前应核对目标目录和空间。

## 租用办法

优先检查已有学校或组内 GPU。需要租用时，先单卡按量短测；GPU 型号、显存、单价和预算上限先登记，是否足够用实测判断。

先验证数据读取、一次推理、短训练、保存与恢复，再估算每组耗时。比较实验应使用同一训练预算；GPU 开机期间会计费，扩容存储可能单独计费，重要模型和日志另存。

官方参考：[AutoDL 快速开始](https://www.autodl.com/docs/quick_start/)、[计费](https://www.autodl.com/docs/price/)、[数据保留](https://www.autodl.com/docs/instance_data/)。本次没有租用实例或运行训练。

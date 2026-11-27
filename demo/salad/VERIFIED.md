# Pushpins 实际成功运行记录 · 2026-11-19

以下是本次 Windows 笔记本在 NNDL 工作目录执行的 PowerShell 命令。此处保留真实调用方式，用户目录用 $HOME 表示；其他成员请用 README.md 的独立环境和仓库内路径，不必照搬运行者的磁盘布局。

```powershell
$env:PYTHONPATH=(Resolve-Path .local-salad/deps).Path
$env:TQDM_MININTERVAL='15'
$env:PYTHONUTF8='1'
$py="$HOME\Desktop\作业们\大三上\NNDL\小组\C_PatchCore基线\.venv\Scripts\python.exe"
& $py -B 'C_后续研究/interfaces/GitHub上传内容/demo/salad/run_demo.py' --category pushpins --archive "$HOME/Downloads/NNDL-team-code-main.zip" --rgb 'D:/浏览器下载文档/mvtec_loco_anomaly_detection' --maps data/mvtec_loco_composition_maps --weights weights/salad_author --out tmp/salad_demo_pushpins_verified *> tmp/salad_demo_pushpins_verified.log
```

实际退出码 0。显卡 RTX 5070 Laptop，PyTorch 2.7.1+cu128。原图/组成图覆盖 train 372、validation 69、test 310；权重五个文件匹配来源记录。作者程序用时 101.1 秒，输出：

```text
Final image auc: 99.5124, img 96.7247, mlp 90.9567, comp 88.0294
Finished: 310 predictions; 101.1 seconds
310 official test images; macro AUROC = 99.5124%
```

最后一行来自用此次推理新生成的 predictions.csv 与原图清单独立复算，不是读取已归档汇总表。

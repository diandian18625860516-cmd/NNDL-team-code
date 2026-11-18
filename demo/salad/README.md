# 从图像到分数：在 Windows 上演示 SALAD 作者权重推理

推荐演示 Pushpins。程序会实际读取原图与作者组成图，加载五个模型文件，用 372 张正常训练图估计统计量、69 张验证图校准，然后给 310 张官方测试图打分。**推理不更新权重。** 本目录是增量包：把它放回已有小组代码的仓库根目录，保留同级的 `src/salad_author_eval` 和 `reports/salad_author_five`。

## 1. 建立独立环境（PowerShell，在仓库根目录）

使用 Windows 64 位、Python 3.12 和支持 CUDA 12.8 的 NVIDIA 驱动。实测机器是 RTX 5070 Laptop 8 GB。先运行 `nvidia-smi` 确认能看到显卡。下列命令按成功运行时的版本编写；本次验证复用了已有环境，没有在全新机器上重装。

```powershell
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r demo/salad/requirements.txt
& $py -m pip check
& $py -c "import torch,torchvision,cv2,pandas,scipy,sklearn,tqdm,positional_encodings; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

新建环境前请确认没有要保留的同名 .venv；已有可用环境也可直接把 $py 指向它。该演示使用作者已发布的组成图，不需要安装 SAM、下载 ImageNet 或启动训练。[PyTorch 对应版本](https://pytorch.org/get-started/previous-versions/)

## 2. 准备四个输入

| 参数 | 指向什么 |
|---|---|
| --archive | 小组已验证的 NNDL-team-code-main.zip，内部有 NNDL-team-code-main/vendor/salad/test_salad.py。不是本补充包。若作者源码未变，可用下文 git archive 命令生成。 |
| --rgb | [MVTec LOCO 原图](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad/downloads)根目录，下面有 pushpins/train、validation 和 test（均在 pushpins 内）。 |
| --maps | [作者组成图下载脚本](https://github.com/MaticFuc/SALAD/blob/main/download_composition_maps.sh)对应的解压根目录，类别内的相对图片路径要与原图相同。 |
| --weights | [作者预训练权重下载脚本](https://github.com/MaticFuc/SALAD/blob/main/download_pretrained_models.sh)所指压缩包的解压根目录，下面有 pushpins/teacher_final.pth 等。 |

权重子目录包含 teacher_final、student_final、autoencoder_final、comp_autoencoder_final、comp_unet_final 五个 .pth。**本包 weight_metadata/ 下的五份 SOURCE.json 直接取自实际权重目录。** 演示程序按类别检查五个文件的大小和 ZIP CRC；匹配后才把 SOURCE.json 复制到权重根/类别/SOURCE.json。已有来源记录若冲突会报错，不会覆盖。无需提交权重本体。

没有原始代码 ZIP 时，在组仓库中运行（若 vendor 已修改，应切回匹配版本）：

```powershell
git archive --format=zip --prefix=NNDL-team-code-main/ -o NNDL-team-code-main.zip HEAD vendor/salad
```

## 3. 跑真实推理

下面的参数与成功运行一致。把路径换成你的目录；若 ZIP 在下载文件夹，直接把 --archive 指到那里。输出必须是新目录，避免覆盖上次结果。

```powershell
& $py -B demo/salad/run_demo.py --category pushpins --archive NNDL-team-code-main.zip --rgb data/mvtec_loco --maps data/mvtec_loco_composition_maps --weights weights/salad_author --out outputs/pushpins_demo_01
```

看到 `310 official test images` 表示测试图已与独立清单逐路径匹配。打开输出目录的 **predictions.csv** 看每张图的融合分数和各分支分数；**run.json** 记录权重、环境、时间和显存；**results.csv** 是作者程序的百分制汇总；**demo_metrics.json** 是本次新生成的 CSV 复算结果，其中 AUROC 为 0–1 小数。参考 macro AUROC **99.5124%**。不同硬件或数值库可能有细微差异。

实测调用见同目录 VERIFIED.md。若只需课堂展示已归档五类结果的复算，运行 `python src/salad_author_eval/reproduce.py`；那一步不执行神经网络。

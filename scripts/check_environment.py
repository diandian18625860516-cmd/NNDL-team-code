"""查看环境；默认只读包版本，--gpu 才加载 PyTorch 检查 CUDA。"""
import argparse
import importlib.metadata
import platform
import sys

PACKAGES = ("torch", "torchvision", "numpy", "Pillow", "scipy", "scikit-learn",
            "opencv-python", "imgaug", "timm", "positional-encodings", "pandas", "tqdm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="在候选训练机器检查 CUDA")
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]}; 系统: {platform.system()} {platform.machine()}")
    for package in PACKAGES:
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: 未发现")
    if args.gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA 不可用：当前机器未通过 SALAD GPU 环境检查。")
                return 1
            print(f"CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
        except Exception as error:
            print(f"PyTorch 检查失败：{error}")
            return 1
    print("此输出为环境清单，不代表依赖兼容或模型已跑通。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

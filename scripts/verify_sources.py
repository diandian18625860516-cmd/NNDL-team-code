"""验证已导入 SALAD 文件的指纹；不连接网络、不运行作者代码。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "vendor/SALAD_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["sha256"].items():
        path = ROOT / "vendor/salad" / relative
        if not path.is_file():
            failures.append(f"缺少：{relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"已改动：{relative}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"通过：{len(manifest['sha256'])} 个导入文件与导入时一致。上游提交号仍待联网核实。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

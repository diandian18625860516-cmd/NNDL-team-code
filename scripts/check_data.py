"""核对原图与组成图的完整相对路径；不导入模型，不修改输入数据。"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("screw_bag", "pushpins", "breakfast_box", "juice_bottle", "splicing_connectors")
SPLITS = ("train", "validation", "test")


def load_paths(config_path):
    """配置中的相对路径以仓库根目录为基准，而不是终端当前目录。"""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    paths = {}
    for key in ("dataset_root", "composition_root"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"配置缺少有效的 {key}")
        path = Path(value).expanduser()
        paths[key] = path if path.is_absolute() else REPO_ROOT / path
    return paths


def image_paths(folder):
    """保留 good/000.png 等相对路径，避免同名图片被当作同一张。"""
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() == ".png"}


def check_category(dataset_root, composition_root, category):
    rows = []
    for split in SPLITS:
        original_dir = dataset_root / category / split
        map_dir = composition_root / category / split
        originals = image_paths(original_dir)
        maps = image_paths(map_dir)
        missing = sorted(originals - maps)
        extra = sorted(maps - originals)
        ok = (original_dir.is_dir() and map_dir.is_dir() and bool(originals)
              and not missing and not extra)
        rows.append({"split": split, "original_count": len(originals),
                     "composition_count": len(maps), "missing_maps": missing,
                     "extra_maps": extra, "passed": ok})
    return {"category": category, "passed": all(row["passed"] for row in rows),
            "checks": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--category", choices=(*CATEGORIES, "all"), default="screw_bag")
    parser.add_argument("--output", type=Path, help="可选：将检查报告保存到 outputs/ 下")
    args = parser.parse_args()
    try:
        paths = load_paths(args.config)
    except (OSError, ValueError) as error:
        parser.exit(2, f"无法读取配置：{error}\n请参考 configs/paths.example.json。\n")
    categories = CATEGORIES if args.category == "all" else (args.category,)
    results = [check_category(paths["dataset_root"], paths["composition_root"], c)
               for c in categories]
    for result in results:
        print(f"\n{result['category']}")
        for row in result["checks"]:
            state = "通过" if row["passed"] else "未通过"
            print(f"  {row['split']}: 原图 {row['original_count']}，组成图 "
                  f"{row['composition_count']}，缺失 {len(row['missing_maps'])}，"
                  f"多余 {len(row['extra_maps'])} — {state}")
            for item in row["missing_maps"][:3]:
                print(f"    缺组成图：{item}")
            for item in row["extra_maps"][:3]:
                print(f"    多余组成图：{item}")
    report = {"check_type": "path_pairing_only", "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n范围：仅检查文件路径配对；未检查图像解码、标签语义、数据泄漏或模型运行。")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

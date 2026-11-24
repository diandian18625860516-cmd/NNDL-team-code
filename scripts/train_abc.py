"""小组共用入口：先按配置准备，再开始实际训练。运行 --help 查看参数。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from salad_training.contract import load_contract, make_schedule, preflight, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--rgb-root", type=Path)
    p.add_argument("--maps-root", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--pairs", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--plan", action="store_true", help="显示确定的设置，不启动模型")
    p.add_argument("--preflight", action="store_true", help="核对真实输入，不启动模型")
    p.add_argument("--smoke", action="store_true", help="运行 100 步工程短训练，不作正式成绩")
    p.add_argument("--stop-after", type=int, help="本次完成指定数量更新后保存，供中断恢复验证")
    p.add_argument("--resume", type=Path, help="恢复本入口保存的 checkpoint_latest.pt")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    c = load_contract(args.config)
    if args.plan:
        schedule = make_schedule(c, c["settings"]["steps"])
        print(json.dumps({"group": c["group"], "variant": c["variant"],
                          "source_counts": {k: len(v) for k, v in c["roles"].items() if isinstance(v, list)},
                          "steps": c["settings"]["steps"], "seed": c["settings"]["seed"],
                          "schedule_sha256": fingerprint(schedule)}, ensure_ascii=False, indent=2))
        return
    if not all((args.rgb_root, args.maps_root, args.weights)):
        p.error("请提供 --rgb-root、--maps-root、--weights")
    result = preflight(c, args.rgb_root, args.maps_root, args.weights, args.pairs)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.preflight:
        return
    if args.out is None or (args.stop_after is not None and args.stop_after < 1):
        p.error("实际训练需要 --out；--stop-after 必须为正数")
    try:
        from salad_training.trainer import train
    except ImportError as e:
        raise SystemExit("训练环境缺少依赖。请先在运行机器安装 requirements-training.txt。原始信息：" + str(e)) from e
    train(c, args)


if __name__ == "__main__":
    main()

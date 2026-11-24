"""不用加载深度学习库即可读取约定、生成采样顺序、核对输入。"""
import hashlib
import json
import math
import random
import zipfile
from pathlib import Path, PurePosixPath


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return stream_sha(stream)


def stream_sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative or ":" in relative:
        raise ValueError("需要不含上级跳转的数据相对路径: " + relative)
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError("文件超出数据根目录: " + relative)
    return target


def validate_roles(roles):
    names = ("optimization", "development", "final_diagnostic", "calibration", "official_test")
    seen = set()
    for name in names:
        values = roles[name]
        if not values or len(values) != len(set(values)) or seen.intersection(values):
            raise ValueError("源图用途为空、重复或交叉: " + name)
        seen.update(values)
        prefix = "screw_bag/" + ("test/" if name == "official_test" else "validation/good/" if name == "calibration" else "train/good/")
        if any(not p.startswith(prefix) for p in values):
            raise ValueError("源图官方划分与用途冲突: " + name)


def load_contract(path):
    path = Path(path).resolve()
    group_ref = read_json(path)
    cfg = read_json(path.parent / group_ref["common"])
    group = group_ref["group"]
    if group not in cfg["groups"]:
        raise ValueError("未知实验组")
    roles = read_json(path.parent / cfg["source_roles"])
    validate_roles(roles)
    maps = read_json(path.parent / cfg["map_hashes"])
    if set(maps) != set(sum((roles[k] for k in ("optimization", "development", "final_diagnostic", "calibration", "official_test")), [])):
        raise ValueError("作者组成图清单与用途清单不一致")
    if cfg["batch_size"] != 1 or cfg["image_size"] != 256 or cfg["imagenet_penalty"] or cfg["photo_loss"]:
        raise ValueError("V1 入口只实现 batch=1、256 像素、无额外 ImageNet/光照目标")
    if cfg["steps"] <= 0 or cfg["scheduler_step"] <= 0:
        raise ValueError("训练步数必须为正")
    if cfg["category"] != "screw_bag" or cfg["workers"] != 0 or cfg["reviewed_replacement_fraction_on_eligible_slots"] != 1.0:
        raise ValueError("V1 仅支持 Screw Bag、串行加载、全部符合条件的位置替换审核样本")
    if cfg["class_weight_mode"] != "preserve_author_onehot_count_behavior":
        raise ValueError("V1 必须保留作者类别权重行为")
    if not 0 < cfg["anomaly_probability"] < 1:
        raise ValueError("扰动概率必须在 0 与 1 之间")
    for key in ("appearance_lr", "composition_lr", "scheduler_gamma", "rank_margin"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError("需要有限正数: " + key)
    expected = {"SB0": ("author", 0, 0), "SB4": ("reviewed", 0, 0), "SB1": ("author", 1, 0), "SB2": ("reviewed", 0, 1), "SB3": ("reviewed", 1, 1)}
    for g, (mode, cons, rank) in expected.items():
        actual = cfg["groups"][g]
        if any(not math.isfinite(actual[k]) or actual[k] < 0 for k in ("spatial_weight", "rank_weight")):
            raise ValueError("目标权重需要有限非负数: " + g)
        if actual["sample_mode"] != mode or bool(actual["spatial_weight"]) != bool(cons) or bool(actual["rank_weight"]) != bool(rank):
            raise ValueError("组别开关定义不一致: " + g)
    return {"settings": cfg, "group": group, "variant": cfg["groups"][group], "roles": roles,
            "map_hashes": maps, "weights": read_json(path.parent / cfg["initial_weights"])}


def make_schedule(contract, steps):
    """全组同样的原图次序；每步种子独立于启用的损失和运行中断。"""
    cfg = contract["settings"]
    paths = sorted(contract["roles"]["optimization"])
    rng = random.Random(cfg["seed"])
    result = []
    while len(result) < steps:
        epoch = paths.copy()
        rng.shuffle(epoch)
        for path in epoch:
            i = len(result)
            if i >= steps:
                break
            local = random.Random(f'{cfg["seed"]}:corruption:{i}')
            result.append({"step": i, "source_path": path,
                           "donor_path": paths[local.randrange(len(paths))],
                           "corrupt": local.random() < cfg["anomaly_probability"],
                           "corruption_index": local.randrange(4),
                           "rgb_seed": cfg["seed"] * 1000003 + i * 3,
                           "corruption_seed": cfg["seed"] * 1000003 + i * 3 + 1,
                           "model_seed": cfg["seed"] * 1000003 + i * 3 + 2})
    return result


def load_pairs(path, contract, verify_files=True):
    if path is None:
        if contract["group"] != "SB0":
            raise ValueError("该组需要 B 交付作者组成图训练配对 JSON；审核 Excel 不能代替实际图像")
        return {}, None
    path = Path(path).resolve()
    obj = read_json(path)
    if obj.get("representation") != "salad_author_indices_v1":
        raise ValueError("配对表示必须明确为作者原始标签体系")
    groups = {}
    ids = set()
    for pair in obj["pairs"]:
        if pair["pair_id"] in ids:
            raise ValueError("重复 pair_id")
        ids.add(pair["pair_id"])
        if pair["purpose"] != "train" or pair["review_status"] != "approved":
            continue
        source = pair["source_path"]
        if source not in contract["roles"]["optimization"]:
            raise ValueError("审核训练对引用开发/诊断源图: " + source)
        if pair["source_map_sha256"] != contract["map_hashes"][source]:
            raise ValueError("配对源图不匹配作者组成图: " + pair["pair_id"])
        for field in ("reviewer", "review_date", "review_reason", "rule_reference"):
            if not isinstance(pair.get(field), str) or not pair[field].strip():
                raise ValueError("缺少人工审核记录: " + field)
        for kind in ("valid", "invalid", "anomaly_mask"):
            f = safe_path(path.parent, pair[kind + "_path"])
            if verify_files and (not f.is_file() or sha(f) != pair[kind + "_sha256"]):
                raise ValueError("配对文件缺失或内容已变化: " + str(f))
        groups.setdefault(source, []).append(pair)
    if not groups and contract["group"] != "SB0":
        raise ValueError("没有可用于训练的已审核作者图配对")
    for pairs in groups.values():
        pairs.sort(key=lambda p: p["pair_id"])
    return groups, path.parent


def weight_content(weights_path, member):
    """只打开指定的、事先登记的权重；不解压整个 2 GB 压缩包。"""
    weights_path = Path(weights_path)
    if weights_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(weights_path) as archive:
            return archive.read(member)
    path = weights_path / member
    if not path.is_file():
        path = weights_path / PurePosixPath(member).name
    return path.read_bytes()


def preflight(contract, rgb_root, maps_root, weights_path, pairs_path=None):
    for source, digest in contract["map_hashes"].items():
        rgb = safe_path(rgb_root, source)
        comp = safe_path(maps_root, source)
        if not rgb.is_file() or rgb.stat().st_size == 0:
            raise ValueError("原图缺失: " + source)
        if not comp.is_file() or sha(comp) != digest:
            raise ValueError("作者组成图缺失或指纹不符: " + source)
    for item in contract["weights"]["members"]:
        data = weight_content(weights_path, item["archive_member"])
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("起始权重不一致: " + item["archive_member"])
    pairs, _ = load_pairs(pairs_path, contract)
    reference = Path(__file__).parent
    for name, digest in read_json(reference / "author_snapshot.json").items():
        if sha(reference / "author_snapshot" / name) != digest:
            raise ValueError("作者参考代码被修改: " + name)
    return {"input_files_passed": True, "group": contract["group"],
            "optimization_sources": len(contract["roles"]["optimization"]),
            "training_pairs": sum(map(len, pairs.values())),
            "formal_steps": contract["settings"]["steps"], "seed": contract["settings"]["seed"],
            "model_runtime_tested": False}

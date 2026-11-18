"""Run the team's original SALAD test entry point; export its scores without reimplementing them."""
import argparse
import csv
from datetime import datetime
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from zipfile import ZipFile


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Team source version mismatch: " + old)
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "rgb", "maps", "weights", "out"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--category", default="screw_bag")
    args = p.parse_args()
    cat, out = args.category, args.out.resolve()
    rgb, maps = args.rgb.resolve(), args.maps.resolve()
    counts = {}
    for split in ("train","validation","test"):
        def paths(base):
            return {x.relative_to(base) for x in (base/cat/split).glob("*/*.png")}
        images = paths(rgb)
        if not images or images != paths(maps):
            raise ValueError(f"Unpaired RGB/maps: {cat}/{split}")
        counts[split] = len(images)
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    # Collect the variables actually used by the authors' test loop, not rounded metrics.
    rows = []
    def record(path, group, label, score, appearance, global_score, composition):
        rows.append((out.name, Path(path).relative_to(rgb).as_posix(), group, label,
                     float(score), float(appearance), float(global_score), float(composition)))
    fields = ("run_id","image_path","anomaly_type","label","score","appearance_score","mahalanobis_score","composition_score")
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone().isoformat()
    tick = time.monotonic()
    with tempfile.TemporaryDirectory(dir=out.parent) as directory, ZipFile(args.archive) as archive:
        prefix = "NNDL-team-code-main/vendor/salad/"
        for n in archive.namelist():
            basename = n.removeprefix(prefix)
            if not n.startswith(prefix) or "/" in basename or not basename.endswith(".py"):
                continue
            source = archive.read(n).decode().replace("\r\n","\n")
            if basename == "salad_dataset.py":
                source = once(source, "import imgaug.augmenters as iaa", "")
                source = once(source, "def get_perlin_noise(x0):","def get_perlin_noise(x0):\n    import imgaug.augmenters as iaa")
                source = once(source, "cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)",
                              "cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)")
            if basename == "train_salad.py":
                source = once(source,"        y_score[defect_class].append(y_score_image)",
                    "        record(path, defect_class, y_true_image, y_score_image, y_score_img_no_mlp, mahalanobis_score, np.max(map_comp))\n        y_score[defect_class].append(y_score_image)")
            if basename == "test_salad.py":
                if cat == "breakfast_box":
                    source = once(source,'torch.load(f"{train_output_dir}/autoencoder_final.pth")',
                                  'load_breakfast_autoencoder(train_output_dir)')
                source = once(source,"num_workers=4","num_workers=0")
                source = once(source,"log(train_output_dir,results)","log(result_directory,results)")
            (Path(directory)/basename).write_text(source,encoding="utf-8")
        sys.path.insert(0,directory)
        import ae, unet, train_salad, test_salad
        if cat == "breakfast_box":
            def load_breakfast_autoencoder(folder):
                # The released checkpoint contains extra top-level directories from the outer ZIP.
                # Rewrap only its original torch records; do not modify tensors or the source file.
                buf=io.BytesIO()
                with ZipFile(Path(folder)/"autoencoder_final.pth") as src, ZipFile(buf,"w") as dst:
                    for entry in src.infolist():
                        if entry.filename.startswith("autoencoder_final/"):
                            dst.writestr(entry.filename,src.read(entry))
                buf.seek(0)
                return torch.load(buf)
            test_salad.load_breakfast_autoencoder=load_breakfast_autoencoder
        train_salad.record = record
        test_salad.result_directory = str(out)
        allowed = [getattr(module,name) for module,names in (
            (ae,"AutoEncoder ConvBlock Decoder DownBlock Downsample Encoder UpBlock Upsample"),
            (unet,"DownBlock Downsample ResidualBlock SkipConnection UNet UpBlock Upsample"),
            (torch.nn,"ReLU SiLU ModuleList Sequential Conv2d ConvTranspose2d Dropout Identity GroupNorm AvgPool2d MaxPool2d Upsample"))
            for name in names.split()]
        sys.argv = ["test_salad.py","--category",cat,"--mvtec_loco_path",str(rgb),
                    "--mvtec_loco_seg_path",str(maps),"--output_dir",str(args.weights.resolve()),"--seed","42"]
        print("Start:",started,"counts:",counts,"device:",torch.cuda.get_device_name(),flush=True)
        with torch.serialization.safe_globals(allowed), torch.no_grad():
            test_salad.test_all()
    if len(rows)!=counts["test"]:
        raise ValueError("Wrong test count")
    with (out/"predictions.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(fields);writer.writerows(rows)
    meta=dict(run_id=out.name,model="SALAD",category=cat,stage="author_weights",seed=42,
        initialization=json.loads((args.weights/cat/"SOURCE.json").read_text(encoding="utf-8")),
        code_version={"team_snapshot":args.archive.name,"source":"vendor/salad","upstream_commit":None,
                      "adapter":"run_salad_author.py; exports original scores, zero Windows workers, Unicode mask reading, lazy training-only imgaug, trusted layer allowlist"},
        score_name="author_fusion",score_direction="higher_is_anomalous",
        score_definition="max(normalized appearance map) + normalized Mahalanobis score + max(normalized composition map)",
        data_split={"counts":counts,"train":"teacher statistics and global distribution","validation":"score/map calibration","test":"all official images"},
        preprocessing="Author RGB Resize(256,256), ImageNet normalization, supplied 6-channel one-hot composition map",
        steps=0,sample_budget=counts,branch_scores=list(fields[5:]),map_location=None,
        original_results="predictions.csv; results.csv (author macro AUROC in percent)",
        rgb_root=str(rgb),composition_root=str(maps),weights_root=str(args.weights.resolve()),
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,python=sys.version,packages={k:version(k) for k in
        ("torch","torchvision","numpy","scikit-learn","scipy","pillow","pandas","opencv-python-headless","positional-encodings")},
        start=started,elapsed_seconds=time.monotonic()-tick,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    if cat=="breakfast_box":
        meta["checkpoint_container_note"]="Repacked only original autoencoder_final/ records in memory; ignored extraneous nested category archives. Tensor and metadata payload bytes unchanged."
    (out/"run.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Finished:",len(rows),"predictions;",round(meta["elapsed_seconds"],1),"seconds",flush=True)


if __name__=="__main__":
    main()

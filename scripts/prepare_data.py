"""Extract, verify, and preview a reproducible Cityscapes subset; no labels/train split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

RGB_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelIds.png"
# Cityscapes raw label IDs, official semantic palette; void classes remain black.
COLORS = {7:(128,64,128),8:(244,35,232),11:(70,70,70),12:(102,102,156),
          13:(190,153,153),17:(153,153,153),19:(250,170,30),20:(220,220,0),
          21:(107,142,35),22:(152,251,152),23:(70,130,180),24:(220,20,60),
          25:(255,0,0),26:(0,0,142),27:(0,0,70),28:(0,60,100),
          31:(0,80,100),32:(0,0,230),33:(119,11,32)}


def select_pairs(rgb_names, mask_names, counts, seed):
    """Stable sample independent of ZIP member order. Fail on missing annotations."""
    rng = random.Random(seed)
    masks = set(mask_names)
    selected = []
    for split, count in sorted(counts.items()):
        if split not in {"train", "val"} or not isinstance(count, int) or count < 1:
            raise ValueError("Use positive counts for official train/val splits only")
        candidates = sorted(n for n in rgb_names
                            if n.startswith(f"leftImg8bit/{split}/") and n.endswith(RGB_SUFFIX))
        if count > len(candidates):
            raise ValueError(f"Requested {count} {split} samples; found {len(candidates)}")
        for name in candidates:
            mask = name.replace("leftImg8bit/", "gtFine/", 1).removesuffix(RGB_SUFFIX) + MASK_SUFFIX
            if mask not in masks:
                raise ValueError(f"Missing annotation: {mask}")
        for name in sorted(rng.sample(candidates, count)):
            mask = name.replace("leftImg8bit/", "gtFine/", 1).removesuffix(RGB_SUFFIX) + MASK_SUFFIX
            selected.append((split, name, mask))
    return selected


def safe_target(base, member):
    target = (base / member).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"Unsafe archive path: {member}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source, dest = ROOT / cfg["dataset_root"], ROOT / cfg["subset_root"]
    manifest, output = ROOT / cfg["manifest"], ROOT / cfg["output_dir"]
    rows = []
    with zipfile.ZipFile(source / "leftImg8bit_trainvaltest.zip") as rgb_zip, \
         zipfile.ZipFile(source / "gtFine_trainvaltest.zip") as mask_zip:
        pairs = select_pairs(rgb_zip.namelist(), mask_zip.namelist(), cfg["subset_counts"], cfg["seed"])
        for index, (split, rgb_name, mask_name) in enumerate(pairs, 1):
            rgb_bytes, mask_bytes = rgb_zip.read(rgb_name), mask_zip.read(mask_name)
            with Image.open(io.BytesIO(rgb_bytes)) as rgb, Image.open(io.BytesIO(mask_bytes)) as mask:
                rgb.load()
                mask.load()
                if rgb.size != mask.size or mask.mode != "L":
                    raise ValueError(f"Invalid image/mask pair: {rgb_name}")
                width, height = rgb.size
                label_ids = sorted(int(v) for v in np.unique(np.asarray(mask)))
                if not set(label_ids) <= set(range(34)) | {255}:
                    raise ValueError(f"Unexpected raw label IDs: {label_ids}")
            for name, content in ((rgb_name, rgb_bytes), (mask_name, mask_bytes)):
                target = safe_target(dest, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(content).digest():
                    target.write_bytes(content)
            sample_id = Path(rgb_name).name.removesuffix(RGB_SUFFIX)
            rows.append(dict(sample_id=sample_id, official_split=split,
                             sequence_id="_".join(sample_id.split("_")[:2]),
                             image_path=(dest / rgb_name).relative_to(ROOT).as_posix(),
                             mask_path=(dest / mask_name).relative_to(ROOT).as_posix(),
                             width=width, height=height,
                             image_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
                             mask_sha256=hashlib.sha256(mask_bytes).hexdigest()))
            if index % 100 == 0:
                print(f"Verified {index}/{len(pairs)} pairs", flush=True)
        for label, archive in (("images", rgb_zip), ("annotations", mask_zip)):
            for name in ("README", "license.txt"):
                if name in archive.namelist():
                    target = dest / "source_info" / label / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(cfg["seed"]).sample(rows, min(cfg["preview_count"], len(rows)))
    palette = np.zeros((256, 3), dtype=np.uint8)
    for key, color in COLORS.items():
        palette[key] = color
    for start in range(0, len(chosen), 5):
        batch = chosen[start:start + 5]
        fig, axes = plt.subplots(len(batch), 2, figsize=(14, 3 * len(batch)), squeeze=False)
        for axes_row, row in zip(axes, batch):
            with Image.open(ROOT / row["image_path"]) as im:
                rgb = np.asarray(im.convert("RGB").resize((768, 384)))
            with Image.open(ROOT / row["mask_path"]) as im:
                mask = np.asarray(im.resize((768, 384), Image.Resampling.NEAREST))
            axes_row[0].imshow(rgb)
            axes_row[1].imshow((0.55 * rgb + 0.45 * palette[mask]).astype(np.uint8))
            axes_row[0].set_title(f'{row["official_split"]}: {row["sample_id"]}', fontsize=9)
            axes_row[1].set_title("Semantic mask overlay (raw label IDs)", fontsize=9)
            for ax in axes_row:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / f"preview_{start // 5 + 1:02d}.png", dpi=100)
        plt.close(fig)
    report = {"seed":cfg["seed"], "verified_pairs":len(rows), "official_split_counts":cfg["subset_counts"],
              "preview_sample_ids":[r["sample_id"] for r in chosen],
              "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "note":"Official split names only. No experimental split or concept/action labels assigned."}
    (output / "data_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {manifest} and {len(chosen)} preview pairs", flush=True)


if __name__ == "__main__":
    main()

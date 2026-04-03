"""
Download timm/mini-imagenet from Hugging Face and save it in
the ImageFolder structure expected by torchvision / timm:

    <out>/
        train/<class_name>/<img>.JPEG
        validation/<class_name>/<img>.JPEG
        test/<class_name>/<img>.JPEG

Usage:
    python download_mini_imagenet.py --out /data/mini-imagenet

Prerequisites:
    pip install datasets pillow tqdm
    huggingface-cli login   # or set HF_TOKEN env var
"""

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


SPLITS = ["train", "validation", "test"]


def save_split(ds, split_dir: str) -> None:
    # Collect class names to build a stable index->name mapping.
    # The "label" feature is a ClassLabel, so ds.features["label"].int2str() works.
    label_feature = ds.features["label"]
    int2str = label_feature.int2str  # int -> class name string

    print(f"Saving {len(ds)} images to {split_dir} ...")
    for i, sample in enumerate(tqdm(ds, desc=os.path.basename(split_dir))):
        class_name = int2str(sample["label"])
        folder = os.path.join(split_dir, class_name)
        os.makedirs(folder, exist_ok=True)
        img = sample["image"]
        # Convert palette/RGBA images to RGB so JPEG encoding works.
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(os.path.join(folder, f"{i:07d}.JPEG"))


def main(out_dir: str, cache_dir: str | None) -> None:
    for split in SPLITS:
        print(f"\n=== Downloading split: {split} ===")
        ds = load_dataset(
            "timm/mini-imagenet",
            split=split,
            cache_dir=cache_dir,
        )
        save_split(ds, os.path.join(out_dir, split))

    print(f"\nDone. Dataset saved to {out_dir}")
    if cache_dir:
        print(f"You may delete the HF cache at {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/data/mini-imagenet",
        help="Root output directory (default: /data/mini-imagenet)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for HuggingFace parquet cache (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()
    main(args.out, args.cache_dir)

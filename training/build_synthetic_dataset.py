"""
build_synthetic_dataset.py – Dataset builder for CNN-CTC training

Uses the equation generator and synthetic renderer to create a labeled
image dataset with train/valid/test splits.

Output structure:
    <out>/
        images/
            train/  valid/  test/
        labels/
            train/  valid/  test/
        metadata.json

Each image file (e.g. images/train/000001.png) has a matching label file
(labels/train/000001.txt) whose content is the canonical equation string.

Sprint 2 implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.equation_generator.generate_equations import generate_equations
from training.synthetic_renderer.render_equation import render_and_save


def build_dataset(
    out_dir: str | Path,
    samples: int = 5000,
    seed: int = 42,
    width: int = 512,
    height: int = 128,
    train_frac: float = 0.80,
    valid_frac: float = 0.10,
) -> dict:
    """
    Build a synthetic equation dataset.

    Args:
        out_dir: Output directory.
        samples: Total number of samples to generate.
        seed: Random seed for reproducibility.
        width: Image width.
        height: Image height.
        train_frac: Fraction of samples for training.
        valid_frac: Fraction of samples for validation.

    Returns:
        Metadata dict with counts and paths.
    """
    out_dir = Path(out_dir)
    test_frac = 1.0 - train_frac - valid_frac
    assert test_frac > 0, "train_frac + valid_frac must be < 1.0"

    train_count = int(samples * train_frac)
    valid_count = int(samples * valid_frac)
    test_count = samples - train_count - valid_count

    # Create directory structure
    splits = ["train", "valid", "test"]
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Generate all equations (allow more than needed, then trim)
    print(f"Generating {samples} equations (seed={seed})...")
    equations = generate_equations(count=samples, seed=seed, deduplicate=True)

    # If deduplication reduced count, generate more with different seeds
    extra_seed = seed + 10000 if seed is not None else 10000
    while len(equations) < samples:
        extra = generate_equations(
            count=samples - len(equations),
            seed=extra_seed,
            deduplicate=True,
        )
        # Deduplicate against existing
        existing = set(equations)
        for eq in extra:
            if eq not in existing:
                equations.append(eq)
                existing.add(eq)
        extra_seed += 1

    equations = equations[:samples]

    # Split
    train_eqs = equations[:train_count]
    valid_eqs = equations[train_count : train_count + valid_count]
    test_eqs = equations[train_count + valid_count :]

    split_data = [
        ("train", train_eqs),
        ("valid", valid_eqs),
        ("test", test_eqs),
    ]

    t0 = time.time()
    rendered = 0

    for split_name, split_eqs in split_data:
        for i, eq in enumerate(split_eqs, start=1):
            filename = f"{i:06d}"
            img_path = out_dir / "images" / split_name / f"{filename}.png"
            lbl_path = out_dir / "labels" / split_name / f"{filename}.txt"

            # Use a deterministic per-image seed
            img_seed = hash((seed, split_name, i)) % (2**31)
            render_and_save(eq, img_path, lbl_path, width=width, height=height, seed=img_seed)
            rendered += 1

            if rendered % 100 == 0 or rendered == samples:
                elapsed = time.time() - t0
                rate = rendered / elapsed if elapsed > 0 else 0
                print(f"  [{rendered}/{samples}] {rate:.1f} img/s", flush=True)

    elapsed_total = time.time() - t0

    metadata = {
        "version": "synthetic_v1",
        "total_samples": samples,
        "splits": {
            "train": train_count,
            "valid": valid_count,
            "test": test_count,
        },
        "image_size": {"width": width, "height": height},
        "seed": seed,
        "format": {
            "image": "greyscale PNG",
            "label": "plain text, one equation per file",
        },
        "supported_symbols": "0123456789x+-=/() ",
        "generation_time_seconds": round(elapsed_total, 2),
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset built in {elapsed_total:.1f}s")
    print(f"  Output: {out_dir}")
    print(f"  Train:  {train_count}")
    print(f"  Valid:  {valid_count}")
    print(f"  Test:   {test_count}")
    print(f"  Total:  {samples}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build synthetic equation dataset for CNN-CTC training.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="training/datasets/synthetic_v1",
        help="Output directory for the dataset.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Total number of samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Image height in pixels.",
    )
    args = parser.parse_args()

    build_dataset(
        out_dir=args.out,
        samples=args.samples,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()

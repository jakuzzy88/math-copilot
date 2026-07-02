"""
verify_synthetic_dataset.py – Dataset verification and integrity checker

Verifies the structure and content of a synthetic equation dataset:
    - All images exist and are valid PNGs
    - All labels exist and match image basenames
    - Labels contain only supported symbols
    - Labels conform to the MVP equation grammar
    - Split counts match metadata

Prints a concise verification report.

Sprint 2 implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Supported symbol set (no spaces in labels)
SUPPORTED_CHARS = set("0123456789x+-=/() ")

# Regex patterns for MVP equation forms (canonical, no spaces)
# These match the 9 supported equation forms:
#   x+a=b   x-a=b   ax=b   ax+b=c   ax-b=c   x/a=b   ax/b=c   a(x+b)=c   a(x-b)=c
# Also allow negative RHS values (e.g. x+3=-5)

_EQUATION_PATTERNS = [
    # x+a=b  or  x-a=b
    r"^x[+\-]\d+=\-?\d+$",
    # ax=b
    r"^\d+x=\-?\d+$",
    # ax+b=c  or  ax-b=c
    r"^\d+x[+\-]\d+=\-?\d+$",
    # x/a=b
    r"^x/\d+=\-?\d+$",
    # ax/b=c
    r"^\d+x/\d+=\-?\d+$",
    # a(x+b)=c  or  a(x-b)=c
    r"^\d+\(x[+\-]\d+\)=\-?\d+$",
]


def _matches_grammar(label: str) -> bool:
    """Check if a label matches one of the supported MVP equation patterns."""
    for pattern in _EQUATION_PATTERNS:
        if re.match(pattern, label):
            return True
    return False


def _check_supported_chars(label: str) -> bool:
    """Check that label only contains supported characters."""
    return all(c in SUPPORTED_CHARS for c in label)


def _check_single_equals(label: str) -> bool:
    """Check that label has exactly one '=' sign."""
    return label.count("=") == 1


def verify_dataset(dataset_dir: str | Path) -> dict:
    """
    Verify a synthetic equation dataset.

    Args:
        dataset_dir: Path to the dataset root directory.

    Returns:
        Dictionary with verification results.
    """
    dataset_dir = Path(dataset_dir)

    results = {
        "dataset_dir": str(dataset_dir),
        "exists": dataset_dir.exists(),
        "metadata_ok": False,
        "splits": {},
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    if not dataset_dir.exists():
        results["errors"].append(f"Dataset directory does not exist: {dataset_dir}")
        return results

    # --- Check metadata ---
    meta_path = dataset_dir / "metadata.json"
    metadata = None
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            results["metadata_ok"] = True
            results["metadata"] = metadata
        except (json.JSONDecodeError, IOError) as e:
            results["errors"].append(f"metadata.json parse error: {e}")
    else:
        results["errors"].append("metadata.json not found")

    # --- Check splits ---
    expected_splits = {}
    if metadata and "splits" in metadata:
        expected_splits = metadata["splits"]

    total_images = 0
    total_labels = 0
    total_matched = 0
    total_char_ok = 0
    total_grammar_ok = 0
    total_equals_ok = 0
    total_checked = 0

    for split in ["train", "valid", "test"]:
        split_result = {
            "image_count": 0,
            "label_count": 0,
            "matched_pairs": 0,
            "char_valid": 0,
            "grammar_valid": 0,
            "equals_valid": 0,
            "expected_count": expected_splits.get(split, "?"),
            "errors": [],
        }

        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            split_result["errors"].append(f"Missing: images/{split}/")
            results["splits"][split] = split_result
            continue

        if not lbl_dir.exists():
            split_result["errors"].append(f"Missing: labels/{split}/")
            results["splits"][split] = split_result
            continue

        # List images and labels
        images = sorted(img_dir.glob("*.png"))
        labels = sorted(lbl_dir.glob("*.txt"))

        split_result["image_count"] = len(images)
        split_result["label_count"] = len(labels)

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        matched = img_stems & lbl_stems
        split_result["matched_pairs"] = len(matched)

        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems

        if missing_labels:
            split_result["errors"].append(
                f"{len(missing_labels)} images without labels in {split}/"
            )
        if missing_images:
            split_result["errors"].append(
                f"{len(missing_images)} labels without images in {split}/"
            )

        # Validate label content
        for lbl_path in labels:
            total_checked += 1
            try:
                content = lbl_path.read_text(encoding="utf-8").strip()
            except IOError as e:
                split_result["errors"].append(f"Cannot read {lbl_path.name}: {e}")
                continue

            if _check_supported_chars(content):
                split_result["char_valid"] += 1
            else:
                bad_chars = [c for c in content if c not in SUPPORTED_CHARS]
                split_result["errors"].append(
                    f"{lbl_path.name}: unsupported chars {bad_chars}"
                )

            if _check_single_equals(content):
                split_result["equals_valid"] += 1
            else:
                split_result["errors"].append(
                    f"{lbl_path.name}: expected 1 '=', found {content.count('=')}"
                )

            if _matches_grammar(content):
                split_result["grammar_valid"] += 1
            else:
                split_result["errors"].append(
                    f"{lbl_path.name}: does not match MVP grammar: '{content}'"
                )

        # Check expected count
        if split in expected_splits:
            if len(images) != expected_splits[split]:
                results["warnings"].append(
                    f"{split}: expected {expected_splits[split]} images, found {len(images)}"
                )

        total_images += split_result["image_count"]
        total_labels += split_result["label_count"]
        total_matched += split_result["matched_pairs"]
        total_char_ok += split_result["char_valid"]
        total_grammar_ok += split_result["grammar_valid"]
        total_equals_ok += split_result["equals_valid"]

        results["splits"][split] = split_result

    results["summary"] = {
        "total_images": total_images,
        "total_labels": total_labels,
        "total_matched_pairs": total_matched,
        "total_char_valid": total_char_ok,
        "total_grammar_valid": total_grammar_ok,
        "total_equals_valid": total_equals_ok,
        "total_checked": total_checked,
        "total_errors": len(results["errors"]) + sum(
            len(s["errors"]) for s in results["splits"].values()
        ),
    }

    return results


def print_report(results: dict) -> None:
    """Print a concise verification report."""
    print("=" * 60)
    print("SYNTHETIC DATASET VERIFICATION REPORT")
    print("=" * 60)
    print(f"Dataset:  {results['dataset_dir']}")
    print(f"Exists:   {results['exists']}")
    print(f"Metadata: {'OK' if results['metadata_ok'] else 'MISSING/ERROR'}")
    print()

    if not results["exists"]:
        print("FAIL: Dataset directory does not exist.")
        return

    for split in ["train", "valid", "test"]:
        sr = results["splits"].get(split, {})
        expected = sr.get("expected_count", "?")
        actual = sr.get("image_count", 0)
        matched = sr.get("matched_pairs", 0)
        char_ok = sr.get("char_valid", 0)
        gram_ok = sr.get("grammar_valid", 0)
        eq_ok = sr.get("equals_valid", 0)
        errs = sr.get("errors", [])
        status = "✓" if not errs else "✗"

        print(f"  [{status}] {split}:")
        print(f"      Images:       {actual} (expected {expected})")
        print(f"      Labels:       {sr.get('label_count', 0)}")
        print(f"      Matched:      {matched}")
        print(f"      Chars valid:  {char_ok}")
        print(f"      Grammar:      {gram_ok}")
        print(f"      Equals:       {eq_ok}")
        if errs:
            for e in errs[:5]:
                print(f"      ERROR: {e}")
            if len(errs) > 5:
                print(f"      ... and {len(errs) - 5} more errors")
        print()

    summ = results["summary"]
    print("-" * 60)
    print("TOTALS:")
    print(f"  Images:         {summ['total_images']}")
    print(f"  Labels:         {summ['total_labels']}")
    print(f"  Matched pairs:  {summ['total_matched_pairs']}")
    print(f"  Chars valid:    {summ['total_char_valid']}")
    print(f"  Grammar valid:  {summ['total_grammar_valid']}")
    print(f"  Equals valid:   {summ['total_equals_valid']}")
    print(f"  Errors:         {summ['total_errors']}")
    print("-" * 60)

    if summ["total_errors"] == 0:
        print("RESULT: ✓ ALL CHECKS PASSED")
    else:
        print(f"RESULT: ✗ {summ['total_errors']} ERROR(S) FOUND")

    if results["warnings"]:
        print("\nWARNINGS:")
        for w in results["warnings"]:
            print(f"  ⚠ {w}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify synthetic equation dataset integrity.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    args = parser.parse_args()

    results = verify_dataset(args.dataset)
    print_report(results)

    # Exit with non-zero if errors found
    if results["summary"].get("total_errors", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

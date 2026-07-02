"""
test_synthetic_renderer.py – Tests for the synthetic equation renderer.

Validates:
    - Renderer creates a valid, non-empty PNG image
    - Image dimensions match requested size
    - Image is greyscale (mode 'L')
    - render_and_save creates matching image/label pairs
    - Deterministic seed produces identical images
    - Label file content matches input

Sprint 2 tests.
"""

import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.synthetic_renderer.render_equation import (
    render_equation_image,
    render_and_save,
)


class TestSyntheticRenderer:
    """Tests for the synthetic renderer module."""

    def test_creates_non_empty_image(self):
        """Renderer creates a non-empty image."""
        img = render_equation_image("3x+4=10", seed=42)
        assert img is not None
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_default_dimensions(self):
        """Default image dimensions are 512×128."""
        img = render_equation_image("x+5=9")
        assert img.size == (512, 128)

    def test_custom_dimensions(self):
        """Custom width/height is respected."""
        img = render_equation_image("2x=8", width=256, height=64, seed=42)
        assert img.size == (256, 64)

    def test_greyscale_mode(self):
        """Image is in greyscale mode 'L'."""
        img = render_equation_image("x/2=5", seed=42)
        assert img.mode == "L"

    def test_image_not_all_white(self):
        """Rendered image is not all-white (text was drawn)."""
        img = render_equation_image("3x+4=10", seed=42)
        import numpy as np
        arr = np.array(img)
        # Should have some dark pixels (< 200)
        assert arr.min() < 200, "Image appears to be all white – no text rendered"

    def test_deterministic_seed(self):
        """Same seed produces identical images."""
        img1 = render_equation_image("2(x+1)=10", seed=42)
        img2 = render_equation_image("2(x+1)=10", seed=42)
        import numpy as np
        assert np.array_equal(np.array(img1), np.array(img2))

    def test_different_seeds_differ(self):
        """Different seeds produce different images."""
        img1 = render_equation_image("x+1=5", seed=42)
        img2 = render_equation_image("x+1=5", seed=99)
        import numpy as np
        assert not np.array_equal(np.array(img1), np.array(img2))

    def test_render_and_save_creates_files(self, tmp_path: Path):
        """render_and_save creates both image and label files."""
        img_path = tmp_path / "images" / "test.png"
        lbl_path = tmp_path / "labels" / "test.txt"

        render_and_save("5x-2=18", img_path, lbl_path, seed=42)

        assert img_path.exists()
        assert lbl_path.exists()

    def test_saved_image_is_valid_png(self, tmp_path: Path):
        """Saved file is a valid PNG that can be opened."""
        img_path = tmp_path / "test.png"
        lbl_path = tmp_path / "test.txt"

        render_and_save("3x/4=6", img_path, lbl_path, seed=42)

        img = Image.open(img_path)
        assert img.format == "PNG"
        assert img.mode == "L"

    def test_label_content_matches(self, tmp_path: Path):
        """Label file content matches the input equation string."""
        img_path = tmp_path / "test.png"
        lbl_path = tmp_path / "test.txt"
        label = "2(x+3)=20"

        render_and_save(label, img_path, lbl_path, seed=42)

        saved_label = lbl_path.read_text(encoding="utf-8")
        assert saved_label == label

    def test_renders_all_supported_forms(self):
        """Renderer handles all supported equation forms without error."""
        forms = [
            "x+3=7",
            "x-2=5",
            "4x=12",
            "3x+4=10",
            "5x-2=18",
            "x/2=5",
            "3x/4=6",
            "2(x+1)=10",
            "3(x-2)=9",
        ]
        for eq in forms:
            img = render_equation_image(eq, seed=42)
            assert img is not None, f"Failed to render '{eq}'"
            assert img.size == (512, 128)


class TestDatasetBuilder:
    """Integration tests: dataset builder creates matching pairs."""

    def test_build_small_dataset(self, tmp_path: Path):
        """Building a small dataset creates correct file structure."""
        from training.build_synthetic_dataset import build_dataset

        metadata = build_dataset(
            out_dir=tmp_path / "test_dataset",
            samples=20,
            seed=42,
        )

        ds_dir = tmp_path / "test_dataset"

        # Check metadata
        assert (ds_dir / "metadata.json").exists()
        assert metadata["total_samples"] == 20

        # Check splits
        train_count = metadata["splits"]["train"]
        valid_count = metadata["splits"]["valid"]
        test_count = metadata["splits"]["test"]
        assert train_count + valid_count + test_count == 20

        # Check files
        for split in ["train", "valid", "test"]:
            img_dir = ds_dir / "images" / split
            lbl_dir = ds_dir / "labels" / split
            assert img_dir.exists()
            assert lbl_dir.exists()

            images = list(img_dir.glob("*.png"))
            labels = list(lbl_dir.glob("*.txt"))
            assert len(images) == len(labels)

            # Check basenames match
            img_stems = {p.stem for p in images}
            lbl_stems = {p.stem for p in labels}
            assert img_stems == lbl_stems

    def test_matching_image_label_content(self, tmp_path: Path):
        """Each label file matches a valid equation and its image exists."""
        from training.build_synthetic_dataset import build_dataset

        build_dataset(
            out_dir=tmp_path / "test_ds",
            samples=10,
            seed=42,
        )

        ds_dir = tmp_path / "test_ds"
        for split in ["train", "valid", "test"]:
            lbl_dir = ds_dir / "labels" / split
            img_dir = ds_dir / "images" / split

            for lbl_path in lbl_dir.glob("*.txt"):
                content = lbl_path.read_text(encoding="utf-8").strip()
                assert len(content) > 0, f"Empty label: {lbl_path}"
                assert "=" in content, f"No '=' in label: {content}"

                img_path = img_dir / f"{lbl_path.stem}.png"
                assert img_path.exists(), f"Missing image for {lbl_path.name}"

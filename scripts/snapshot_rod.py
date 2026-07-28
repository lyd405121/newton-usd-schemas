"""
Load every .usda in asset directory with simulate_rod.RodSimulation,
simulate 1 frame, save a screenshot.

Usage:
    conda run -n newton python scripts/snapshot_rod.py
    conda run -n newton python scripts/snapshot_rod.py --asset-dir /path/to/assets
    conda run -n newton python scripts/snapshot_rod.py --out /path/to/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Remove repo root from sys.path to avoid shadowing conda-installed newton
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path = [p for p in sys.path if Path(p).resolve() != REPO_ROOT]

import newton.viewer  # noqa: E402
from simulate_rod import RodSimulation  # noqa: E402

import newton_usd_schemas  # noqa: E402, F401

DEFAULT_ASSET_DIR = REPO_ROOT / "asset"
DEFAULT_OUT = SCRIPT_DIR / "snapshots_rod"


def snapshot(usd_path: Path, out_dir: Path, viewer) -> Path | None:
    print(f"  {usd_path.name} ...", end=" ", flush=True)
    try:
        sim = RodSimulation(usd_path=str(usd_path), viewer=viewer, num_steps=1, substeps=10, iterations=5, fps=60)
    except Exception as e:
        print(f"SKIP ({e})")
        return None

    sim.step()
    sim.render()

    frame = viewer.get_frame().numpy()

    img_path = out_dir / f"{usd_path.stem}.png"
    try:
        from PIL import Image

        Image.fromarray(frame[..., :3]).save(img_path)
    except ImportError:
        import numpy as np

        img_path = img_path.with_suffix(".npy")
        np.save(img_path, frame)

    print(f"saved -> {img_path.name}")
    return img_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory containing .usda asset files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    usda_files = sorted(args.asset_dir.rglob("*.usda"))
    print(f"Found {len(usda_files)} .usda files, saving to {args.out}/\n")

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)

    saved = []
    for usd_path in usda_files:
        result = snapshot(usd_path, args.out, viewer)
        if result:
            saved.append(result)

    viewer.close()
    print(f"\nDone. {len(saved)}/{len(usda_files)} screenshots saved.")


if __name__ == "__main__":
    main()

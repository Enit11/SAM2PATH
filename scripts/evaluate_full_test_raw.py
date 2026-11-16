"""Export raw four-class argmax predictions for reproducible test evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

# The script is executed from ``scripts/``; make the repository root importable
# before importing the project inference module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam2path_inference import GlomeruliInference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--config", default="configs.Glomeruli_uni2_spherical_hrrefine")
    parser.add_argument("--project-root", default=r"C:\Users\ASUS\SAM2PATH")
    args = parser.parse_args()

    image_dir, mask_dir = Path(args.image_dir), Path(args.mask_dir)
    output_dir = Path(args.prediction_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in image_dir.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    inferencer = GlomeruliInference(
        args.checkpoint, args.project_root, config_module=args.config
    )
    inferencer.load_model()
    print(f"[INFO] Exporting raw argmax masks for {len(names)} test images.")
    for index, name in enumerate(names, start=1):
        mask_path = mask_dir / name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        image = inferencer.read_rgb_img(str(image_dir / name))
        image_tensor = inferencer.preprocess_image(image)
        probabilities = inferencer.predict_probabilities(image_tensor)
        prediction = probabilities.argmax(axis=0).astype("uint8")
        if not cv2.imwrite(str(output_dir / name), prediction):
            raise RuntimeError(f"Failed to write prediction for {name}")
        print(f"[{index}/{len(names)}] {name}", flush=True)


if __name__ == "__main__":
    main()

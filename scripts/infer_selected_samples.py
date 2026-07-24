import os
import sys

PROJECT_ROOT = r"C:\Users\ASUS\SAM2PATH"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sam2path_inference import GlomeruliInference


def resolve_sample_name(name: str, image_dir: str) -> str | None:
    stem = name if name.lower().endswith(".png") else f"{name}.png"
    normalized_stem = stem.replace("-", "").lower()
    direct = os.path.join(image_dir, stem)
    if os.path.exists(direct):
        return stem

    candidates = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(".png") and f.replace("-", "").lower() == normalized_stem
    )
    if candidates:
        return candidates[0]

    if stem.startswith("05452"):
        alt = stem.replace("05452", "08452", 1)
        if os.path.exists(os.path.join(image_dir, alt)):
            return alt
        normalized_alt = alt.replace("-", "").lower()
        candidates = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith(".png") and f.replace("-", "").lower() == normalized_alt
        )
        if candidates:
            return candidates[0]

    return None


def main():
    project_root = PROJECT_ROOT
    ckpt_path = (
        r"C:\Users\ASUS\SAM2PATH\outputs\glomeruli\run\glomeruli_10pct_mt_scfpb_sipp_seed42\Logs"
        r"\20260722-003303\sam2path\version_0\checkpoints\epoch=99-step=17700.ckpt"
    )
    image_dir = r"C:\Users\ASUS\SAM2PATH\dataset\test\image"
    mask_dir = r"C:\Users\ASUS\SAM2PATH\dataset\test\mask"
    output_dir = r"C:\Users\ASUS\SAM2PATH\vis_compare\selected_test_compare_sipp_instance_refined"

    requested = [
        "05452-PAS-04",
        "08626PAS-02",
        "08626PAS-04",
        "09156PAS-03",
        "09147PAS-01",
    ]

    os.makedirs(output_dir, exist_ok=True)
    inferencer = GlomeruliInference(ckpt_path=ckpt_path, project_root=project_root)
    inferencer.load_model()

    resolved = []
    missing = []
    for name in requested:
        actual = resolve_sample_name(name, image_dir)
        if actual is None:
            missing.append(name)
        else:
            resolved.append((name, actual))

    if missing:
        print("[WARN] Missing samples:", ", ".join(missing))
    print("[INFO] Using samples:")
    for requested_name, actual_name in resolved:
        print(f"  {requested_name} -> {actual_name}")

    for idx, (requested_name, actual_name) in enumerate(resolved):
        image_path = os.path.join(image_dir, actual_name)
        mask_path = os.path.join(mask_dir, actual_name)
        result = inferencer.infer_image(image_path, mask_path)
        canvas = inferencer.create_comparison_image(actual_name, result)
        out_path = os.path.join(output_dir, f"selected_{idx:02d}_{actual_name}")
        import cv2

        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        inferencer.save_probability_maps(output_dir, actual_name, result)
        print(f"[OK] {requested_name} saved to {out_path}")


if __name__ == "__main__":
    sys.exit(main())

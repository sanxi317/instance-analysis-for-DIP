"""Copy the existing experiment images; make identical-coordinate crops.

No denoising, interpolation, metric calculation, or contrast changes are done.
"""
from pathlib import Path
from PIL import Image
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ASSETS = HERE / "assets"
ROI = (80, 150, 240, 310)  # x0, y0, x1, y1 in the original 512 x 512 image
FILES = {
    "ldct": "task2/images/ldct_input.png",
    "fdct": "task2/images/fdct_ground_truth.png",
    "mean": "task2/images/mean.png",
    "gaussian": "task2/images/gaussian.png",
    "adaptive_median": "task2/images/adaptive_median.png",
    "nlm": "task2/images/nlm.png",
    "wavelet": "task2/images/wavelet.png",
    "nearest": "task3/output/nearest_image.png",
    "bilinear": "task3/output/bilinear_image.png",
    "bicubic": "task3/output/bicubic_image.png",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

ASSETS.mkdir(exist_ok=True)
manifest = {"roi_xyxy": ROI, "crop_description": "Local structure only; no lesion annotation", "files": []}
for name, relative in FILES.items():
    source = PROJECT / relative
    target = ASSETS / f"{name}.png"
    shutil.copy2(source, target)
    with Image.open(source) as im:
        if im.size != (512, 512) or im.mode != "L":
            raise ValueError(f"Unexpected image: {source}, {im.size}, {im.mode}")
        crop = ASSETS / f"{name}_crop.png"
        im.crop(ROI).save(crop)
    manifest["files"].append({"source": relative, "copy": target.name, "sha256": sha(source), "crop": crop.name, "crop_sha256": sha(crop)})
(HERE / "asset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Prepared {len(FILES)} original images and same-coordinate crops")

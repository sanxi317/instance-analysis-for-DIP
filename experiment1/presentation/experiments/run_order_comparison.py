#!/usr/bin/env python3
"""Rerun direct fourfold interpolation and both processing orders.

Uses the task2 uniform-window uint8 pair, not the old task5 truncation branch.
Keeps old task scripts/results untouched. Saves every 2048x2048 output and a
single bicubic-upsampled FDCT reference; that reference is not acquired HR GT.
"""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import skimage
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

HERE = Path(__file__).resolve().parent
PRESENTATION = HERE.parent
PROJECT = PRESENTATION.parent
INPUT_DIR = PRESENTATION / "assets"
OUTPUT = HERE / "results"
SCALE = 4
PARAMETERS = {"h": 4, "templateWindowSize": 7, "searchWindowSize": 21}
ROI512 = (80, 150, 240, 310)
ROI2048 = tuple(v * SCALE for v in ROI512)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode != "L" or im.size != (512, 512):
            raise ValueError(f"Expected 512x512 grayscale uint8 input: {path}")
        return np.array(im)

def denoise_nlm(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, None, **PARAMETERS)

def up4(image: np.ndarray, method: int = cv2.INTER_CUBIC) -> np.ndarray:
    return cv2.resize(image, None, fx=SCALE, fy=SCALE, interpolation=method)

def save(name: str, array: np.ndarray) -> dict:
    path = OUTPUT / f"{name}.png"
    Image.fromarray(array).save(path)
    return {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype),
            "min": int(array.min()), "max": int(array.max()), "sha256": sha(path)}

def evaluate(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    assert reference.shape == candidate.shape == (2048, 2048)
    return (float(peak_signal_noise_ratio(reference, candidate, data_range=255)),
            float(structural_similarity(reference, candidate, data_range=255)))

def main() -> None:
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    cv2.setNumThreads(2)
    OUTPUT.mkdir(exist_ok=True)
    ldct_path, fdct_path = INPUT_DIR / "ldct.png", INPUT_DIR / "fdct.png"
    ldct, fdct = read_image(ldct_path), read_image(fdct_path)
    print("Loaded the paired 512x512 lung-window inputs", flush=True)
    dicom_validation = {"available": False}
    fd = PROJECT / "full_3mm/L067/full_3mm/L067_FD_3_1.CT.0002.0001.2015.12.22.18.12.07.5968.358090401.IMA"
    ld = PROJECT / "quarter_3mm/L067/quarter_3mm/L067_QD_3_1.CT.0004.0001.2015.12.22.18.12.56.428910.358293547.IMA"
    if fd.is_file() and ld.is_file():
        import SimpleITK as sitk
        geometries = []
        for path, expected in [(fd, fdct), (ld, ldct)]:
            image = sitk.ReadImage(str(path))
            hu = np.squeeze(sitk.GetArrayFromImage(image)).astype(np.float32)
            normalized = np.rint((np.clip(hu, -1350., 150.) + 1350.) / 1500. * 255.).astype(np.uint8)
            assert np.array_equal(normalized, expected), f"PNG/DICOM mismatch: {path}"
            geometries.append((image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()))
        assert geometries[0] == geometries[1], "Paired DICOM geometry differs"
        dicom_validation = {"available": True, "exact_png_match": True,
                            "geometries_match": True, "fdct_sha256": sha(fd), "ldct_sha256": sha(ld),
                            "SimpleITK_version": sitk.Version_VersionString()}

    native_nlm = denoise_nlm(ldct)
    old_nlm = read_image(INPUT_DIR / "nlm.png")
    native_match = bool(np.array_equal(native_nlm, old_nlm))
    if not native_match:
        raise ValueError("Current selected NLM does not match the existing task2 result")
    reference = up4(fdct)
    nearest = up4(native_nlm, cv2.INTER_NEAREST)
    bilinear = up4(native_nlm, cv2.INTER_LINEAR)
    # Both orders have the same 512 input and 2048 output and common reference.
    A = up4(denoise_nlm(ldct))
    print("Saved-order A computed; computing interpolation before denoising (B)", flush=True)
    B = denoise_nlm(up4(ldct))
    print("Both orders computed; calculating metrics", flush=True)
    arrays = {"reference_fdct4": reference, "nlm_native": native_nlm,
              "nlm4_nearest": nearest, "nlm4_bilinear": bilinear,
              "order_A_nlm_then_bicubic": A, "order_B_bicubic_then_nlm": B}
    output_records = {name: save(name, array) for name, array in arrays.items()}
    candidates = [
        ("interpolation", "NLM + Nearest", "nlm4_nearest", nearest),
        ("interpolation", "NLM + Bilinear", "nlm4_bilinear", bilinear),
        ("interpolation", "NLM + Bicubic", "order_A_nlm_then_bicubic", A),
        ("order", "A: NLM then Bicubic", "order_A_nlm_then_bicubic", A),
        ("order", "B: Bicubic then NLM", "order_B_bicubic_then_nlm", B),
    ]
    rows = []
    for category, method, name, image in candidates:
        psnr, ssim = evaluate(reference, image)
        rows.append({"category": category, "method": method, "image": f"{name}.png",
                     "PSNR_dB": psnr, "SSIM": ssim})
        print(f"{method}: PSNR={psnr:.6f}, SSIM={ssim:.8f}", flush=True)
    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Exact crops only; no contrast adjustment or further image interpolation.
    crops = {"direct_nearest_crop": nearest, "direct_bilinear_crop": bilinear,
             "direct_bicubic_crop": A, "order_B_crop": B, "reference_fdct4_crop": reference}
    crop_records = {}
    for name, array in crops.items():
        path = INPUT_DIR / f"{name}.png"
        Image.fromarray(array).crop(ROI2048).save(path)
        crop_records[name] = {"file": path.name, "sha256": sha(path), "roi_xyxy": ROI2048,
                              "shape": [640, 640]}
    manifest = {"experiment_label": "2026-10-13 lecture supplement",
                "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - start,
                "command": [sys.executable, *sys.argv], "script_sha256": sha(Path(__file__)),
                "python": sys.version, "platform": platform.platform(),
                "packages": {"numpy": np.__version__, "opencv": cv2.__version__, "skimage": skimage.__version__},
                "cv_threads": 2, "inputs": {"ldct": {"file": "ldct.png", "sha256": sha(ldct_path)},
                                             "fdct": {"file": "fdct.png", "sha256": sha(fdct_path)}},
                "dicom_validation": dicom_validation, "native_nlm_matches_saved_task2": native_match,
                "parameters": PARAMETERS, "scale_per_axis": SCALE,
                "normalization": {"wl_hu": -600, "ww_hu": 1500, "uint8_quantization": "np.rint"},
                "reference": {"description": "FDCT upsampled with INTER_CUBIC, not acquired high-resolution GT",
                              "file": "reference_fdct4.png", "shape": [2048, 2048], "data_range": 255},
                "order_A": "up4(denoise_nlm(ldct))", "order_B": "denoise_nlm(up4(ldct))",
                "outputs": output_records, "crops": crop_records,
                "metrics": rows, "limitations": ["single L067 slice, no lesion annotation",
                    "reference interpolation may influence interpolation ranking",
                    "same NLM pixel windows cover different physical extents in A and B",
                    "not an exhaustive search of all denoiser/interpolator combinations"]}
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed and saved execution evidence in {OUTPUT}", flush=True)

if __name__ == "__main__":
    main()

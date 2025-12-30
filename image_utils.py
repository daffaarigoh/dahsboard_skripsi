from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
from PIL import Image


class ImageUtils:
    """Utilitas untuk image I/O dan metrik waktu/FPS."""

    ALLOWED_EXTS = {"jpg", "jpeg", "png"}
    ALLOWED_MIME = {"image/jpeg", "image/png"}

    @staticmethod
    def open_image(file) -> Image.Image:
        return Image.open(file).convert("RGB")

    @staticmethod
    def calculate_inference_time(time_start: float, time_end: float) -> float:
        return (time_end - time_start) * 1000.0  # ms

    @staticmethod
    def calculate_fps(time_start: float, time_end: float) -> float:
        dt = time_end - time_start
        return (1.0 / dt) if dt > 0 else 0.0

    @staticmethod
    def result_plot_to_rgb(result) -> np.ndarray:
        plotted_bgr = result.plot()
        return plotted_bgr[:, :, ::-1]

    @staticmethod
    def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
        name = getattr(uploaded_file, "name", "")
        mime = getattr(uploaded_file, "type", "")

        ext = ""
        if "." in name:
            ext = name.rsplit(".", 1)[-1].lower()

        if ext not in ImageUtils.ALLOWED_EXTS:
            return False, f"File '{name}' tidak valid. Hanya menerima: {', '.join(sorted(ImageUtils.ALLOWED_EXTS))}."
        if mime and mime not in ImageUtils.ALLOWED_MIME:
            return False, f"File '{name}' MIME type '{mime}' tidak didukung. Hanya JPEG/PNG."
        return True, ""

    @staticmethod
    def try_load_metrics_from_weights(weights_path: str) -> Tuple[Optional[Dict[str, float]], str]:
        """Baca metrik offline (Dice/IoU/mAP50) dari JSON di folder weights."""
        try:
            p = Path(weights_path)
            if not p.exists():
                return None, "weights_path belum valid."

            candidates = [
                p.with_name(f"{p.stem}_metrics.json"),
                p.with_name("metrics.json"),
                p.with_name("model_metrics.json"),
            ]

            for c in candidates:
                if c.is_file():
                    with c.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    def pick(*keys):
                        for k in keys:
                            if k in data:
                                return data[k]
                        return None

                    dice = pick("dice", "Dice", "dice_score")
                    iou = pick("iou", "IoU", "iou_score")
                    map50 = pick("map50", "mAP50", "map_50", "metrics/mAP50(M)")

                    metrics: Dict[str, float] = {}
                    if dice is not None:
                        metrics["dice"] = float(dice)
                    if iou is not None:
                        metrics["iou"] = float(iou)
                    if map50 is not None:
                        metrics["map50"] = float(map50)

                    if not metrics:
                        return None, f"File metrics ditemukan ({c.name}) tapi key (dice/iou/map50) tidak ada."
                    return metrics, f"Loaded dari {c.name}"

            return None, "Tidak menemukan file metrics JSON di folder weights."
        except Exception as e:
            return None, f"Gagal membaca metrics JSON: {e}"

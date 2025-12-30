from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import streamlit as st
from PIL import Image
from ultralytics import YOLO


class ModelHandler:
    """Wrapper untuk load & inferensi model YOLO."""

    def __init__(self) -> None:
        self.model_instance: Optional[YOLO] = None

    @staticmethod
    @st.cache_resource
    def _load_model_cached(path: str) -> YOLO:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File model tidak ditemukan: {path}")
        return YOLO(str(p))

    def load_model(self, path: str) -> YOLO:
        self.model_instance = self._load_model_cached(path)
        return self.model_instance

    def predict(self, image: Image.Image, imgsz: int = 640) -> Any:
        """predict(image, imgsz=640) -> Results"""
        if self.model_instance is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")
        return self.model_instance.predict(image, imgsz=imgsz, verbose=False)

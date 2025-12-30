from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from image_utils import ImageUtils
from model_handler import ModelHandler


@dataclass
class PolypSegmentationApp:
    """Aplikasi Streamlit utama (disusun mengikuti sequence diagram)."""

    weights_path: str = "best.pt"
    uploaded_files: List = field(default_factory=list)

    # sesuai sequence diagram: imgsz=640 (fixed)
    imgsz: int = 640

    # Metrik offline di sidebar
    dice: Optional[float] = None
    iou: Optional[float] = None
    map50: Optional[float] = None

    # dependency
    model_handler: ModelHandler = field(default_factory=ModelHandler)

    def render_sidebar(self) -> None:
        st.sidebar.header("⚙️ Pengaturan")

        self.weights_path = st.sidebar.text_input(
            "Path model (.pt)",
            value=self.weights_path,
            help="Letakkan file best.pt di folder yang sama dengan app.py atau tuliskan path lengkapnya.",
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Metrik Evaluasi Model")

        # --- Ambil metrik hanya dari JSON ---
        metrics, _note = ImageUtils.try_load_metrics_from_weights(self.weights_path)

        if metrics is not None:
            self.dice = metrics.get("dice", None)
            self.iou  = metrics.get("iou", None)
            self.map50 = metrics.get("map50", None)
        else:
            self.dice, self.iou, self.map50 = None, None, None
            st.sidebar.warning("metrics.json tidak ditemukan / format tidak sesuai.")

        # --- Tampilkan hanya 3 metrik ---
        st.sidebar.write(f"**Dice**: {self.dice if self.dice is not None else '-'}")
        st.sidebar.write(f"**IoU**: {self.iou if self.iou is not None else '-'}")
        st.sidebar.write(f"**mAP50**: {self.map50 if self.map50 is not None else '-'}")

        st.sidebar.markdown("---")
        

    def validate_files_or_stop(self) -> None:
        invalid_msgs = []
        for f in self.uploaded_files:
            ok, msg = ImageUtils.validate_uploaded_file(f)
            if not ok:
                invalid_msgs.append(msg)

        if invalid_msgs:
            st.error("❌ Ada file yang tidak valid:\n- " + "\n- ".join(invalid_msgs))
            st.stop()

    def calculate_summary(self, all_ms: List[float], all_fps: List[float]) -> Dict[str, float]:
        return {
            "avg_ms": float(np.mean(all_ms)) if all_ms else 0.0,
            "avg_fps": float(np.mean(all_fps)) if all_fps else 0.0,
        }

    def run_app(self) -> None:
        st.set_page_config(page_title="Segmentasi Polip Kolonoskopi", layout="wide")
        st.title("🩺 Segmentasi Polip Kolonoskopi (YOLOv11-Seg)")

        self.render_sidebar()

        st.subheader("1️⃣ Upload Banyak Citra (JPG/JPEG/PNG)")
        self.uploaded_files = st.file_uploader(
            "Pilih satu atau beberapa file gambar",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if not self.uploaded_files:
            st.info("Silakan upload satu atau beberapa gambar kolonoskopi terlebih dahulu.")
            return

        # Validasi ketat: jika ada file invalid -> error & stop
        self.validate_files_or_stop()

        # Load model
        try:
            self.model_handler.load_model(self.weights_path)
        except Exception as e:
            st.error(f"Gagal load model: {e}")
            return

        all_fps: List[float] = []
        all_ms: List[float] = []
        rows = []

        # loop per citra
        for idx, uploaded_file in enumerate(self.uploaded_files, start=1):
            st.markdown("---")
            st.markdown(f"### 🖼️ Citra #{idx}: {uploaded_file.name}")

            image: Image.Image = ImageUtils.open_image(uploaded_file)

            col1, col2 = st.columns(2)

            # 1) Tampilkan Original (col1)
            with col1:
                st.image(image, caption="Gambar Original", use_column_width=True)

            # 2) predict(image, imgsz=640)
            with st.spinner("Sedang memproses citra..."):
                t0 = time.time()
                results = self.model_handler.predict(image=image, imgsz=self.imgsz)
                t1 = time.time()

            result = results[0]

            # 3) Tampilkan Hasil Segmentasi (col2)
            with col2:
                plotted_rgb = ImageUtils.result_plot_to_rgb(result)
                st.image(plotted_rgb, caption="Hasil Segmentasi", use_column_width=True)

            # 4) tampilkan waktu & FPS
            ms = ImageUtils.calculate_inference_time(t0, t1)
            fps = ImageUtils.calculate_fps(t0, t1)
            st.write(f"- Waktu pemrosesan: **{ms:.2f} ms/frame**")
            st.write(f"- FPS: **{fps:.2f} frame/detik**")

            num_polyp = None
            if getattr(result, "masks", None) is not None:
                try:
                    num_polyp = int(len(result.masks.data))
                    st.write(f"- Jumlah objek polip tersegmentasi: **{num_polyp}**")
                except Exception:
                    pass

            # 5) simpan metrik ke list
            all_ms.append(ms)
            all_fps.append(fps)
            rows.append(
                {"idx": idx, "file": uploaded_file.name, "ms_per_frame": ms, "fps": fps, "num_polyp": num_polyp}
            )

        # Hitung rata-rata
        summary = self.calculate_summary(all_ms, all_fps)

        # Tampilkan rangkuman metrik total
        st.markdown("---")
        st.markdown("## 📊 Rangkuman Metrik Total")
        st.write(f"- **Rata-rata waktu pemrosesan**: **{summary['avg_ms']:.2f} ms/frame**")
        st.write(f"- **Rata-rata FPS**: **{summary['avg_fps']:.2f} frame/detik**")

#!/usr/bin/env python3
"""Interactive Qt particle viewer for Armor Penetration Lab.

Usage:
    python viewer_qt.py RUN_DIR

The viewer reads the solver's frames.bin format lazily, one frame at a time.
It requires PySide6 and NumPy.
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


FIELD_INFO = {
    "dmg": ("Damage", "D"),
    "spd": ("Speed", "m/s"),
    "P": ("Pressure", "GPa"),
    "T": ("Temperature", "K"),
    "epsp": ("Plastic strain", ""),
    "svm": ("von Mises stress", "GPa"),
}
FIELD_INDICES = {
    2: {"spd": 2, "dmg": 3, "P": 4, "T": 5, "epsp": 6, "svm": 7},
    3: {"spd": 3, "dmg": 4, "P": 5, "T": 6, "epsp": 7, "svm": 8},
}


class DataSet:
    """Read metadata and provide lazy access to individual binary frames."""

    def __init__(self, run_dir: str | os.PathLike[str]):
        self.path = os.fspath(run_dir)
        self.meta = self._read_meta()
        self.dim = int(self.meta.get("DIM", "2"))
        if self.dim not in (2, 3):
            raise ValueError(f"Unsupported spatial dimension: {self.dim}")
        self.n = int(self.meta["N"])
        self.np = int(self.meta["NP"])
        expected_fields = 8 if self.dim == 2 else 9
        self.nf = int(self.meta.get("NFIELDS", str(expected_fields)))
        if self.n <= 0 or not 0 <= self.np <= self.n or self.nf != expected_fields:
            raise ValueError(
                f"Invalid metadata: N={self.n}, NP={self.np}, NFIELDS={self.nf}, DIM={self.dim}"
            )

        self.block = 4 + self.nf * self.n * 4
        self.frames_path = os.path.join(self.path, "frames.bin")
        size = os.path.getsize(self.frames_path)
        if size == 0 or size % self.block:
            raise ValueError(
                f"frames.bin is empty, truncated, or has an invalid size ({size} bytes)"
            )
        self.count = size // self.block
        self._file = open(self.frames_path, "rb")

    def _read_meta(self) -> dict[str, str]:
        result: dict[str, str] = {}
        with open(os.path.join(self.path, "meta.txt"), encoding="utf-8") as stream:
            for line in stream:
                if "=" in line:
                    key, value = line.rstrip().split("=", 1)
                    result[key] = value
        return result

    def frame(self, index: int) -> tuple[float, np.ndarray]:
        if not 0 <= index < self.count:
            raise IndexError(f"Frame index {index} is outside [0, {self.count - 1}]")
        self._file.seek(index * self.block)
        raw = self._file.read(self.block)
        if len(raw) != self.block:
            raise ValueError(f"Unexpected end of frames.bin at frame {index}")
        time_value = struct.unpack_from("<f", raw, 0)[0]
        fields = np.frombuffer(
            raw, dtype="<f4", offset=4, count=self.n * self.nf
        ).reshape(self.nf, self.n)
        return float(time_value), fields

    def close(self) -> None:
        self._file.close()


class Viewport(QWidget):
    """CPU-rendered orthographic particle view with a selectable particle index."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(720, 600)
        self.data: DataSet | None = None
        self.frame_index: int | None = None
        self.field = "dmg"
        self.yaw = 0.65
        self.pitch = 0.45
        self.zoom = 1.0
        self.pan = np.array([0.0, 0.0])
        self.slice_mm = 0.0
        self.drag = None
        self.limits = (0.0, 1.2)
        self.selected_particle: int | None = None
        self.on_pick = None
        self._projected = None

    def set_frame(self, data: DataSet, index: int) -> None:
        self.data = data
        self.frame_index = index
        self.update()

    def reset_camera(self) -> None:
        self.yaw, self.pitch, self.zoom = 0.65, 0.45, 1.0
        self.pan[:] = 0.0
        self.update()

    def _values(self, fields: np.ndarray) -> np.ndarray:
        index = FIELD_INDICES[self.data.dim][self.field]
        values = fields[index].astype(np.float64)
        if self.field in ("P", "svm"):
            values /= 1.0e9
        return values

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(14, 14, 22))
        if self.data is None or self.frame_index is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(20, 30, "Open a run directory to view results")
            return

        time_value, fields = self.data.frame(self.frame_index)
        dim = self.data.dim
        x = fields[0].astype(np.float64)
        y = fields[1].astype(np.float64)
        z = fields[2].astype(np.float64) if dim == 3 else np.zeros_like(x)
        values = self._values(fields)

        visible = np.ones(self.data.n, dtype=bool)
        if dim == 3 and self.slice_mm > 0.0:
            visible &= np.abs(z) <= self.slice_mm * 1.0e-3

        center_x = 0.5 * (float(x.min()) + float(x.max()))
        center_y = 0.5 * (float(y.min()) + float(y.max()))
        center_z = 0.5 * (float(z.min()) + float(z.max()))
        xx, yy, zz = x - center_x, y - center_y, z - center_z

        cos_yaw, sin_yaw = np.cos(self.yaw), np.sin(self.yaw)
        rotated_x = cos_yaw * xx - sin_yaw * zz
        rotated_z = sin_yaw * xx + cos_yaw * zz
        cos_pitch, sin_pitch = np.cos(self.pitch), np.sin(self.pitch)
        rotated_y = cos_pitch * yy - sin_pitch * rotated_z
        depth = sin_pitch * yy + cos_pitch * rotated_z

        extent = max(
            float(x.max() - x.min()),
            float(y.max() - y.min()),
            float(z.max() - z.min()),
            1.0e-9,
        )
        scale = min((self.width() - 48) / extent, (self.height() - 20) / extent) * self.zoom
        px = (self.width() * 0.5 + self.pan[0] + rotated_x * scale).astype(np.int32)
        py = (self.height() * 0.5 + self.pan[1] - rotated_y * scale).astype(np.int32)
        visible &= (px >= 1) & (px < self.width() - 42)
        visible &= (py >= 1) & (py < self.height() - 1)
        self._projected = (px, py, visible)

        low, high = self.limits
        normalized = np.clip((values - low) / max(high - low, 1.0e-12), 0.0, 1.0)
        palette = np.array(
            [[38, 26, 89], [26, 115, 242], [26, 230, 191], [250, 217, 38], [230, 64, 26]],
            dtype=np.float64,
        )
        position = normalized * (len(palette) - 1)
        palette_index = np.minimum(position.astype(np.int32), len(palette) - 2)
        blend = position - palette_index
        colors = (
            palette[palette_index] * (1.0 - blend[:, None])
            + palette[palette_index + 1] * blend[:, None]
        )

        depth_norm = (depth - float(depth.min())) / max(float(np.ptp(depth)), 1.0e-12)
        colors *= (0.52 + 0.48 * depth_norm[:, None])
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        colors[: self.data.np] = np.clip(colors[: self.data.np].astype(np.int16) + 28, 0, 255)
        damage_index = FIELD_INDICES[dim]["dmg"]
        if self.field == "dmg":
            colors[fields[damage_index] >= 1.0] = (255, 150, 20)

        image = np.full((self.height(), self.width(), 3), (14, 14, 22), dtype=np.uint8)
        for particle_index in np.argsort(depth):
            if not visible[particle_index]:
                continue
            radius = 2 if particle_index < self.data.np else 1
            ix, iy = px[particle_index], py[particle_index]
            image[
                max(0, iy - radius):min(self.height(), iy + radius + 1),
                max(0, ix - radius):min(self.width() - 41, ix + radius + 1),
            ] = colors[particle_index]

        qimage = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        painter.drawImage(0, 0, qimage)

        if self.selected_particle is not None and visible[self.selected_particle]:
            painter.setPen(QColor(255, 255, 255))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(px[self.selected_particle]) - 5,
                                int(py[self.selected_particle]) - 5, 10, 10)

        title, unit = FIELD_INFO[self.field]
        particle_count = int(visible.sum())
        failed_count = int((fields[damage_index] >= 1.0).sum())
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(12, 22, f"{title} [{unit}]  t = {time_value * 1.0e6:.2f} μs")
        painter.drawText(
            12,
            self.height() - 12,
            f"Visible particles: {particle_count}/{self.data.n}    D ≥ 1: {failed_count}",
        )
        painter.drawText(self.width() - 120, 20, f"{high:.3g} {unit}")
        painter.drawText(self.width() - 120, self.height() - 20, f"{low:.3g} {unit}")
        if dim == 3:
            painter.setPen(QColor(160, 175, 195))
            painter.drawText(
                18, 48,
                f"Camera: yaw={self.yaw:.2f}, pitch={self.pitch:.2f}, zoom={self.zoom:.2f}",
            )

    def mousePressEvent(self, event) -> None:
        self.drag = (
            event.position().x(),
            event.position().y(),
            self.yaw,
            self.pitch,
            self.pan.copy(),
            event.button(),
        )

    def mouseMoveEvent(self, event) -> None:
        if self.drag is None:
            return
        x0, y0, yaw0, pitch0, pan0, button = self.drag
        dx = event.position().x() - x0
        dy = event.position().y() - y0
        if button == Qt.MouseButton.LeftButton:
            self.yaw = yaw0 + dx * 0.012
            self.pitch = float(np.clip(pitch0 + dy * 0.012, -1.45, 1.45))
        elif button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self.pan = pan0 + np.array([dx, dy])
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.drag is not None and event.button() == Qt.MouseButton.LeftButton:
            x0, y0 = self.drag[0], self.drag[1]
            dx = event.position().x() - x0
            dy = event.position().y() - y0
            if dx * dx + dy * dy < 25 and self._projected is not None:
                px, py, visible = self._projected
                distance2 = (px - event.position().x()) ** 2 + (py - event.position().y()) ** 2
                distance2[~visible] = np.inf
                particle_index = int(np.argmin(distance2))
                if distance2[particle_index] < 100:
                    self.selected_particle = particle_index
                    if self.on_pick is not None:
                        self.on_pick(particle_index)
        self.drag = None

    def wheelEvent(self, event) -> None:
        factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self.zoom = float(np.clip(self.zoom * factor, 0.15, 12.0))
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, run_dir: str):
        super().__init__()
        self.setWindowTitle("Armor Penetration Lab — Qt Particle Viewer")
        self.dataset = DataSet(run_dir)
        self.current_frame = 0
        self.playing = False

        self.viewport = Viewport()
        self.viewport.data = self.dataset
        self.viewport.on_pick = self.show_particle
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(self.viewport)

        playback = QHBoxLayout()
        layout.addLayout(playback)
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_playback)
        playback.addWidget(self.play_button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, self.dataset.count - 1))
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.load_frame)
        playback.addWidget(self.slider)
        self.frame_label = QLabel()
        playback.addWidget(self.frame_label)

        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.field_combo = QComboBox()
        for key, (label, unit) in FIELD_INFO.items():
            display = f"{label} [{unit}]" if unit else label
            self.field_combo.addItem(display, key)
        self.field_combo.currentIndexChanged.connect(self.change_field)
        controls.addWidget(self.field_combo)

        self.slice_spin = QDoubleSpinBox()
        self.slice_spin.setRange(0.0, 1000.0)
        self.slice_spin.setDecimals(2)
        self.slice_spin.setSuffix(" mm")
        self.slice_spin.setSpecialValueText("Full volume")
        self.slice_spin.valueChanged.connect(self.change_slice)
        controls.addWidget(self.slice_spin)

        self.particle_label = QLabel("Click a particle to inspect its values.")
        self.particle_label.setWordWrap(True)
        controls.addWidget(self.particle_label, 1)

        for label, callback in (
            ("Isometric", lambda: self.set_camera(0.65, 0.45)),
            ("Front", lambda: self.set_camera(0.0, 0.0)),
            ("Top", lambda: self.set_camera(0.0, np.pi / 2.0)),
            ("Side", lambda: self.set_camera(np.pi / 2.0, 0.0)),
            ("Reset", self.viewport.reset_camera),
            ("Export PNG", self.export_png),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            controls.addWidget(button)

        self.load_frame(0)
        self.change_field(self.field_combo.currentIndex())

    def load_frame(self, index: int) -> None:
        self.current_frame = int(index)
        self.viewport.set_frame(self.dataset, self.current_frame)
        self.frame_label.setText(f"Frame {self.current_frame + 1}/{self.dataset.count}")

    def next_frame(self) -> None:
        self.load_frame((self.current_frame + 1) % self.dataset.count)

    def toggle_playback(self) -> None:
        if self.dataset.count < 2:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("▶ Play")
            return
        self.playing = not self.playing
        self.play_button.setText("⏸ Pause" if self.playing else "▶ Play")
        if self.playing:
            self.timer.start(60)
        else:
            self.timer.stop()

    def change_field(self, _index: int = 0) -> None:
        key = self.field_combo.currentData()
        if key not in FIELD_INFO:
            return
        self.viewport.field = key
        if key == "dmg":
            self.viewport.limits = (0.0, 1.2)
        else:
            values = [
                self.viewport._values(self.dataset.frame(index)[1])
                for index in sorted({0, self.dataset.count - 1})
            ]
            combined = np.concatenate(values)
            low = 290.0 if key == "T" else 0.0 if key in ("spd", "epsp", "svm") else float(np.percentile(combined, 2))
            high = float(np.percentile(combined, 98))
            if high <= low:
                high = low + 1.0
            self.viewport.limits = (low, high)
        self.viewport.update()

    def change_slice(self, value: float) -> None:
        self.viewport.slice_mm = float(value)
        self.viewport.update()

    def set_camera(self, yaw: float, pitch: float) -> None:
        self.viewport.yaw = yaw
        self.viewport.pitch = pitch
        self.viewport.zoom = 1.0
        self.viewport.pan[:] = 0.0
        self.viewport.update()

    def show_particle(self, particle_index: int) -> None:
        _, fields = self.dataset.frame(self.current_frame)
        dim = self.dataset.dim
        indices = [("X", 0, 1.0, "m"), ("Y", 1, 1.0, "m")]
        if dim == 3:
            indices.append(("Z", 2, 1.0, "m"))
        indices.extend([
            ("speed", FIELD_INDICES[dim]["spd"], 1.0, "m/s"),
            ("damage", FIELD_INDICES[dim]["dmg"], 1.0, "D"),
            ("pressure", FIELD_INDICES[dim]["P"], 1.0e-9, "GPa"),
            ("temperature", FIELD_INDICES[dim]["T"], 1.0, "K"),
            ("plastic strain", FIELD_INDICES[dim]["epsp"], 1.0, ""),
            ("von Mises", FIELD_INDICES[dim]["svm"], 1.0e-9, "GPa"),
        ])
        details = [f"{name}={float(fields[index, particle_index]) * factor:.4g} {unit}".rstrip()
                   for name, index, factor, unit in indices]
        self.particle_label.setText(f"Particle index {particle_index}: " + "; ".join(details))
        self.viewport.update()

    def export_png(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export current view", "", "PNG image (*.png)"
        )
        if not file_name:
            return
        if not file_name.lower().endswith(".png"):
            file_name += ".png"
        if not self.viewport.grab().save(file_name):
            QMessageBox.critical(self, "Export failed", f"Could not save {file_name}")

    def closeEvent(self, event) -> None:
        self.dataset.close()
        event.accept()


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python viewer_qt.py RUN_DIR")
        return 2

    app = QApplication(sys.argv)
    try:
        window = MainWindow(sys.argv[1])
        window.resize(1280, 850)
        window.show()
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(None, "Viewer error", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit tests for binary frame parsing and the lightweight renderers."""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import gui2


class FrameToolsTests(unittest.TestCase):
    def _write_run(self, dim: int, fields: np.ndarray, times=(0.0,)) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run_dir = Path(temporary.name)
        nfields, particle_count = fields.shape
        with (run_dir / "frames.bin").open("wb") as stream:
            for time_value in times:
                stream.write(struct.pack("<f", time_value))
                stream.write(np.asarray(fields, dtype="<f4").tobytes(order="C"))
        (run_dir / "meta.txt").write_text(
            f"VER=2\nDIM={dim}\nNFIELDS={nfields}\nN={particle_count}\nNP=2\n"
            "verdict=EMBEDDED\nvres=0\n",
            encoding="utf-8",
        )
        return run_dir

    @staticmethod
    def _fields_2d() -> np.ndarray:
        # Field-major order: X, Y, speed, damage, pressure, temperature, epsp, svm.
        return np.array(
            [
                [0.000, 0.001, 0.000, 0.001],
                [0.000, 0.000, 0.001, 0.001],
                [100.0, 110.0, 120.0, 130.0],
                [0.0, 0.2, 1.0, 1.1],
                [1.0e9, 2.0e9, 3.0e9, 4.0e9],
                [293.0, 300.0, 310.0, 320.0],
                [0.0, 0.1, 0.2, 0.3],
                [0.5e9, 1.0e9, 1.5e9, 2.0e9],
            ],
            dtype=np.float32,
        )

    def test_field_indices_match_2d_and_3d_layouts(self):
        self.assertEqual(gui2.field_idx(2)["dmg"], 3)
        self.assertEqual(gui2.field_idx(2)["svm"], 7)
        self.assertEqual(gui2.field_idx(3)["dmg"], 4)
        self.assertEqual(gui2.field_idx(3)["svm"], 8)

    def test_read_frames_decodes_field_major_records(self):
        fields = self._fields_2d()
        run_dir = self._write_run(2, fields, times=(0.0, 1.25e-6))
        frames = gui2.read_frames(run_dir, fields.shape[1], fields.shape[0])
        self.assertEqual(len(frames), 2)
        self.assertAlmostEqual(frames[1][0], 1.25e-6, places=12)
        np.testing.assert_allclose(frames[0][1], fields[0])
        np.testing.assert_allclose(frames[0][4], fields[3])

    def test_read_frames_rejects_truncated_and_empty_files(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run_dir = Path(temporary.name)
        path = run_dir / "frames.bin"
        path.write_bytes(b"\x00\x00\x00")
        with self.assertRaises(ValueError):
            gui2.read_frames(run_dir, n=4, nfields=8)
        path.write_bytes(b"")
        with self.assertRaises(ValueError):
            gui2.read_frames(run_dir, n=4, nfields=8)

    def test_scalar_limits_and_2d_ppm_render(self):
        fields = self._fields_2d()
        run_dir = self._write_run(2, fields)
        frames = gui2.read_frames(run_dir, fields.shape[1], fields.shape[0])
        self.assertEqual(gui2.field_limits(frames, 2, "dmg", 2), (0.0, 1.2))
        pressure_limits = gui2.field_limits(frames, 2, "P", 2)
        self.assertGreater(pressure_limits[1], pressure_limits[0])
        ppm = gui2.render_frame(
            frames[0], 2, gui2.compute_bounds(frames), "dmg", (0.0, 1.2), dim=2,
            wpx=64, hpx=48,
        )
        self.assertTrue(ppm.startswith(b"P6 64 48 255 "))
        self.assertEqual(len(ppm.split(b" ", 4)[4]), 64 * 48 * 3)

    def test_3d_renderer_handles_z_and_depth_slice(self):
        two_d = self._fields_2d()
        fields_3d = np.vstack((two_d[:2], np.array([[0.0, 0.001, -0.001, 0.002]], dtype=np.float32), two_d[2:]))
        run_dir = self._write_run(3, fields_3d)
        frames = gui2.read_frames(run_dir, fields_3d.shape[1], fields_3d.shape[0])
        ppm = gui2.render_3d_frame(
            frames[0], 2, gui2.compute_bounds(frames), "dmg", (0.0, 1.2),
            yaw=0.65, pitch=0.45, zwin=0.0015, wpx=64, hpx=48,
        )
        self.assertTrue(ppm.startswith(b"P6 64 48 255 "))
        self.assertEqual(len(ppm.split(b" ", 4)[4]), 64 * 48 * 3)


if __name__ == "__main__":
    unittest.main()

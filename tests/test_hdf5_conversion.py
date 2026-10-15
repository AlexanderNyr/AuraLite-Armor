"""Tests for the frames.bin to HDF5/XDMF converter."""

import struct
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

import convert_hdf5


class Hdf5ConversionTests(unittest.TestCase):
    def test_writes_time_major_hdf5_and_temporal_xdmf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            output_path = root / "results.h5"
            particle_count = 4
            field_count = 9
            fields = np.arange(field_count * particle_count, dtype=np.float32).reshape(
                field_count, particle_count
            )
            with (run_dir / "meta.txt").open("w", encoding="utf-8") as stream:
                stream.write(
                    "VER=2\nDIM=3\nNFIELDS=9\nN=4\nNP=2\n"
                    "verdict=EMBEDDED\nvres=0\n"
                )
            with (run_dir / "frames.bin").open("wb") as stream:
                for time_value in (0.0, 2.5e-6):
                    stream.write(struct.pack("<f", time_value))
                    stream.write(fields.astype("<f4").tobytes(order="C"))

            hdf_path, xdmf_path = convert_hdf5.convert(run_dir, output_path)
            self.assertTrue(hdf_path.is_file())
            self.assertTrue(xdmf_path.is_file())
            with h5py.File(hdf_path, "r") as h5:
                self.assertEqual(h5["X"].shape, (2, particle_count))
                self.assertEqual(h5["damage"].shape, (2, particle_count))
                self.assertAlmostEqual(float(h5["time"][1]), 2.5e-6, places=12)
                np.testing.assert_allclose(h5["speed"][0], fields[3])
                np.testing.assert_allclose(h5["damage"][1], fields[4])

            xml_root = ET.parse(xdmf_path).getroot()
            grids = xml_root.findall('.//Grid[@GridType="Uniform"]')
            self.assertEqual(len(grids), 2)
            self.assertAlmostEqual(float(grids[0].find("Time").get("Value")), 0.0, places=12)
            self.assertAlmostEqual(float(grids[1].find("Time").get("Value")), 2.5e-6, places=12)
            hyperslabs = xml_root.findall('.//DataItem[@ItemType="HyperSlab"]')
            self.assertEqual(len(hyperslabs), 2 * field_count)
            self.assertEqual(hyperslabs[1].get("Dimensions"), f"1 {particle_count}")
            selector = hyperslabs[1].find("DataItem")
            self.assertEqual(" ".join(selector.text.split()), "0 0 1 1 1 4")
            second_frame_selector = hyperslabs[field_count].find("DataItem")
            self.assertEqual(" ".join(second_frame_selector.text.split()), "1 0 1 1 1 4")

    def test_rejects_incomplete_frames_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "meta.txt").write_text(
                "VER=2\nDIM=2\nNFIELDS=8\nN=4\nNP=2\n", encoding="utf-8"
            )
            (run_dir / "frames.bin").write_bytes(b"truncated")
            with self.assertRaises(ValueError):
                convert_hdf5.convert(run_dir, root / "results.h5")


if __name__ == "__main__":
    unittest.main()

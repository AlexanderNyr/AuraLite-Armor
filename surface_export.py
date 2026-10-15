#!/usr/bin/env python3
"""Extract a smoothed VTK surface from one HDF5 result frame.

Usage:
    python surface_export.py RESULTS.h5 FRAME OUTPUT.vtp [--field damage]

Requires: numpy, h5py, and VTK. The output is a visualization isosurface, not
an exact physical boundary of the SPH material.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import h5py
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except ImportError as exc:
    raise SystemExit("Missing dependencies. Install them with: pip install vtk h5py numpy") from exc


FIELD_ALIASES = {
    "dmg": "damage",
    "spd": "speed",
    "P": "pressure",
    "T": "temperature",
    "plastic": "epsp",
    "mises": "svm",
}


def export_surface(input_path: Path, frame_index: int, output_path: Path,
                   field_name: str = "damage") -> Path:
    field_name = FIELD_ALIASES.get(field_name, field_name)
    with h5py.File(input_path, "r") as h5:
        if "X" not in h5 or "Y" not in h5:
            raise ValueError("HDF5 input is missing coordinate datasets X and Y")
        frame_count = int(h5["X"].shape[0])
        if not 0 <= frame_index < frame_count:
            raise IndexError(f"Frame index {frame_index} is outside [0, {frame_count - 1}]")
        if field_name not in h5:
            available = ", ".join(sorted(h5.keys()))
            raise KeyError(f"Field {field_name!r} not found. Available datasets: {available}")

        x = np.asarray(h5["X"][frame_index], dtype=np.float32)
        y = np.asarray(h5["Y"][frame_index], dtype=np.float32)
        z = np.asarray(h5["Z"][frame_index], dtype=np.float32) if "Z" in h5 else np.zeros_like(x)
        values = np.asarray(h5[field_name][frame_index], dtype=np.float32)

    if not (x.size and x.shape == y.shape == z.shape == values.shape):
        raise ValueError("Coordinate and scalar-field arrays are empty or have inconsistent shapes")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = vtk.vtkPoints()
    xyz = np.ascontiguousarray(np.column_stack((x, y, z)), dtype=np.float32)
    points.SetData(numpy_to_vtk(xyz, deep=True))

    particles = vtk.vtkPolyData()
    particles.SetPoints(points)
    scalars = numpy_to_vtk(values, deep=True)
    scalars.SetName(field_name)
    particles.GetPointData().SetScalars(scalars)

    extent = max(float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z)), 1.0e-6)
    splatter = vtk.vtkGaussianSplatter()
    splatter.SetInputData(particles)
    splatter.SetRadius(extent * 0.035)
    splatter.SetSampleDimensions(160, 160, 160)
    splatter.SetNormalWarping(0)
    splatter.Update()

    contour = vtk.vtkContourFilter()
    contour.SetInputConnection(splatter.GetOutputPort())
    contour.SetValue(0, 0.12)
    contour.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(contour.GetOutputPort())
    if writer.Write() != 1:
        raise OSError(f"Failed to write VTK surface: {output_path}")
    print(f"Wrote smoothed surface: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input HDF5 file")
    parser.add_argument("frame", type=int, help="zero-based frame index")
    parser.add_argument("output", type=Path, help="output .vtp file")
    parser.add_argument("--field", default="damage", help="scalar field to attach (default: damage)")
    args = parser.parse_args()
    export_surface(args.input, args.frame, args.output, args.field)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

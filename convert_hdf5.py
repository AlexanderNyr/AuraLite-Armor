#!/usr/bin/env python3
"""Convert an Armor Penetration Lab run to HDF5 and XDMF.

Usage:
    python convert_hdf5.py RUN_DIR OUTPUT.h5

The HDF5 file stores one time-major array per field. The accompanying XDMF
file describes a temporal collection for tools such as ParaView.
"""

from __future__ import annotations

import argparse
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise SystemExit("Missing dependency. Install it with: pip install h5py") from exc


FIELD_NAMES_2D = ["X", "Y", "speed", "damage", "pressure", "temperature", "epsp", "svm"]
FIELD_NAMES_3D = ["X", "Y", "Z", "speed", "damage", "pressure", "temperature", "epsp", "svm"]


def read_meta(run_dir: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with (run_dir / "meta.txt").open(encoding="utf-8") as stream:
        for line in stream:
            if "=" in line:
                key, value = line.rstrip().split("=", 1)
                meta[key] = value
    return meta


def _xdmf_data_item(parent: ET.Element, dataset: str, frame_index: int,
                    frame_count: int, particle_count: int, hdf_name: str) -> None:
    """Add a standards-based XDMF hyperslab selecting one frame row."""
    slab = ET.SubElement(
        parent,
        "DataItem",
        {"ItemType": "HyperSlab", "Dimensions": f"1 {particle_count}", "Type": "HyperSlab"},
    )
    selector = ET.SubElement(slab, "DataItem", {"Dimensions": "3 2", "Format": "XML"})
    selector.text = f"{frame_index} 0 1 1 1 {particle_count}"
    source = ET.SubElement(
        slab,
        "DataItem",
        {
            "Dimensions": f"{frame_count} {particle_count}",
            "NumberType": "Float",
            "Precision": "4",
            "Format": "HDF",
        },
    )
    source.text = f"{hdf_name}:/{dataset}"


def write_xdmf(hdf_path: Path, xdmf_path: Path, names: list[str],
               times: list[float], particle_count: int, dim: int) -> None:
    relative_hdf = os.path.relpath(hdf_path, xdmf_path.parent)
    root = ET.Element("Xdmf", {"Version": "3.0"})
    domain = ET.SubElement(root, "Domain")
    collection = ET.SubElement(
        domain,
        "Grid",
        {"Name": "SPH", "GridType": "Collection", "CollectionType": "Temporal"},
    )
    coordinate_names = ["X", "Y", "Z"] if dim == 3 else ["X", "Y"]
    geometry_type = "XYZ" if dim == 3 else "XY"
    coordinate_count = len(coordinate_names)

    for frame_index, time_value in enumerate(times):
        grid = ET.SubElement(
            collection,
            "Grid",
            {"Name": f"frame_{frame_index:05d}", "GridType": "Uniform"},
        )
        ET.SubElement(grid, "Time", {"Value": f"{time_value:.9g}"})
        ET.SubElement(
            grid,
            "Topology",
            {"TopologyType": "Polyvertex", "NumberOfElements": str(particle_count)},
        )
        geometry = ET.SubElement(grid, "Geometry", {"GeometryType": geometry_type})
        for name in coordinate_names:
            _xdmf_data_item(geometry, name, frame_index, len(times), particle_count, relative_hdf)

        for name in names[coordinate_count:]:
            attribute = ET.SubElement(
                grid,
                "Attribute",
                {"Name": name, "AttributeType": "Scalar", "Center": "Node"},
            )
            _xdmf_data_item(attribute, name, frame_index, len(times), particle_count, relative_hdf)

    ET.indent(root, space="  ")
    xdmf_path.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(root, encoding="utf-8")
        + b"\n"
    )


def convert(run_dir: Path, output_path: Path) -> tuple[Path, Path]:
    meta = read_meta(run_dir)
    dim = int(meta.get("DIM", "2"))
    if dim not in (2, 3):
        raise ValueError(f"Unsupported dimension in meta.txt: {dim}")
    particle_count = int(meta["N"])
    field_count = int(meta.get("NFIELDS", "8" if dim == 2 else "9"))
    names = FIELD_NAMES_2D if dim == 2 else FIELD_NAMES_3D
    if particle_count <= 0 or field_count != len(names):
        raise ValueError(
            f"Invalid run metadata: N={particle_count}, NFIELDS={field_count}, DIM={dim}"
        )

    frames_path = run_dir / "frames.bin"
    block_size = 4 + particle_count * field_count * 4
    file_size = frames_path.stat().st_size
    if file_size == 0 or file_size % block_size:
        raise ValueError(
            f"frames.bin has an incomplete or empty frame (size={file_size}, block={block_size})"
        )
    frame_count = file_size // block_size
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xdmf_path = Path(str(output_path) + ".xdmf")
    times: list[float] = []

    with h5py.File(output_path, "w") as h5:
        h5.attrs.update(meta)
        h5.attrs["dimension"] = dim
        h5.attrs["coordinate_unit"] = "m"
        h5.attrs["time_unit"] = "s"
        h5.attrs["pressure_unit"] = "Pa"
        h5.create_dataset("time", shape=(frame_count,), dtype="f4", compression="gzip")
        datasets = {
            name: h5.create_dataset(
                name,
                shape=(frame_count, particle_count),
                dtype="f4",
                chunks=(1, particle_count),
                compression="gzip",
            )
            for name in names
        }

        with frames_path.open("rb") as stream:
            for frame_index in range(frame_count):
                raw = stream.read(block_size)
                if len(raw) != block_size:
                    raise ValueError(f"Unexpected end of frames.bin at frame {frame_index}")
                time_value = struct.unpack_from("<f", raw, 0)[0]
                times.append(float(time_value))
                h5["time"][frame_index] = time_value
                fields = np.frombuffer(
                    raw, dtype="<f4", offset=4, count=particle_count * field_count
                ).reshape(field_count, particle_count)
                for field_index, name in enumerate(names):
                    datasets[name][frame_index] = fields[field_index]

    write_xdmf(output_path, xdmf_path, names, times, particle_count, dim)
    print(f"Wrote {output_path} ({frame_count} frames, {particle_count} particles)")
    print(f"Wrote {xdmf_path}")
    return output_path, xdmf_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="solver output directory")
    parser.add_argument("output", type=Path, help="output HDF5 file")
    args = parser.parse_args()
    convert(args.run_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

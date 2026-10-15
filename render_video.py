#!/usr/bin/env python3
"""Render frames.bin as an MP4 using the project's CPU particle renderer.

Usage:
    python render_video.py RUN_DIR OUTPUT.mp4 [FIELD] [FPS]

FIELD may be damage, speed, pressure, temperature, plastic, mises, or a raw
field key (dmg, spd, P, T, epsp, svm). Requires imageio and imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install them with: pip install imageio imageio-ffmpeg"
    ) from exc

from gui2 import (
    compute_bounds,
    field_limits,
    read_frames,
    read_meta,
    render_3d_frame,
    render_frame,
)


FIELD_ALIASES = {
    "damage": "dmg",
    "pressure": "P",
    "temperature": "T",
    "speed": "spd",
    "plastic": "epsp",
    "mises": "svm",
}
VALID_FIELDS = {"dmg", "spd", "P", "T", "epsp", "svm"}


def ppm_to_rgb(raw: bytes) -> np.ndarray:
    """Decode the binary P6 PPM images produced by gui2.py."""
    position = 0
    tokens: list[bytes] = []
    for _ in range(4):
        while position < len(raw) and raw[position] in b" \t\r\n":
            position += 1
        start = position
        while position < len(raw) and raw[position] not in b" \t\r\n":
            position += 1
        if position == start:
            raise ValueError("Malformed PPM header")
        tokens.append(raw[start:position])
        if position < len(raw):
            if raw[position:position + 2] == b"\r\n":
                position += 2
            else:
                position += 1
    if tokens[0] != b"P6" or tokens[3] != b"255":
        raise ValueError("Expected an 8-bit binary P6 PPM image")
    width, height = int(tokens[1]), int(tokens[2])
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=position)
    expected = width * height * 3
    if pixels.size != expected:
        raise ValueError(f"Malformed PPM payload: expected {expected} bytes, got {pixels.size}")
    return pixels.reshape(height, width, 3)


def render_video(run_dir: Path, output_path: Path, field: str = "dmg", fps: int = 25) -> Path:
    if fps <= 0:
        raise ValueError("FPS must be a positive integer")
    field = FIELD_ALIASES.get(field, field)
    if field not in VALID_FIELDS:
        raise ValueError(f"Unsupported field {field!r}. Choose one of: {', '.join(sorted(VALID_FIELDS))}")

    meta = read_meta(run_dir)
    dim = int(meta.get("DIM", "2"))
    particle_count = int(meta["N"])
    field_count = int(meta.get("NFIELDS", "8" if dim == 2 else "9"))
    projectile_count = int(meta["NP"])
    frames = read_frames(run_dir, particle_count, field_count)
    if not frames:
        raise ValueError("The run contains no frames")
    bounds = compute_bounds(frames)
    limits = field_limits(frames, projectile_count, field, dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            if dim == 3:
                ppm = render_3d_frame(
                    frame, projectile_count, bounds, field, limits,
                    yaw=0.65, pitch=0.45,
                )
            else:
                ppm = render_frame(
                    frame, projectile_count, bounds, field, limits, dim=2,
                )
            writer.append_data(ppm_to_rgb(ppm))

    print(f"Wrote video: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="solver output directory")
    parser.add_argument("output", type=Path, help="output MP4 path")
    parser.add_argument("field", nargs="?", default="dmg", help="field to render (default: dmg)")
    parser.add_argument("fps", nargs="?", type=int, default=25, help="video frame rate (default: 25)")
    args = parser.parse_args()
    render_video(args.run_dir, args.output, args.field, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

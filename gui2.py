# -*- coding: utf-8 -*-
"""
Armor Penetration Lab — Tkinter interface for the hybrid TL/UL SPH solver
sph_core2 (C++17).

Features include v2 result files, scalar-field color maps, hybrid TL/UL and
XSPH controls, VTK export, velocity sweeps, an automatic V50 search, run
cleanup, and a headless result-file self-test.

Requirements: Python 3.8+, NumPy, and Tkinter.
"""

import os
import re
import sys
import glob
import queue
import shutil
import struct
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IS_WIN = platform.system() == "Windows"
POPEN_FLAGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WIN else {}
SOLVERS = {2: ("sph_core2.exe" if IS_WIN else "sph_core2"),
           3: ("sph_core3.exe" if IS_WIN else "sph_core3")}

MATERIALS = ["steel", "tungsten", "rha", "hha", "aluminum"]
MAT_LABEL = {
    "steel": "Hardened steel (core)",
    "tungsten": "Tungsten heavy alloy",
    "rha": "Rolled homogeneous armor (RHA)",
    "hha": "High-hardness armor (HHA)",
    "aluminum": "Aluminum 5083",
}
NOSES = [("flat", "Flat"), ("ogive", "Ogive"), ("pointed", "Pointed")]
VERDICT_LABELS = {
    "PENETRATION": ("PENETRATION", "#e74c3c"),
    "RICOCHET": ("RICOCHET", "#2ecc71"),
    "EMBEDDED": ("EMBEDDED", "#f1c40f"),
}
FIELDS = [
    ("dmg",  "Damage D"),
    ("spd",  "Speed, m/s"),
    ("P",    "Pressure, GPa"),
    ("T",    "Temperature, K"),
    ("epsp", "Plastic strain"),
    ("svm",  "von Mises stress, GPa"),
]
FIELD_UNITS = {
    "dmg": "D",
    "spd": "m/s",
    "P": "GPa",
    "T": "K",
    "epsp": "dimensionless",
    "svm": "GPa",
}

def field_idx(dim):
    """Return field indices for 2D (without Z) or 3D (with Z) frames."""
    if dim == 3:
        return {"X": 0, "Y": 1, "Z": 2, "spd": 3, "dmg": 4, "P": 5,
                "T": 6, "epsp": 7, "svm": 8}
    return {"X": 0, "Y": 1, "spd": 2, "dmg": 3, "P": 4, "T": 5,
            "epsp": 6, "svm": 7}

# ----------------------------------------------------------------------------
# Read simulation output
# ----------------------------------------------------------------------------
def read_meta(run_dir):
    meta = {}
    with open(os.path.join(run_dir, "meta.txt"), encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v
    return meta


def read_frames(run_dir, n, nfields):
    """Read frames.bin v2 records: [time f32][field-major float32 arrays]."""
    path = os.path.join(run_dir, "frames.bin")
    frames = []
    block = 4 + nfields * n * 4
    with open(path, "rb") as f:
        data = f.read()
    if n <= 0 or nfields <= 0:
        raise ValueError("Invalid frame dimensions")
    if not data:
        raise ValueError("frames.bin contains no frames")
    if len(data) % block != 0:
        raise ValueError(
            f"Corrupt frames.bin: file size {len(data)} is not a multiple of frame size {block}")
    off = 0
    while off + block <= len(data):
        t = struct.unpack_from("<f", data, off)[0]
        arr = np.frombuffer(data, dtype="<f4", count=nfields * n, offset=off + 4)
        frames.append((t,) + tuple(arr.reshape(nfields, n)))
        off += block
    return frames


# ----------------------------------------------------------------------------
# Color maps and rasterization
# ----------------------------------------------------------------------------
_CMAP_STOPS = np.array([      # Jet-like palette, dark at the low end
    [0.15, 0.10, 0.35],
    [0.10, 0.45, 0.95],
    [0.10, 0.90, 0.75],
    [0.98, 0.85, 0.15],
    [0.90, 0.25, 0.10],
]) * 255
_CMAP_X = np.linspace(0, 1, len(_CMAP_STOPS))


def cmap(vals01):
    """Map an (N,) array in [0, 1] to an (N, 3) uint8 color array."""
    r = np.interp(vals01, _CMAP_X, _CMAP_STOPS[:, 0])
    g = np.interp(vals01, _CMAP_X, _CMAP_STOPS[:, 1])
    b = np.interp(vals01, _CMAP_X, _CMAP_STOPS[:, 2])
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def field_limits(frames, n_proj, fkey, dim=2):
    """Robust (2nd–98th percentile) limits for normalizing a field."""
    if not frames:
        return 0.0, 1.0
    idx = field_idx(dim)[fkey] + 1
    vals = np.concatenate([frames[0][idx], frames[-1][idx]])
    if fkey == "dmg":
        return 0.0, 1.2
    if fkey == "P" or fkey == "svm":
        vals = vals / 1e9
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if fkey in ("spd", "epsp", "svm"):
        lo = 0.0
    if fkey == "T":
        lo = 290.0
    if hi - lo < 1e-12:
        hi = lo + 1.0
    return lo, hi


def render_frame(frame, n_proj, bounds, field, lims, dim=2, zwin=None,
                 wpx=640, hpx=560):
    t = frame[0]
    FIX = field_idx(dim)
    X, Y = frame[1], frame[2]
    Z = frame[3] if dim == 3 else None
    idx = FIX[field] + 1
    val = frame[idx].astype(np.float64)
    if field == "P" or field == "svm":
        val = val / 1e9
    x0, x1, y0, y1 = bounds
    img = np.zeros((hpx, wpx, 3), dtype=np.uint8)
    img[:, :] = (14, 14, 22)

    ix = ((X - x0) / (x1 - x0) * (wpx - 42)).astype(np.int32)   # reserve the right side for the color scale
    iy = ((y1 - Y) / (y1 - y0) * (hpx - 2)).astype(np.int32)
    ok = (ix >= 0) & (ix < wpx - 41) & (iy >= 0) & (iy < hpx - 1)
    if dim == 3 and zwin is not None and zwin > 0:
        ok &= np.abs(Z) <= zwin      # depth slice

    v01 = np.clip((val - lims[0]) / (lims[1] - lims[0]), 0, 1)
    col = cmap(v01)
    col = np.where(ok[:, None], col, 0).astype(np.uint8)
    # brighten projectile particles slightly
    colp = np.clip(col[:n_proj].astype(int) + 25, 0, 255).astype(np.uint8)
    col[:n_proj] = colp
    # depth shading in 3D
    if dim == 3:
        zmax = max(float(np.abs(Z).max()), 1e-9)
        shade = (1.0 - 0.55 * np.abs(Z) / zmax).clip(0.4, 1.0)
        col = (col.astype(float) * shade[:, None]).astype(np.uint8)
        colp = np.clip(col[:n_proj].astype(int) + 40, 0, 255).astype(np.uint8)
        col[:n_proj] = colp
    # show failed particles in orange on the damage map
    if field == "dmg":
        brk = frame[FIX["dmg"] + 1] >= 1.0
        col[brk] = (255, 150, 20)

    for ox in (0, 1):
        for oy in (0, 1):
            sel = ok
            img[iy[sel] + oy, ix[sel] + ox] = col[sel]

    # color-scale strip on the right
    grad = cmap(np.linspace(1, 0, hpx))
    img[:, wpx - 32:wpx - 12] = grad[:, None, :]
    ppm = b"P6 %d %d 255 " % (wpx, hpx) + img.tobytes()
    return ppm


def render_3d_frame(frame, n_proj, bounds, field, lims, yaw, pitch,
                    zoom=1.0, panx=0.0, pany=0.0, zwin=None,
                    wpx=640, hpx=560):
    """Render an isometric 3D view without external rendering libraries.

    Uses an orthographic projection: mouse movement changes yaw/pitch, the
    wheel zooms, and the middle mouse button pans the view.
    """
    FIX = field_idx(3)
    X = frame[1].astype(np.float64)
    Y = frame[2].astype(np.float64)
    Z = frame[3].astype(np.float64)
    val = frame[FIX[field] + 1].astype(np.float64)
    if field in ("P", "svm"):
        val /= 1e9

    x0, x1, y0, y1 = bounds
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    # bounds computed from XY; include Z explicitly in the 3D scale.
    scale_world = max(x1 - x0, y1 - y0, float(Z.max() - Z.min()), 1e-9)
    scale = min((wpx - 54) / scale_world, (hpx - 8) / scale_world) * zoom

    xx, yy, zz = X - cx, Y - cy, Z - 0.5 * (Z.min() + Z.max())
    cyw, syw = np.cos(yaw), np.sin(yaw)
    x1r = cyw * xx - syw * zz
    z1r = syw * xx + cyw * zz
    cyp, syp = np.cos(pitch), np.sin(pitch)
    y2r = cyp * yy - syp * z1r
    depth = syp * yy + cyp * z1r

    ix = (wpx * 0.5 + panx + x1r * scale).astype(np.int32)
    iy = (hpx * 0.5 + pany - y2r * scale).astype(np.int32)
    ok = (ix >= 1) & (ix < wpx - 42) & (iy >= 1) & (iy < hpx - 1)
    if zwin is not None and zwin > 0:
        ok &= np.abs(Z) <= zwin

    v01 = np.clip((val - lims[0]) / max(lims[1] - lims[0], 1e-12), 0, 1)
    col = cmap(v01).astype(np.float64)
    dmin, dmax = float(depth.min()), float(depth.max())
    d01 = (depth - dmin) / max(dmax - dmin, 1e-12)
    # Far particles are dimmer, near particles brighter.
    col *= (0.52 + 0.48 * d01)[:, None]
    col = np.clip(col, 0, 255).astype(np.uint8)
    col[:n_proj] = np.clip(col[:n_proj].astype(int) + 28, 0, 255).astype(np.uint8)
    if field == "dmg":
        col[frame[FIX["dmg"] + 1] >= 1.0] = (255, 150, 20)

    img = np.full((hpx, wpx, 3), (14, 14, 22), dtype=np.uint8)
    # Painter's algorithm: draw far particles first.
    order = np.argsort(depth)
    for k in order:
        if not ok[k]:
            continue
        r = 2 if k < n_proj else 1
        x, y = ix[k], iy[k]
        img[max(0, y-r):min(hpx, y+r+1),
            max(0, x-r):min(wpx-41, x+r+1)] = col[k]

    grad = cmap(np.linspace(1, 0, hpx))
    img[:, wpx-32:wpx-12] = grad[:, None, :]
    return b"P6 %d %d 255 " % (wpx, hpx) + img.tobytes()

def compute_bounds(frames, pad=0.006):
    xs = np.concatenate([f[1] for f in (frames[0], frames[-1])])
    ys = np.concatenate([f[2] for f in (frames[0], frames[-1])])
    x0, x1 = xs.min() - pad, xs.max() + pad
    y0, y1 = ys.min() - pad, ys.max() + pad
    W, H = 598, 560                      # reserve 42 px on the right for the scale
    sx, sy = (x1 - x0) / W, (y1 - y0) / H
    s = max(sx, sy)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - s * W / 2, cx + s * W / 2, cy - s * H / 2, cy + s * H / 2)


# ----------------------------------------------------------------------------
# Run the solver locally or through the WSL bridge
# ----------------------------------------------------------------------------
def build_solver_command(arg_str, out_name, dim=2):
    exe_name = SOLVERS.get(dim, SOLVERS[2])
    args = arg_str.split() + [f"out={out_name}"]
    if IS_WIN:
        exe = os.path.join(HERE, exe_name)
        if os.path.exists(exe):
            return [exe] + args, HERE
        wsl_dir = subprocess.run(
            ["wsl", "wslpath", "-a", HERE.replace("\\", "/")],
            capture_output=True, text=True).stdout.strip()
        lin = SOLVERS.get(dim, SOLVERS[2]).replace(".exe", "")
        cmd = ["wsl", "bash", "-c",
               f"cd '{wsl_dir}' && ./{lin} {arg_str} out='{out_name}'"]
        return cmd, HERE
    else:
        exe = os.path.join(HERE, exe_name)
        return [exe] + args, HERE


def solver_exists(dim=2):
    """Check for a local solver binary, or a WSL solver on Windows."""
    exe = os.path.join(HERE, SOLVERS.get(dim, SOLVERS[2]))
    if os.path.isfile(exe) and (IS_WIN or os.access(exe, os.X_OK)):
        return True
    if IS_WIN:
        try:
            return subprocess.run(["wsl", "true"],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  **POPEN_FLAGS).returncode == 0
        except OSError:
            pass
    return False


def run_solver_blocking(args, out_name, dim=2):
    """Run the solver, combine stdout/stderr, and check its exit code."""
    cmd, cwd = build_solver_command(args, out_name, dim)
    pr = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                        **POPEN_FLAGS)
    output = (pr.stdout or "") + (pr.stderr or "")
    if pr.returncode != 0:
        raise RuntimeError(f"Solver exited with code {pr.returncode}: {output[-1000:]}")
    return output


# ----------------------------------------------------------------------------
#  GUI
# ----------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Armor Penetration Lab — Hybrid TL/UL SPH")
        root.configure(bg="#181820")

        self.q = queue.Queue()
        self.frames = []
        self.photos = {}
        self.n_proj = 0
        self.bounds = None
        self.field = "dmg"
        self.lims = {}
        self.playing = False
        self.play_delay = 80
        self.last_ppm = None
        self.cur = 0
        self.dim = 2                     # dimension of the current run
        self.view_yaw = 0.65
        self.view_pitch = 0.45
        self.view_zoom = 1.0
        self.view_panx = 0.0
        self.view_pany = 0.0
        self._drag = None
        self.run_id = len(glob.glob(os.path.join(HERE, "run_*")))
        self.sweep_results = []          # [(v, verdict, vres)]

        self._build_controls()
        self._build_canvas()
        root.after(100, self._poll_queue)
        if not (solver_exists(2) or solver_exists(3)):
            messagebox.showwarning(
                "Solver not found",
                "Could not find sph_core2/sph_core3 next to gui2.py.\n"
                "Build it with: g++ -O3 -march=native -fopenmp -std=c++17 "
                "sph_core2.cpp -o sph_core2")

    # ---------------- Parameter controls ----------------
    def _build_controls(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure(".", background="#181820", foreground="#eee")
        s.configure("TLabel", background="#181820", foreground="#eee")
        s.configure("TLabelframe", background="#181820", foreground="#eee")
        s.configure("TLabelframe.Label", background="#181820", foreground="#8fd")
        s.configure("TButton", padding=4)
        s.configure("TCheckbutton", background="#181820", foreground="#eee")

        left = ttk.Frame(self.root)
        left.pack(side="left", fill="y", padx=8, pady=8)

        def add_scale(parent, label, frm, to, init, step=1.0, fmt="{:.0f}"):
            box = ttk.Frame(parent); box.pack(fill="x", pady=2)
            var = tk.DoubleVar(value=init)
            lab = ttk.Label(box, text=f"{label}: {fmt.format(init)}")
            lab.pack(anchor="w")
            def upd(v, lab=lab, var=var, label=label, fmt=fmt, step=step):
                val = round(float(v) / step) * step
                var.set(val)
                lab.config(text=f"{label}: {fmt.format(val)}")
            sc = ttk.Scale(box, from_=frm, to=to, variable=var, command=upd)
            sc.pack(fill="x")
            return var

        g1 = ttk.Labelframe(left, text=" Projectile ")
        g1.pack(fill="x", pady=4)
        self.v_var = add_scale(g1, "Impact speed, m/s", 100, 2000, 800, 25)
        self.pl_var = add_scale(g1, "Length, mm", 8, 60, 20, 1)
        self.pd_var = add_scale(g1, "Diameter, mm", 4, 30, 8, 1)

        row = ttk.Frame(g1); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Material:").pack(side="left")
        self.pmat = tk.StringVar(value="steel")
        cb = ttk.Combobox(row, textvariable=self.pmat, state="readonly", width=22,
                          values=[f"{m} — {MAT_LABEL[m]}" for m in MATERIALS])
        cb.current(0); cb.pack(side="right")

        row = ttk.Frame(g1); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Nose:").pack(side="left")
        self.nose = tk.StringVar(value="flat")
        cb2 = ttk.Combobox(row, textvariable=self.nose, state="readonly", width=22,
                           values=[f"{k} — {label}" for k, label in NOSES])
        cb2.current(0); cb2.pack(side="right")

        g2 = ttk.Labelframe(left, text=" Armor ")
        g2.pack(fill="x", pady=4)
        self.t_var = add_scale(g2, "Thickness, mm", 4, 80, 15, 1)
        self.a_var = add_scale(g2, "Obliquity, °", 0, 80, 60, 5)
        self.w_var = add_scale(g2, "Width along z, mm (3D)", 10, 100, 40, 2)
        row = ttk.Frame(g2); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Material:").pack(side="left")
        self.amat = tk.StringVar(value="rha")
        cb3 = ttk.Combobox(row, textvariable=self.amat, state="readonly", width=22,
                           values=[f"{m} — {MAT_LABEL[m]}" for m in MATERIALS])
        cb3.current(2); cb3.pack(side="right")

        g3 = ttk.Labelframe(left, text=" Simulation ")
        g3.pack(fill="x", pady=4)
        self.dx_var = add_scale(g3, "Resolution dx, mm", 0.25, 1.0, 0.5, 0.05, "{:.2f}")
        self.te_var = add_scale(g3, "End time, μs", 20, 150, 60, 5)
        row = ttk.Frame(g3); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Color field:").pack(side="left")
        self.field_var = tk.StringVar(value="dmg")
        cbf = ttk.Combobox(row, textvariable=self.field_var, state="readonly",
                           width=24, values=[f"{k} — {label}" for k, label in FIELDS])
        cbf.current(0); cbf.pack(side="right")
        cbf.bind("<<ComboboxSelected>>", self._on_field_change)

        ge = ttk.Labelframe(left, text=" Solver ")
        ge.pack(fill="x", pady=4)
        row = ttk.Frame(ge); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Dimension:").pack(side="left")
        self.dim_var = tk.StringVar(value="2")
        cbd = ttk.Combobox(row, textvariable=self.dim_var, state="readonly",
                           width=26, values=["2 — plane strain",
                                             "3 — full 3D"])
        cbd.current(0); cbd.pack(side="right")
        cbd.bind("<<ComboboxSelected>>", lambda e: None)
        self.tl_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ge, text="Hybrid TL/UL (solid response)",
                        variable=self.tl_var).pack(anchor="w")
        self.xs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ge, text="XSPH smoothing", variable=self.xs_var).pack(anchor="w")
        self.as_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ge, text="Artificial stress (tensile stability)",
                        variable=self.as_var).pack(anchor="w")
        self.vtk_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ge, text="Export VTK (ParaView)",
                        variable=self.vtk_var).pack(anchor="w")

        self.run_btn = ttk.Button(left, text="▶  RUN", command=self.launch)
        self.run_btn.pack(fill="x", pady=(8, 2))
        rowb = ttk.Frame(left); rowb.pack(fill="x")
        self.sweep_btn = ttk.Button(rowb, text="Σ Speed sweep",
                                    command=self.launch_sweep)
        self.sweep_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.v50_btn = ttk.Button(rowb, text="◎ Auto V50",
                                  command=self.launch_v50)
        self.v50_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))
        rowb2 = ttk.Frame(left); rowb2.pack(fill="x", pady=2)
        self.plot_btn = ttk.Button(rowb2, text="📈 Residual speed vs impact speed",
                                   command=self.show_plot)
        self.plot_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        self.clean_btn = ttk.Button(rowb2, text="🗑 Clean runs",
                                    command=self.cleanup_runs)
        self.clean_btn.pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.pbar = ttk.Progressbar(left, maximum=100)
        self.pbar.pack(fill="x", pady=4)
        self.status = ttk.Label(left, text="Ready", wraplength=240)
        self.status.pack(fill="x")

        self.verdict_lab = tk.Label(left, text="—", font=("Arial", 16, "bold"),
                                    bg="#181820", fg="#888")
        self.verdict_lab.pack(fill="x", pady=6)

    # ---------------- Canvas and playback ----------------
    def _build_canvas(self):
        right = ttk.Frame(self.root)
        right.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)

        self.canvas = tk.Canvas(right, width=640, height=560,
                                bg="#101018", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._view_press)
        self.canvas.bind("<B1-Motion>", self._view_rotate)
        self.canvas.bind("<ButtonPress-2>", self._view_pan_press)
        self.canvas.bind("<B2-Motion>", self._view_pan)
        self.canvas.bind("<MouseWheel>", self._view_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._view_wheel(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._view_wheel(e, -1))
        self.img_id = None
        self.legend_ids = []

        bar = ttk.Frame(right); bar.pack(fill="x", pady=4)
        self.play_btn = ttk.Button(bar, text="⏵", width=3, command=self.toggle_play)
        self.play_btn.pack(side="left")
        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(bar, from_=0, to=1, variable=self.frame_var,
                                command=lambda v: self.show_frame(int(float(v))))
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.time_lab = ttk.Label(bar, text="t = 0 μs", width=14)
        self.time_lab.pack(side="right")

        bar2 = ttk.Frame(right); bar2.pack(fill="x")
        ttk.Label(bar2, text="Z slice, mm (0 = all, 3D):").pack(side="left")
        ttk.Button(bar2, text="Isometric", command=lambda: self._set_view_preset("iso")).pack(side="left", padx=2)
        ttk.Button(bar2, text="Front", command=lambda: self._set_view_preset("front")).pack(side="left", padx=2)
        ttk.Button(bar2, text="Top", command=lambda: self._set_view_preset("top")).pack(side="left", padx=2)
        ttk.Button(bar2, text="Side", command=lambda: self._set_view_preset("side")).pack(side="left", padx=2)
        self.zwin_var = tk.DoubleVar(value=0)
        ttk.Scale(bar2, from_=0, to=100, variable=self.zwin_var,
                  command=lambda v: self._on_zwin()).pack(side="left", fill="x",
                                                          expand=True, padx=6)
        ttk.Button(bar2, text="Reset 3D", command=self._reset_view).pack(side="right")
        ttk.Button(bar2, text="Save PPM", command=self._save_current_image).pack(side="right", padx=3)
        ttk.Label(bar2, text="Playback:").pack(side="right")
        self.speed_var = tk.StringVar(value="1x")
        speed = ttk.Combobox(bar2, textvariable=self.speed_var, state="readonly",
                             width=5, values=["0.25x", "0.5x", "1x", "2x", "4x"])
        speed.pack(side="right", padx=3)
        speed.bind("<<ComboboxSelected>>", self._on_speed)

    def _on_speed(self, _event=None):
        try:
            factor = float(self.speed_var.get().rstrip("x"))
            self.play_delay = max(10, int(80.0 / factor))
        except (ValueError, AttributeError):
            self.play_delay = 80

    def _set_view_preset(self, name):
        if self.dim != 3:
            return
        presets = {
            "iso": (0.65, 0.45),
            "front": (0.0, 0.0),
            "top": (0.0, 1.570796),
            "side": (1.570796, 0.0),
        }
        self.view_yaw, self.view_pitch = presets[name]
        self.view_zoom = 1.0
        self.view_panx = self.view_pany = 0.0
        self.photos = {}
        if self.frames:
            self.show_frame(self.cur)

    def _save_current_image(self):
        if self.last_ppm is None:
            messagebox.showinfo("Export", "Load a simulation frame first")
            return
        # PPM is lossless and does not require Pillow. It opens in ParaView,
        # GIMP, ImageMagick and most scientific image viewers.
        path = os.path.join(HERE, f"frame_{self.cur:04d}.ppm")
        try:
            with open(path, "wb") as f:
                f.write(self.last_ppm)
            self.status.config(text=f"Frame saved: {os.path.basename(path)}")
        except OSError as e:
            messagebox.showerror("Export failed", str(e))

    def _reset_view(self):
        self.view_yaw, self.view_pitch = 0.65, 0.45
        self.view_zoom = 1.0
        self.view_panx = self.view_pany = 0.0
        self.photos = {}
        if self.frames:
            self.show_frame(self.cur)

    def _view_press(self, event):
        if self.dim == 3:
            self._drag = (event.x, event.y, self.view_yaw, self.view_pitch)

    def _view_rotate(self, event):
        if self.dim != 3 or self._drag is None:
            return
        x0, y0, yaw0, pitch0 = self._drag
        self.view_yaw = yaw0 + (event.x - x0) * 0.012
        self.view_pitch = max(-1.45, min(1.45, pitch0 + (event.y - y0) * 0.012))
        self.photos = {}
        self.show_frame(self.cur)

    def _view_pan_press(self, event):
        if self.dim == 3:
            self._drag = (event.x, event.y, self.view_panx, self.view_pany)

    def _view_pan(self, event):
        if self.dim != 3 or self._drag is None:
            return
        x0, y0, px0, py0 = self._drag
        self.view_panx = px0 + event.x - x0
        self.view_pany = py0 + event.y - y0
        self.photos = {}
        self.show_frame(self.cur)

    def _view_wheel(self, event, direction=None):
        if self.dim != 3:
            return
        if direction is None:
            direction = 1 if event.delta > 0 else -1
        self.view_zoom = max(0.15, min(12.0, self.view_zoom * (1.12 if direction > 0 else 1/1.12)))
        self.photos = {}
        self.show_frame(self.cur)

    def _on_zwin(self):
        self.photos = {}
        if self.frames:
            self.show_frame(self.cur)

    def _zwin_m(self):
        w = self.zwin_var.get()
        return w * 1e-3 if w > 0 else None

    # ---------------- Run management ----------------
    def _on_field_change(self, _evt=None):
        self.field = self.field_var.get().split(" ")[0]
        self.photos = {}
        if self.frames:
            self.lims[self.field] = field_limits(self.frames, self.n_proj,
                                                 self.field, self.dim)
            self.show_frame(self.cur)

    def _ui_dim(self):
        return 3 if self.dim_var.get().startswith("3") else 2

    def _args_from_ui(self):
        pm = self.pmat.get().split(" ")[0]
        am = self.amat.get().split(" ")[0]
        ns = self.nose.get().split(" ")[0]
        s = (f"v={self.v_var.get():.0f} angle={self.a_var.get():.0f} "
             f"t_plate={self.t_var.get():.0f} proj_l={self.pl_var.get():.0f} "
             f"proj_d={self.pd_var.get():.0f} dx={self.dx_var.get():.2f} "
             f"t_end={self.te_var.get():.0f} proj_mat={pm} arm_mat={am} "
             f"nose={ns} tl={1 if self.tl_var.get() else 0} "
             f"xs={0.3 if self.xs_var.get() else 0} "
             f"as={0.3 if self.as_var.get() else 0} "
             f"vtk={1 if self.vtk_var.get() else 0}")
        if self._ui_dim() == 3:
            s += f" w_plate={self.w_var.get():.0f}"
        return s

    def launch(self):
        if not self._solver_ok():
            return
        self.run_btn.config(state="disabled")
        self.sweep_btn.config(state="disabled")
        self.v50_btn.config(state="disabled")
        self.verdict_lab.config(text="Running...", fg="#888")
        self.run_id += 1
        out = f"run_{self.run_id}"
        args = self._args_from_ui()
        dim = self._ui_dim()
        threading.Thread(target=self._worker, args=(args, out, dim),
                         daemon=True).start()

    def _solver_ok(self):
        d = self._ui_dim()
        if not solver_exists(d):
            messagebox.showerror(
                "Solver not found",
                f"Build {SOLVERS[d]} "
                f"(g++ -O3 -march=native -fopenmp -std=c++17"
                f"{' -DDIM=3' if d == 3 else ''} sph_core2.cpp -o {SOLVERS[d]})")
            return False
        return True

    def _worker(self, args, out, dim):
        try:
            cmd, cwd = build_solver_command(args, out, dim)
            pr = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  **POPEN_FLAGS)
            for line in pr.stdout:
                m = re.match(r"PROGRESS (\d+) (\d+)", line)
                if m:
                    self.q.put(("progress",
                                100.0 * int(m.group(1)) / int(m.group(2))))
                if line.startswith("RESULT"):
                    self.q.put(("status", line.strip()))
            pr.wait()
            if pr.returncode != 0:
                self.q.put(("error", f"Solver exited with code {pr.returncode}"))
                return
            self.q.put(("done", os.path.join(HERE, out)))
        except Exception as e:
            self.q.put(("error", str(e)))

    # ---------------- Velocity sweep ----------------
    def launch_sweep(self):
        top = tk.Toplevel(self.root)
        top.title("Velocity sweep: start / end / step")
        top.configure(bg="#181820")
        ents = []
        for lbl, dv in (("Start, m/s", 400), ("End, m/s", 1400), ("Step, m/s", 200)):
            ttk.Label(top, text=lbl).pack()
            e = ttk.Entry(top); e.insert(0, str(dv)); e.pack(padx=10)
            ents.append(e)

        def go():
            try:
                v0, v1, dv = (float(e.get()) for e in ents)
            except ValueError:
                messagebox.showerror("Velocity sweep", "All values must be numeric")
                return
            if dv <= 0 or v1 < v0:
                messagebox.showerror("Velocity sweep", "Step must be > 0 and end must be >= start")
                return
            base = self._args_from_ui()
            dim = self._ui_dim()
            top.destroy()
            self._block_buttons(True)
            threading.Thread(target=self._sweep_worker, args=(base, v0, v1, dv, dim),
                             daemon=True).start()
        ttk.Button(top, text="Run sweep", command=go).pack(pady=8)

    def _sweep_worker(self, base, v0, v1, dv, dim):
        try:
            results = []
            v = v0
            k = 0
            while v <= v1 + 1e-9:
                k += 1
                args = re.sub(r"v=\d+", f"v={v:.0f}", base) + " frames=2"
                self.q.put(("status", f"Sweep: {v:.0f} m/s ..."))
                outp = run_solver_blocking(args, f"sweep_{k}", dim)
                m = re.search(r"verdict=(\w+) vres=([0-9.+-]+)", outp)
                if m:
                    results.append((v, m.group(1), float(m.group(2))))
                v += dv
            self.sweep_results.extend(results)
            lines = ["  V (m/s) | verdict     | V_res (m/s)",
                     "  -------+------------+----------"]
            for v, verd, vr in results:
                verdict_label = VERDICT_LABELS.get(verd, (verd, ""))[0]
                lines.append(f"  {v:6.0f} | {verdict_label:<10} | {vr:6.0f}")
            self.q.put(("sweep_done", "\n".join(lines)))
        except Exception as e:
            self.q.put(("error", str(e)))

    # ---------------- Automatic V50 bisection ----------------
    def launch_v50(self):
        top = tk.Toplevel(self.root)
        top.title("Automatic V50 search (bisection)")
        top.configure(bg="#181820")
        ents = []
        for lbl, dv in (("Lower speed, m/s", 300), ("Upper speed, m/s", 1200),
                        ("Tolerance, m/s", 25)):
            ttk.Label(top, text=lbl).pack()
            e = ttk.Entry(top); e.insert(0, str(dv)); e.pack(padx=10)
            ents.append(e)
        ttk.Label(top, text="V50: estimated boundary between penetration and no penetration",
                  wraplength=220).pack()

        def go():
            try:
                vlo, vhi, tol = (float(e.get()) for e in ents)
            except ValueError:
                messagebox.showerror("V50", "All values must be numeric")
                return
            if vlo < 0 or vhi <= vlo or tol <= 0:
                messagebox.showerror("V50", "Upper speed must exceed lower speed and tolerance must be > 0")
                return
            base = self._args_from_ui()
            dim = self._ui_dim()
            top.destroy()
            self._block_buttons(True)
            threading.Thread(target=self._v50_worker, args=(base, vlo, vhi, tol, dim),
                             daemon=True).start()
        ttk.Button(top, text="Search", command=go).pack(pady=8)

    def _v50_probe(self, base, v, tag, dim):
        args = re.sub(r"v=\d+", f"v={v:.0f}", base) + " frames=1"
        self.q.put(("status", f"V50: testing {v:.0f} m/s ..."))
        outp = run_solver_blocking(args, f"v50_{tag}_{v:.0f}", dim)
        m = re.search(r"verdict=(\w+) vres=([0-9.+-]+)", outp)
        if not m:
            return None, None
        return m.group(1), float(m.group(2))

    def _v50_worker(self, base, vlo, vhi, tol, dim):
        try:
            log = []
            verd_lo, _ = self._v50_probe(base, vlo, "lo", dim)
            if verd_lo == "PENETRATION":
                self.q.put(("v50_done", f"{vlo:.0f} m/s already penetrates; lower the lower bound."))
                return
            verd_hi, _ = self._v50_probe(base, vhi, "hi", dim)
            if verd_hi is not None and verd_hi != "PENETRATION":
                self.q.put(("v50_done", f"{vhi:.0f} m/s does not penetrate; raise the upper bound."))
                return
            it = 0
            while vhi - vlo > tol and it < 12:
                it += 1
                vm = 0.5 * (vlo + vhi)
                verd, vres = self._v50_probe(base, vm, str(it), dim)
                log.append(f"  {it}. {vm:6.0f} m/s -> {VERDICT_LABELS.get(verd, (verd,))[0]}")
                self.sweep_results.append((vm, verd, vres if vres else 0.0))
                if verd == "PENETRATION":
                    vhi = vm
                else:
                    vlo = vm
            v50 = 0.5 * (vlo + vhi)
            log.append(f"\n  V50 ≈ {v50:.0f} m/s  (bracket [{vlo:.0f}, {vhi:.0f}])")
            self.q.put(("v50_done", "\n".join(log)))
        except Exception as e:
            self.q.put(("error", str(e)))

    # ---------------- Residual-speed plot ----------------
    def show_plot(self):
        if not self.sweep_results:
            messagebox.showinfo("Plot", "Run a velocity sweep or V50 search first.")
            return
        top = tk.Toplevel(self.root)
        top.title("Residual speed vs impact speed")
        cv = tk.Canvas(top, width=560, height=420, bg="#101018",
                       highlightthickness=0)
        cv.pack()
        data = sorted(self.sweep_results)
        vs = [d[0] for d in data]
        vr = [d[2] for d in data]
        x0, x1 = min(vs) * 0.95, max(vs) * 1.05
        y0 = 0.0
        y1 = max(max(vr), max(vs)) * 1.05
        W, H, P0 = 560, 420, 50

        def px(v): return P0 + (v - x0) / (x1 - x0) * (W - P0 - 20)
        def py(u): return H - P0 - (u - y0) / (y1 - y0) * (H - 2 * P0)

        cv.create_line(P0, H - P0, W - 20, H - P0, fill="#888")  # X
        cv.create_line(P0, H - P0, P0, 20, fill="#888")          # Y
        cv.create_text(W // 2, H - 12, text="Impact speed, m/s", fill="#ccc")
        cv.create_text(14, H // 2, text="Residual speed, m/s", fill="#ccc", angle=90)
        for fr in (0.25, 0.5, 0.75, 1.0):
            vx_ = x0 + fr * (x1 - x0)
            cv.create_text(px(vx_), H - P0 + 14, text=f"{vx_:.0f}", fill="#777")
            uy = y0 + fr * (y1 - y0)
            cv.create_text(26, py(uy), text=f"{uy:.0f}", fill="#777")
        cv.create_line(px(x0), py(x0), px(y1), py(min(y1, x1)),  # residual speed = impact speed
                       fill="#444", dash=(4, 4))
        for v, verd, u in data:
            color = VERDICT_LABELS.get(verd, ("", "#aaa"))[1]
            cv.create_oval(px(v) - 4, py(u) - 4, px(v) + 4, py(u) + 4,
                           fill=color, outline="")
        if len(data) > 1:
            pts = []
            for v, _, u in data:
                pts += [px(v), py(u)]
            cv.create_line(*pts, fill="#888")

    # ---------------- Run cleanup ----------------
    def cleanup_runs(self):
        pats = ["run_*", "sweep_*", "v50_*", "test_run*", "v2_*", "cmp_*"]
        dirs = []
        for p in pats:
            dirs += [d for d in glob.glob(os.path.join(HERE, p))
                     if os.path.isdir(d)]
        if not dirs:
            self.status.config(text="No run folders to clean")
            return
        if messagebox.askyesno("Clean runs",
                               f"Delete {len(dirs)} run folders?"):
            n = 0
            for d in dirs:
                try:
                    shutil.rmtree(d); n += 1
                except OSError:
                    pass
            self.status.config(text=f"Deleted run folders: {n}")

    # ---------------- Event queue ----------------
    def _block_buttons(self, blocked):
        st = "disabled" if blocked else "normal"
        self.run_btn.config(state=st)
        self.sweep_btn.config(state=st)
        self.v50_btn.config(state=st)

    def _poll_queue(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "progress":
                    self.pbar["value"] = val
                elif kind == "status":
                    self.status.config(text=val)
                elif kind == "error":
                    self.status.config(text="Error: " + val)
                    self._block_buttons(False)
                elif kind == "sweep_done":
                    self._block_buttons(False)
                    self.status.config(text="Velocity sweep complete")
                    messagebox.showinfo("Velocity sweep results", val)
                elif kind == "v50_done":
                    self._block_buttons(False)
                    self.status.config(text="V50 search complete")
                    messagebox.showinfo("Estimated ballistic limit", val)
                elif kind == "done":
                    self._load_run(val)
                    self._block_buttons(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ---------------- Load and display results ----------------
    def _load_run(self, run_dir):
        meta = read_meta(run_dir)
        ver = meta.get("VER", "1")
        if ver != "2":
            messagebox.showerror("Unsupported format",
                                 "This run was generated by the v1 solver; "
                                 "open it with gui.py instead.")
            return
        n = int(meta["N"])
        self.dim = int(meta.get("DIM", "2"))
        self.view_yaw, self.view_pitch = 0.65, 0.45
        self.view_zoom = 1.0
        self.view_panx = self.view_pany = 0.0
        nfields = int(meta.get("NFIELDS", "8" if self.dim == 2 else "9"))
        self.n_proj = int(meta["NP"])
        self.frames = read_frames(run_dir, n, nfields)
        self.bounds = compute_bounds(self.frames)
        self.photos = {}
        self.lims = {}
        self.slider.config(to=len(self.frames) - 1)
        verd = meta.get("verdict", "?")
        label, color = VERDICT_LABELS.get(verd, (verd, "#888"))
        vres = float(meta.get("vres", 0))
        ang = float(meta.get("angle_out", 0))
        tl = float(meta.get("tl", 0))
        self.verdict_lab.config(text=f"{label}  (V_res={vres:.0f} m/s)", fg=color)
        self.status.config(
            text=f"[{self.dim}D] exit angle {ang:+.0f}°, projectile intact "
                 f"{float(meta.get('intact', 0)) * 100:.0f}%, "
                 f"TL particles remaining: {tl * 100:.0f}%")
        self.lims[self.field] = field_limits(self.frames, self.n_proj,
                                             self.field, self.dim)
        self.show_frame(0)
        self.playing = True
        self.play_btn.config(text="⏸")
        self._animate()

    def show_frame(self, k):
        if not self.frames:
            return
        k = max(0, min(k, len(self.frames) - 1))
        self.cur = k
        key = (k, self.field, self._zwin_m(), self.dim,
               round(self.view_yaw, 4), round(self.view_pitch, 4),
               round(self.view_zoom, 4), round(self.view_panx, 1), round(self.view_pany, 1))
        if key not in self.photos:
            lim = self.lims.setdefault(
                self.field,
                field_limits(self.frames, self.n_proj, self.field, self.dim))
            if self.dim == 3:
                ppm = render_3d_frame(
                    self.frames[k], self.n_proj, self.bounds, self.field, lim,
                    self.view_yaw, self.view_pitch, self.view_zoom,
                    self.view_panx, self.view_pany, self._zwin_m())
            else:
                ppm = render_frame(self.frames[k], self.n_proj, self.bounds,
                                   self.field, lim, self.dim, self._zwin_m())
            self.last_ppm = ppm
            self.photos[key] = tk.PhotoImage(data=ppm)
        if self.img_id is None:
            self.img_id = self.canvas.create_image(0, 0, anchor="nw",
                                                   image=self.photos[key])
        else:
            self.canvas.itemconfig(self.img_id, image=self.photos[key])
        # color-scale legend
        for i in self.legend_ids:
            self.canvas.delete(i)
        self.legend_ids = []
        lo, hi = self.lims[self.field]
        unit = FIELD_UNITS.get(self.field, "")
        title = dict(FIELDS).get(self.field, self.field).split(",")[0]
        self.legend_ids.append(self.canvas.create_text(
            616, 14, text=f"{title} [{unit}]", fill="#ddd", font=("Arial", 8)))
        self.legend_ids.append(self.canvas.create_text(
            622, 40, text=f"{hi:.2g}", fill="#ddd", font=("Arial", 8)))
        self.legend_ids.append(self.canvas.create_text(
            622, 540, text=f"{lo:.2g}", fill="#ddd", font=("Arial", 8)))
        self.frame_var.set(k)
        self.time_lab.config(text=f"t = {self.frames[k][0] * 1e6:.1f} μs")

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="⏸" if self.playing else "⏵")
        if self.playing:
            self._animate()

    def _animate(self):
        if not self.playing or not self.frames:
            return
        nxt = self.cur + 1
        if nxt >= len(self.frames):
            nxt = 0
        self.show_frame(nxt)
        self.root.after(self.play_delay, self._animate)


# ----------------------------------------------------------------------------
# Headless self-test
# ----------------------------------------------------------------------------
def selftest(run_dir="run_1"):
    if not os.path.isdir(run_dir):
        print(f"Run directory not found: {run_dir}")
        return 1
    meta = read_meta(run_dir)
    dim = int(meta.get("DIM", "2"))
    n = int(meta["N"])
    nfields = int(meta.get("NFIELDS", "8" if dim == 2 else "9"))
    npr = int(meta["NP"])
    frames = read_frames(run_dir, n, nfields)
    print(f"meta: verdict={meta['verdict']} vres={meta['vres']} "
          f"N={n} NP={npr} DIM={dim} NFIELDS={nfields}")
    print(f"Frames: {len(frames)}, time: 0..{frames[-1][0] * 1e6:.1f} μs")
    for fkey, field in FIELDS:
        lo, hi = field_limits(frames, npr, fkey, dim)
        print(f"  field {fkey:5s}: [{lo:.3g} .. {hi:.3g}]")
    bounds = compute_bounds(frames)
    z_windows = [None] if dim == 2 else [None, 3e-3]
    for z_window in z_windows:
        limits = field_limits(frames, npr, "dmg", dim)
        if dim == 3:
            ppm = render_3d_frame(
                frames[-1], npr, bounds, "dmg", limits,
                yaw=0.65, pitch=0.45, zwin=z_window,
            )
        else:
            ppm = render_frame(
                frames[-1], npr, bounds, "dmg", limits, dim=2, zwin=z_window,
            )
        assert ppm.startswith(b"P6 "), "Rasterization failed"
    print(f"Frame render: OK ({len(ppm) / 1e6:.1f} MB PPM)")
    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        rd = sys.argv[2] if len(sys.argv) > 2 else "run_1"
        sys.exit(selftest(rd))
    root = tk.Tk()
    App(root)
    root.mainloop()

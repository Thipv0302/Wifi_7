"""
plotting/style.py -- He mau va cau hinh chung cho tat ca cac hinh.

Bang mau phan loai da qua kiem chung: nam trong dai do sang L 0.43-0.77,
chroma >= 0.1, khoang cach nho nhat giua cac cap ke nhau voi nguoi mu mau
dE = 9.1 (protan). Mau duoc gan CO DINH theo thuc the (AC / cau hinh), khong
xoay vong theo thu tu ve.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIG_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Mau co dinh cho tung cau hinh (Fig. 5, Fig. 6)
CONFIG_COLOR = {
    "Default EDCA": MUTED,
    "Opt. EDCA": SERIES[1],
    "Opt. MLO EDCA": SERIES[0],
    "EDCA": SERIES[1],
    "MLO EDCA": SERIES[0],
    "Target eps": SERIES[2],
}

# Mau co dinh cho tung AC (Fig. 3)
AC_COLOR = {f"AC{i+1}": SERIES[i] for i in range(5)}

MARKERS = ["o", "s", "^", "D", "v"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "font.family": "DejaVu Sans",
        "figure.dpi": 130,
    })


# ---------------------------------------------------------------------------
# He mau / kieu trinh bay "giong paper" (Fig. 3 - Fig. 6)
# ---------------------------------------------------------------------------
# Paper ve bang MATLAB: nen trang, khung hop day du, luoi mo, chu serif.
# Cac ma mau duoi day lay tu chinh cac hinh trong ban PDF de doi chieu truc
# tiep duoc tung duong / tung cot.

PAPER = {
    "ac1": "#E3A857",        # Fig. 3 -- AC1 (Fixed), cam nhat
    "ac2": "#4EA6DC",        # Fig. 3 -- AC2 (Varied), xanh lam
    "edca": "#0000FF",       # Fig. 4/6 -- Single-Link EDCA
    "mlo": "#FF0000",        # Fig. 4/6 -- MLO EDCA
    "bar_edca": "#5BA3D5",   # Fig. 5 -- Opt. EDCA
    "bar_mlo": "#F4A7B9",    # Fig. 5 -- Opt. MLO EDCA
    "bar_def": "#EFB765",    # Fig. 5 -- Default EDCA
    "bar_eps": "#6A3D9A",    # Fig. 5 -- Target eps_i
    "target": "#000000",     # Fig. 6 -- Target eps (net cham)
}


def paper_style() -> None:
    """Kieu trinh bay bam sat cac hinh trong paper (MATLAB, nen trang)."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#000000",
        "axes.titlecolor": "#000000",
        "axes.titlesize": 11,
        "axes.labelsize": 12,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.color": "#c8c8c8",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "text.color": "#000000",
        "legend.frameon": True,
        "legend.edgecolor": "#000000",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.0,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "figure.dpi": 130,
    })


# ---------------------------------------------------------------------------
# He mau cho cac hinh DO DOI CHIEU (benchmark GA vs GNN)
# ---------------------------------------------------------------------------
# Bang Okabe-Ito: an toan voi moi dang mu mau va con phan biet duoc khi in den
# trang -- dieu kien bat buoc cho hinh dua vao paper. Moi phuong phap duoc gan
# CO DINH mot bo (mau, net, marker) de doi chieu duoc giua cac hinh.

BENCH = {
    "ga":            {"c": "#000000", "ls": "-",  "m": "o"},
    "ga_gnn":        {"c": "#0072B2", "ls": "--", "m": "s"},
    "ga_gnn_policy": {"c": "#D55E00", "ls": "-",  "m": "^"},
    "policy":        {"c": "#009E73", "ls": ":",  "m": "*"},
}

# Hau to them vao ten file hinh (main.py dat "_quick" khi chay che do nhanh de
# khong de len hinh cua lan chay day du).
NAME_SUFFIX = ""


def save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}{NAME_SUFFIX}.png")
    fig.savefig(path, bbox_inches="tight", dpi=190)
    plt.close(fig)
    return path


def annotate(ax, text: str, loc: str = "upper left") -> None:
    xy = {"upper left": (0.02, 0.97), "upper right": (0.98, 0.97),
          "lower left": (0.02, 0.04), "lower right": (0.98, 0.04)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8, color=INK_2)

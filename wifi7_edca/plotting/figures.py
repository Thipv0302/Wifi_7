"""
plotting/figures.py -- MODULE VE KET QUA.

Moi ham `plot_figX(data)` nhan dict do `training/experiments.py` sinh ra va
xuat file PNG vao thu muc `figures/`. Module nay KHONG tinh toan lai gi.

Fig. 2 dung kieu trinh bay rieng (so do minh hoa cua mo hinh vung AIFS).
Fig. 3 - Fig. 6 duoc ve theo kieu cua paper (`plotting.style.paper_style`):
nen trang, khung hop, chu serif, cung bo cuc / cung thang truc / cung he mau
voi cac hinh trong ban PDF, de co the dat canh nhau ma doi chieu.
"""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting.style import (AC_COLOR, CONFIG_COLOR, GRID, INK_2, MARKERS,
                            MUTED, PAPER, SERIES, annotate, apply_style,
                            paper_style, save)

apply_style()


# ===========================================================================
# Fig. 2 -- mo hinh vung AIFS
# ===========================================================================


def plot_fig2(d: Dict) -> str:
    apply_style()
    n = d["n_slots"]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))

    rows = [("$\\varphi_j$", d["phi"]),
            ("$Z_j$", [d["z_count"][z] for z in d["zone_of_slot"]]),
            ("Slot", list(range(1, n + 1))),
            ("Zone", [z + 1 for z in d["zone_of_slot"]])]

    for r, (label, values) in enumerate(rows):
        y = len(rows) - r - 1
        for i, v in enumerate(values):
            color = (SERIES[d["zone_of_slot"][i] % len(SERIES)]
                     if label == "Zone" else "white")
            ax.add_patch(plt.Rectangle((i, y), 1, 0.9, facecolor=color,
                                       edgecolor=INK_2, linewidth=0.8,
                                       alpha=0.35 if label == "Zone" else 1.0))
            ax.text(i + 0.5, y + 0.45, str(v), ha="center", va="center",
                    fontsize=9)
        ax.text(-0.25, y + 0.45, label, ha="right", va="center", fontsize=9.5,
                color=INK_2)

    # danh dau slot ma tung AC bat dau duoc tranh chap (gop cac AC cung AIFS)
    by_offset: Dict[int, list] = {}
    for k, h in enumerate(d["h"]):
        by_offset.setdefault(int(h), []).append(k + 1)
    for h, ks in by_offset.items():
        label = ("AIFS$_{" + ",".join(str(k) for k in ks) + "}$"
                 + f" (AIFSN={d['aifsn'][ks[0] - 1]})")
        ax.annotate(label, xy=(h + 0.5, len(rows)),
                    xytext=(h + 0.5, len(rows) + 0.45),
                    ha="center", fontsize=8, color=INK_2,
                    arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.9))

    ax.set_xlim(-0.1, n)
    ax.set_ylim(0, len(rows) + 1.1)
    ax.axis("off")
    ax.set_title("Fig. 2 -- Mo hinh vung AIFS: cac AC co AIFS lon hon chi bat dau "
                 "dem lui tu vung sau\n"
                 f"$Z_j$ = {d['z_count']}   |   $\\pi_j$ = "
                 f"{np.round(d['pi'], 3).tolist()}")
    return save(fig, "fig02_aifs_zone_model")


# ===========================================================================
# Fig. 3 -- do nhay tham so (doi chieu Fig. 3 cua paper)
# ===========================================================================


def plot_fig3(d: Dict) -> str:
    """(a) mat 3D theta(AIFSN_2, TXOP_2) cho AC1/AC2, (b) P_loss theo AIFSN_2.

    Bo cuc, thang truc va he mau bam theo Fig. 3 cua paper: hai mat 3D long
    nhau (AC1 co dinh -- cam, AC2 thay doi -- xanh) va mot do thi ban log
    ben duoi.
    """
    paper_style()
    aifsn2 = np.array(d["aifsn2"], dtype=float)
    txop2 = np.array(d["txop2"], dtype=float)
    th1 = np.array(d["theta_ac1"])          # (n_aifsn, n_txop)
    th2 = np.array(d["theta_ac2"])

    fig = plt.figure(figsize=(6.8, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.40,
                          left=0.11, right=0.97, top=0.99, bottom=0.09)

    # --- (a) hai mat 3D --------------------------------------------------
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    X, Y = np.meshgrid(aifsn2, txop2, indexing="ij")
    common = dict(rstride=1, cstride=1, linewidth=0.25, antialiased=True,
                  edgecolors="#3a3a3a", shade=False)
    ax.plot_surface(X, Y, th1, color=PAPER["ac1"], **common)
    ax.plot_surface(X, Y, th2, color=PAPER["ac2"], **common)
    ax.set_xlabel("AC$_2$  AIFSN", labelpad=10)
    ax.set_ylabel("AC$_2$  TXOP ($\\mu$s)", labelpad=24)
    ax.set_zlabel("Delayed Reliability Index $\\theta$", labelpad=1)
    ax.set_xlim(aifsn2.min(), aifsn2.max())
    ax.set_ylim(txop2.min(), txop2.max())
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 2000, 4000, 6000, 8000])
    # Goc nhin chon sao cho truc theta nam BEN TRAI va hai mat cat nhau o giua
    # khung hinh, giong Fig. 3a cua paper.
    ax.view_init(elev=20, azim=-122)
    ax.set_box_aspect((1.45, 1.0, 0.72), zoom=1.02)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor("#d0d0d0")
    ax.legend(handles=[Patch(facecolor=PAPER["ac1"], edgecolor="#3a3a3a",
                             label="AC$_1$ (Fixed)"),
                       Patch(facecolor=PAPER["ac2"], edgecolor="#3a3a3a",
                             label="AC$_2$ (Varied)")],
              loc="upper right", bbox_to_anchor=(1.08, 1.02), fontsize=9)
    ax.set_title("(a)", y=-0.10, fontsize=12)

    # --- (b) P_loss ------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(aifsn2, d["ploss_ac1"], "-o", color=PAPER["ac1"],
                 markerfacecolor="none", markeredgewidth=1.3,
                 label="AC$_1$  (Fixed)")
    ax2.semilogy(aifsn2, d["ploss_ac2"], "-s", color=PAPER["ac2"],
                 markerfacecolor="none", markeredgewidth=1.3,
                 label="AC$_2$  (Varied)")
    ax2.set_xlabel("AC$_2$  AIFSN")
    ax2.set_ylabel("Packet Loss Rate $P_{loss}$")
    ax2.set_xlim(aifsn2.min(), aifsn2.max())
    ax2.set_xticks([0, 5, 10, 15])
    ax2.grid(True, which="both", linewidth=0.5, color="#d8d8d8")
    ax2.legend(loc="center right", fontsize=9.5)
    ax2.set_title("(b)", y=-0.36, fontsize=12)

    return save(fig, "fig03_parameter_sensitivity")


# ===========================================================================
# Fig. 4 -- hoi tu cua GA (doi chieu Fig. 4 cua paper)
# ===========================================================================


def plot_fig4(d: Dict) -> str:
    """Bon duong: best/mean cua EDCA don link (xanh) va MLO EDCA (do)."""
    paper_style()
    hist = d["history"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    style = {"EDCA": PAPER["edca"], "MLO EDCA": PAPER["mlo"]}
    for tag in ("EDCA", "MLO EDCA"):
        h = hist[tag]
        ax.plot(np.arange(len(h["best"])), h["best"], "-", color=style[tag],
                label=f"{tag} Best")
    for tag in ("EDCA", "MLO EDCA"):
        h = hist[tag]
        ax.plot(np.arange(len(h["mean"])), h["mean"], "--", color=style[tag],
                linewidth=1.5, label=f"{tag} Mean")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Value")
    n_max = max(len(hist[t]["best"]) for t in hist)
    ax.set_xlim(0, max(300, int(np.ceil(n_max / 50.0) * 50)))
    # Thang truc tung cua paper la [-10, 40]; chi noi ra neu duong best vuot 40
    # (khong bao gio cat mat du lieu).
    top = max(40.0, float(np.ceil(max(max(hist[t]["best"]) for t in hist)
                                  / 10.0) * 10))
    ax.set_ylim(-10, top)
    ax.set_yticks(np.arange(-10, top + 1, 10))
    ax.grid(True, linewidth=0.5, color="#d8d8d8")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    return save(fig, "fig04_ga_convergence")


# ===========================================================================
# Fig. 5 -- so sanh QoS ba cau hinh (doi chieu Fig. 5 cua paper)
# ===========================================================================


def _bar_group(ax, x, values, width, offset, color, label, lo):
    """Ve mot nhom cot tren truc log.

    Gia tri NHO HON can duoi cua truc (giu dung thang do cua paper) van phai
    nhin thay duoc, neu khong cot bi bien mat hoan toan. Cac cot nhu vay duoc
    ve thanh mot doan ngan sat day truc va GACH CHEO de danh dau "nam duoi
    khung hinh" -- gia tri that co trong `results/fig45.json`.
    """
    v = np.asarray(values, dtype=float)
    under = v < lo
    height = np.where(under, lo * 2.2, np.maximum(v, lo))
    bars = ax.bar(x + offset, height, width, color=color, label=label,
                  edgecolor="#2b2b2b", linewidth=0.6)
    for b, u in zip(bars, under):
        if u:
            b.set_hatch("///")
    return bars


def plot_fig5(d: Dict) -> str:
    """(a) Pr(D > D_max) kem nguong eps_i, (b) P_loss -- ba cau hinh."""
    paper_style()
    names = [f"AC$_{i + 1}$" for i in range(len(d["ac_names"]))]
    eps = np.array(d["epsilon"])
    x = np.arange(len(names), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # --- (a) xac suat vi pham tre + nguong -------------------------------
    w = 0.20
    lo_a = 1e-8
    series_a = [
        ("Opt. EDCA", PAPER["bar_edca"], d["configs"]["Opt. EDCA"]["violation"]),
        ("Opt. MLO EDCA", PAPER["bar_mlo"], d["configs"]["Opt. MLO EDCA"]["violation"]),
        ("Default EDCA", PAPER["bar_def"], d["configs"]["Default EDCA"]["violation"]),
        ("Target $\\epsilon_i$", PAPER["bar_eps"], eps)]
    for k, (lab, col, vals) in enumerate(series_a):
        _bar_group(axes[0], x, vals, w, (k - 1.5) * w, col, lab, lo_a)
    axes[0].set_ylim(lo_a, 1.0)
    axes[0].set_ylabel("$\\Pr(D > D_{\\max})$")

    # --- (b) xac suat mat goi --------------------------------------------
    w = 0.26
    lo_b = 1e-10
    series_b = [
        ("Opt. EDCA", PAPER["bar_edca"], d["configs"]["Opt. EDCA"]["p_loss"]),
        ("Opt. MLO EDCA", PAPER["bar_mlo"], d["configs"]["Opt. MLO EDCA"]["p_loss"]),
        ("Default EDCA", PAPER["bar_def"], d["configs"]["Default EDCA"]["p_loss"])]
    for k, (lab, col, vals) in enumerate(series_b):
        _bar_group(axes[1], x, vals, w, (k - 1) * w, col, lab, lo_b)
    axes[1].set_ylim(lo_b, 1.0)
    axes[1].set_ylabel("Packet Loss Probability ($P_{loss}$)")

    for ax, sub, loc, series in ((axes[0], "(a)", "upper left", series_a),
                                 (axes[1], "(b)", "lower right", series_b)):
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("Access Category (AC)")
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.grid(True, which="both", axis="y", linewidth=0.5, color="#dcdcdc")
        ax.grid(False, axis="x")
        # legend dung o mau SACH (khong gach cheo) du cot dau tien co the la
        # cot bi kep xuong day truc
        ax.legend(handles=[Patch(facecolor=col, edgecolor="#2b2b2b", label=lab)
                           for lab, col, _ in series],
                  loc=loc, fontsize=8.5)
        ax.set_title(sub, y=-0.36, fontsize=12)

    fig.tight_layout()
    fig.text(0.5, -0.02, "cot gach cheo: gia tri nam DUOI can duoi cua truc "
                         "(thang truc giu dung nhu Fig. 5 cua paper)",
             ha="center", fontsize=7.5, color="#555555")
    return save(fig, "fig05_qos_comparison")


# ===========================================================================
# Fig. 6 -- anh huong cua nguong eps_1 (doi chieu Fig. 6 cua paper)
# ===========================================================================


def plot_fig6(d: Dict) -> str:
    """(a) gia tri thich nghi theo eps_1, (b) tong chi so tin cay tre."""
    paper_style()
    eps1 = np.array(d["eps1"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    spec = (("EDCA", PAPER["edca"], "-o"), ("MLO EDCA", PAPER["mlo"], "--^"))
    for tag, col, fmt in spec:
        axes[0].semilogx(eps1, np.array(d[tag]["fitness"], dtype=float), fmt,
                         color=col, label=tag, markerfacecolor="none",
                         markeredgewidth=1.4)
        axes[1].semilogx(eps1, np.array(d[tag]["sum_theta"], dtype=float), fmt,
                         color=col, label=tag, markerfacecolor="none",
                         markeredgewidth=1.4)
    axes[1].semilogx(eps1, d["target_sum_theta"], ":", color=PAPER["target"],
                     linewidth=1.8, label="Target $\\epsilon$")

    axes[0].set_ylabel("Fitness Value")
    # Thang truc tung cua paper la [0, 40]; chi noi ra neu du lieu vuot 40.
    fit_max = np.nanmax([np.nanmax(np.array(d[t]["fitness"], dtype=float))
                         for t, _, _ in spec])
    axes[0].set_ylim(0, max(40.0, float(np.ceil(fit_max / 10.0) * 10)))
    axes[1].set_ylabel("$\\sum_{k=1}^{I_{all}}\\ \\theta_k$")

    both = np.concatenate([np.array(d[t]["sum_theta"], dtype=float)
                           for t, _, _ in spec]
                          + [np.array(d["target_sum_theta"], dtype=float)])
    both = both[np.isfinite(both)]
    axes[1].set_ylim(np.floor(both.min() - 1), np.ceil(both.max() + 1))

    for ax, sub, loc in ((axes[0], "(a)", "center right"),
                         (axes[1], "(b)", "upper right")):
        ax.set_xlabel("$\\epsilon_1$")
        ax.set_xlim(eps1.min(), eps1.max())
        ax.grid(True, which="both", linewidth=0.5, color="#dcdcdc")
        ax.legend(loc=loc, fontsize=9)
        ax.set_title(sub, y=-0.36, fontsize=12)

    fig.tight_layout()
    return save(fig, "fig06_epsilon_impact")

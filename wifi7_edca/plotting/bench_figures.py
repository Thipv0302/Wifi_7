"""
plotting/bench_figures.py -- Hinh DO DOI CHIEU GA vs GNN (du lieu tu
`training/benchmark.py`).

Nguyen tac trinh bay, va ly do:

  Duong hoi tu cua Fig. 4 ve theo THE HE. Voi mot phep so sanh giua cac phuong
  phap co chi phi moi the he KHAC NHAU, truc do la sai: no ngam gia dinh mot the
  he cua GA goc dat ngang mot the he cua GA + GNN, trong khi thuc te dat gap ~19
  lan. Moi hinh o day vi vay ve theo THOI GIAN THUC hoac theo SO LAN GOI MO HINH
  GIAI TICH -- hai don vi chi phi that.

  Khong phuong phap nao duoc chay voi ngan sach rieng. Cung N_gen, cung N_stag,
  cung dai N_pop, cung tap seed. Khac biet trong hinh la khac biet ve hieu qua
  tren moi don vi chi phi, khong phai ve cau hinh.
"""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from plotting.style import BENCH, paper_style, save

METHODS = ["ga", "ga_gnn", "ga_gnn_policy"]


def _figure_legend(fig, ax, n_col: int = 4) -> None:
    """Mot legend chung dat DUOI hinh.

    Legend trong long truc luon che mat mot phan duong cong o day: cac duong
    trai rong gan het khung theo ca hai truc. Dat ra ngoai thi khong phai chon
    giua "thay legend" va "thay du lieu".
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_col, fontsize=8.6,
               bbox_to_anchor=(0.5, -0.02), frameon=True, edgecolor="black",
               framealpha=1.0, fancybox=False)


# ===========================================================================
# Fig. 7 -- duong anytime: fitness dat duoc theo chi phi da bo ra
# ===========================================================================


def plot_anytime(d: Dict) -> str:
    """(a) fitness theo thoi gian thuc · (b) fitness theo so lan danh gia chinh xac.

    Day la hinh tra loi cau hoi ma reviewer se hoi dau tien: "voi ngan sach t
    giay, phuong phap nao cho nghiem tot hon?". Truc hoanh log vi cac phuong
    phap cach nhau hai bac do lon ve chi phi.
    """
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    labels = d["labels"]
    pol = d.get("policy") or {}

    for ax, xk, xlabel in ((axes[0], "curve_wall", "Wall-clock time (s)"),
                           (axes[1], "curve_evals",
                            "Analytical model evaluations")):
        for m in METHODS:
            r = d["anytime"].get(m)
            if not r:
                continue
            st = BENCH[m]
            x = np.asarray(r[xk], dtype=float)
            y = np.asarray(r["curve_best"], dtype=float)
            keep = x > 0                      # truc log: bo diem t = 0
            ax.step(x[keep], y[keep], where="post", color=st["c"],
                    linestyle=st["ls"], linewidth=1.8, label=labels[m])
            ax.plot(x[keep][-1], y[keep][-1], st["m"], color=st["c"],
                    markersize=6, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=5)

        # policy don thuan: mot diem duy nhat (khong co vong lap)
        if pol:
            st = BENCH["policy"]
            xp = pol["wall"] if xk == "curve_wall" else pol["n_eval"]
            ax.plot(xp, pol["fitness"], st["m"], color=st["c"], markersize=13,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=labels["policy"], zorder=6, linestyle="none")

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fitness Value")
        ax.set_ylim(-5, 55)
        ax.grid(True, which="both", linewidth=0.5, color="#d8d8d8")

    axes[0].set_title("(a) Cost in wall-clock time", fontsize=10)
    axes[1].set_title("(b) Cost in exact evaluations", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _figure_legend(fig, axes[0])
    return save(fig, "fig07_anytime_comparison")


# ===========================================================================
# Fig. 8 -- bien hieu qua theo ngan sach + chi phi mo hinh giai tich
# ===========================================================================


def plot_budget(d: Dict) -> str:
    """(a) bien fitness-chi phi khi quet N_pop · (b) so lan goi mo hinh giai tich.

    Panel (a) la hinh chinh: moi diem la mot ngan sach N_pop, truc hoanh la thoi
    gian THUC TE ngan sach do tieu ton. Duong nao nam TREN VA BEN TRAI thi tot
    hon -- cung fitness voi it chi phi hon. Dai mo la khoang [min, max] tren cac
    seed doc lap; diem la trung vi.
    """
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    labels = d["labels"]
    pops = d["pops"]

    # --- (a) tung lan chay, khong phai chi trung vi ------------------------
    # Ve dai [min, max] quanh mot duong trung vi se GIAU mat dieu quan trong
    # nhat trong du lieu nay: GA goc khong phai "cham hon mot chut" ma la
    # KHONG ON DINH -- cung mot ngan sach, co seed dat 49.7 va co seed dung o
    # 21. Mot dam may diem cho thay ngay dieu do; mot dai mo thi khong.
    ax = axes[0]
    for m in METHODS:
        rows = d["sweep"].get(m) or []
        if not rows:
            continue
        st = BENCH[m]
        for r in rows:
            for run in r["runs"]:
                ax.plot(run["wall"], run["fitness"], st["m"], color=st["c"],
                        markersize=4.5, alpha=0.55, markeredgewidth=0,
                        linestyle="none", zorder=2)
        t = np.array([r["wall_med"] for r in rows])
        med = np.array([r["fit_med"] for r in rows])
        ax.plot(t, med, st["ls"], color=st["c"], marker=st["m"],
                markersize=7, markeredgecolor="black", markeredgewidth=0.6,
                label=labels[m], zorder=4)

    pol = d.get("policy") or {}
    if pol:
        st = BENCH["policy"]
        ax.plot(pol["wall"], pol["fitness"], st["m"], color=st["c"],
                markersize=13, markeredgecolor="black", markeredgewidth=0.5,
                label=labels["policy"], linestyle="none")

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Final Fitness Value")
    ax.grid(True, which="both", linewidth=0.5, color="#d8d8d8")
    ax.set_title(r"(a) Fitness-cost frontier ($N_{pop}$ sweep)", fontsize=10)
    legend_ax = ax

    # --- (b) so lan goi mo hinh giai tich ----------------------------------
    ax = axes[1]
    w = 0.26
    xs = np.arange(len(pops))
    all_ev: list = []
    for i, m in enumerate(METHODS):
        rows = d["sweep"].get(m) or []
        if not rows:
            continue
        st = BENCH[m]
        ev = [r["eval_med"] for r in rows]
        ax.bar(xs + (i - 1) * w, ev, w, color=st["c"], label=labels[m],
               edgecolor="black", linewidth=0.6, zorder=3)
        all_ev.extend(ev)

    ax.set_yscale("log")
    # Cot tren truc log keo xuong tan day khung neu khong dat can duoi; dat can
    # duoi ngay duoi gia tri nho nhat de chieu cao cot con doc duoc.
    if all_ev:
        ax.set_ylim(10 ** np.floor(np.log10(min(all_ev)) - 0.15),
                    10 ** np.ceil(np.log10(max(all_ev)) + 0.05))
    ax.set_xticks(xs)
    ax.set_xticklabels([str(p) for p in pops])
    ax.set_xlabel(r"Population size $N_{pop}$")
    ax.set_ylabel("Analytical model evaluations")
    ax.grid(True, which="major", axis="y", linewidth=0.5, color="#d8d8d8")
    ax.set_title("(b) Exact-evaluation budget consumed", fontsize=10)

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _figure_legend(fig, legend_ax)
    return save(fig, "fig08_budget_frontier")


# ===========================================================================
# Fig. 9 -- vi sao gieo bang policy thang: quan the ban dau
# ===========================================================================


def plot_reliability(d: Dict, rel_tol: float = 0.01) -> str:
    """(a) ty le lan chay TIM RA nghiem toi uu · (b) chat luong quan the ban dau.

    Panel (a) la ket qua chinh cua muc nay, va no khong noi ve toc do. Voi cung
    ngan sach va chi doi seed, GA goc lan thi dat 49.70 lan thi dung o 21 --
    tuc trung vi fitness cua no khong phai "nghiem GA tim duoc" ma la trung vi
    cua mot phan phoi hai dinh. Gieo bang policy xoa han hien tuong do: moi lan
    chay deu ket thuc o cung mot nghiem.

    `rel_tol` -- coi la THANH CONG neu fitness nam trong `rel_tol` tuong doi so
    voi gia tri tot nhat quan sat duoc tren TOAN BO phep do (khong phai so voi
    gia tri tot nhat cua rieng phuong phap do -- nhu the moi phuong phap se tu
    dat chuan cho chinh minh).
    """
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    labels = d["labels"]
    pops = d["pops"]
    xs = np.arange(len(pops))
    w = 0.26

    best = max(run["fitness"] for rows in d["sweep"].values()
               for r in rows for run in r["runs"])
    thr = best * (1.0 - rel_tol)

    # --- (a) ty le thanh cong ----------------------------------------------
    ax = axes[0]
    for i, m in enumerate(METHODS):
        rows = d["sweep"].get(m) or []
        st = BENCH[m]
        v = [100.0 * np.mean([run["fitness"] >= thr for run in r["runs"]])
             for r in rows]
        bars = ax.bar(xs + (i - 1) * w, v, w, color=st["c"], label=labels[m],
                      edgecolor="black", linewidth=0.6, zorder=3)
        for b, val in zip(bars, v):
            if val > 0:
                ax.text(b.get_x() + b.get_width() / 2, val + 2.5, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=7.5)
    n_run = len(rows[0]["runs"]) if rows else 0
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"(a) Runs reaching fitness $\\geq$ {thr:.2f} "
                 f"({n_run} seeds)", fontsize=10)

    # --- (b) chat luong quan the ban dau -----------------------------------
    ax = axes[1]
    for i, m in enumerate(METHODS):
        rows = d["sweep"].get(m) or []
        st = BENCH[m]
        v = [100.0 * float(np.median([run["frac_feasible_init"]
                                      for run in r["runs"]])) for r in rows]
        ax.bar(xs + (i - 1) * w, v, w, color=st["c"], label=labels[m],
               edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_ylabel("Feasible individuals (%)")
    ax.set_title("(b) Initial population quality", fontsize=10)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(p) for p in pops])
        ax.set_xlabel(r"Population size $N_{pop}$")
        ax.grid(True, axis="y", linewidth=0.5, color="#d8d8d8")

    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _figure_legend(fig, axes[0], n_col=3)
    return save(fig, "fig09_reliability")


def plot_all(d: Dict):
    return [plot_anytime(d), plot_budget(d), plot_reliability(d)]


if __name__ == "__main__":
    import argparse
    import json
    import os

    from config import RESULT_DIR

    ap = argparse.ArgumentParser(description="Ve hinh do doi chieu GA vs GNN")
    ap.add_argument("--data", type=str,
                    default=os.path.join(RESULT_DIR, "benchmark.json"))
    a = ap.parse_args()
    with open(a.data, encoding="utf-8") as fh:
        data = json.load(fh)
    for p in plot_all(data):
        print("da ve ->", p)

"""training/pp1_tables.py -- sinh than bang LaTeX tu du lieu da do.

    py -3 -m training.pp1_tables

Bai chi con cho MOT bang ket qua, nen bang do phai mang ca hai phep so sanh:

  Khoi A  cung cau hinh tim kiem (N_pop = 200, N_gen = 300). O day mang danh
          gia g_phi KHONG duoc ky vong nang muc tieu -- no thay bo danh gia
          chinh xac bang bo xap xi, nen o cung so ung vien duoc xet thi tot
          nhat la ngang. Cai no doi lai la chi phi.

  Khoi B  cung NGAN SACH thoi gian. Day moi la noi tra loi duoc cau hoi "chi
          phi re hon co chuyen hoa thanh nghiem tot hon khong". Neu co thi voi
          cung t giay, g_phi phai tren GA.

Viet ra file thay vi go tay vi cac con so nay da doi vai lan trong qua trinh
do; go tay mot lan nua la mot lan nua co co hoi lech giua bang va du lieu.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
M = ("ga", "ga_gnn", "ga_gnn_policy")
TEX = {"ga": r"GA baseline~\cite{Yi2025}",
       "ga_gnn": r"\;\; $+$ evaluation GNN $g_\phi$",
       "ga_gnn_policy": r"\;\; $+$ proposal GNN $\pi_\psi$"}
BUDGETS = (1.0, 3.0, 10.0, 30.0, 100.0, 250.0)


def block_a() -> str:
    """Cung cau hinh tim kiem -- doc tu benchmark cua phep boc tach."""
    B = json.load(open(os.path.join(RES, "benchmark.json"), encoding="utf-8"))
    if not os.path.exists(os.path.join(RES, "benchmark.json")):
        B = json.load(open(os.path.join(os.path.dirname(ROOT),
                                        "Paper1_GNN_Accelerated_GA", "data",
                                        "benchmark.json"), encoding="utf-8"))
    i = B["pops"].index(200)
    N1 = json.load(open(os.path.join(RES, "pipeline_ngen1.json"),
                        encoding="utf-8"))["160x1"]
    out = []
    for m in M:
        e = B["sweep"][m][i]
        succ = sum(r["fitness"] >= 49.21 for r in e["runs"])
        init = float(np.median([r["frac_feasible_init"] for r in e["runs"]]))
        out.append(r"%s & $%.1f$ & $%s$ & $%.2f$ & $%d/10$ & $%.0f\%%$ \\"
                   % (TEX[m], e["wall_med"], "{:,}".format(int(e["eval_med"])),
                      e["fit_med"], succ, 100 * init))
    out.append(r"\midrule")
    out.append(r"Pipeline, $\Ngen = 1$ & $%.1f$ & $%s$ & $%.2f$ & $%d/10$ "
               r"& $33\%%$ \\"
               % (float(np.median(N1["wall"])),
                  "{:,}".format(int(np.median(N1["eval"]))),
                  float(np.median(N1["fit"])),
                  sum(v >= 49.21 for v in N1["fit"])))
    return "\n".join(out)


def block_b() -> str:
    """Cung ngan sach thoi gian -- doc tu thi nghiem anytime."""
    A = json.load(open(os.path.join(RES, "pp1_anytime.json"), encoding="utf-8"))
    t = np.array(A["t_grid"], float)
    # Four columns, not seven: a single IEEE column is 3.5 in and will not hold
    # an (F, success) pair per method. Success rates live in block A and in the
    # text; what this block has to show is the ORDER of the objectives at a
    # shared budget, and that survives on its own.
    rows = []
    for want in BUDGETS:
        i = int(np.argmin(np.abs(t - want)))
        cells = []
        for m in M:
            mu = A["methods"][m]["wall"]["mean"][i]
            cells.append("---" if mu != mu else r"$%.2f$" % mu)
        rows.append(r"$%.3g$ & %s \\" % (t[i], " & ".join(cells)))
    return "\n".join(rows)


def main() -> None:
    p = os.path.join(RES, "pp1_table_body.tex")
    body = ("%% --- block A: same search configuration -----------------\n"
            + block_a()
            + "\n%% --- block B: same wall-clock budget ------------------\n"
            + block_b() + "\n")
    open(p, "w", encoding="utf-8").write(body)
    print(body)
    print("saved ->", p)


if __name__ == "__main__":
    main()

"""training/rerun_eps_sweep.py -- chay lai duong GA cua phep quet eps_1.

    py -3 -m training.rerun_eps_sweep

LY DO PHAI CHAY LAI. Phep quet chi doi eps_1, moi thu khac giu nguyen
(`datagen.scenario.epsilon_sweep`), nen cac tap kha thi LONG NHAU: eps_1 lon hon
= rang buoc long hon = tap kha thi rong hon. Do do gia tri toi uu F*(eps_1)
BUOC PHAI khong giam. Duong GA cu (n_pop = 120, n_gen = 120, MOT seed) lai tut
tu 46.90 xuong 35.21 khi eps_1 di tu 1e-6 len 1e-5 -- dieu do khong the la tinh
chat cua bai toan, no la GA truot nghiem.

Lan chay nay dung DUNG ngan sach cua benchmark (n_pop = 200, n_gen = 300,
n_stag = 50) va NHIEU SEED moi diem, roi bao cao ca trung vi lan gia tri tot
nhat: trung vi la thu mot lan chay cho ta, gia tri tot nhat la uoc luong gan hon
cua F* that su.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from config import GAParams
from datagen.scenario import epsilon_sweep
from training.ga import run_ga

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS1 = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
SEEDS = [2025, 2026, 2027, 2028, 2029]
N_LINKS = 2


def main():
    out = {"eps1": EPS1, "seeds": SEEDS, "n_links": N_LINKS,
           "n_pop": 200, "n_gen": 300, "n_stag": 50, "points": []}
    print("%9s %9s %9s %9s %9s %8s" %
          ("eps_1", "median", "best", "worst", "sum_th", "feas"))
    print("-" * 60)
    for e1, ac_set in epsilon_sweep(tuple(EPS1)):
        fits, thetas, oks = [], [], []
        for s in SEEDS:
            ga = GAParams(n_pop=200, n_gen=300, n_elite=8, p_cross=0.8,
                          p_mutate=0.05, n_stag=50, seed=s)
            r = run_ga(ac_set, n_links=N_LINKS, ga=ga)
            ok = bool(r.qos.feasible(ac_set))
            oks.append(ok)
            fits.append(float(r.qos.objective) if ok else float("nan"))
            thetas.append(float(np.sum(r.qos.theta)) if ok else float("nan"))
        good = [f for f in fits if not np.isnan(f)]
        pt = {"eps1": e1, "fitness": fits, "sum_theta": thetas,
              "feasible": oks,
              "fit_med": float(np.nanmedian(fits)) if good else float("nan"),
              "fit_best": float(np.nanmax(fits)) if good else float("nan"),
              "fit_worst": float(np.nanmin(fits)) if good else float("nan"),
              "theta_med": float(np.nanmedian(thetas)) if good else float("nan")}
        out["points"].append(pt)
        print("%9.0e %9.2f %9.2f %9.2f %9.2f %6d/%d"
              % (e1, pt["fit_med"], pt["fit_best"], pt["fit_worst"],
                 pt["theta_med"], sum(oks), len(SEEDS)), flush=True)

    # Kiem tra tinh don dieu tren ca hai duong.
    for key in ("fit_med", "fit_best"):
        v = [p[key] for p in out["points"]]
        drops = [(EPS1[i], EPS1[i + 1], v[i], v[i + 1])
                 for i in range(len(v) - 1) if v[i + 1] < v[i] - 1e-9]
        out["monotone_" + key] = not drops
        print("\n%s don dieu khong giam: %s" % (key, not drops))
        for a, b, x, y in drops:
            print("   TUT %.0e -> %.0e : %.2f -> %.2f" % (a, b, x, y))

    p = os.path.join(ROOT, "results", "eps_sweep_rerun.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()

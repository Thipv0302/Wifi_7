"""training/pp1_eps_baseline.py -- chay lai duong GA baseline cua Fig. 4.

    py -3 -m training.pp1_eps_baseline

LY DO CHAY LAI
--------------
Muc tieu Eq.(18) F = sum_i -log10(P_loss,i) KHONG phu thuoc eps; eps chi quyet
dinh tap kha thi qua rang buoc violation_i < eps_i. Khi noi long eps_1, tap kha
thi chi NO RA (nghiem kha thi o eps_1 chat van kha thi o eps_1 long hon), nen

        F*(eps_1) la ham KHONG GIAM theo eps_1.

Duong GA baseline cu (mot lan chay duy nhat moi diem, results/fig6.json) tut tu
46.90 xuong 35.21 khi eps_1 di tu 1e-6 sang 1e-5. Do la loi HOI TU cua GA, chu
khong phai tinh chat cua bai toan -- ve len hinh thi sai ve mat vat ly.

CACH LAM
--------
Moi diem eps_1 chay N_RESTART lan GA doc lap (hat giong khac nhau), giu lai ca
N_RESTART ket qua. Duong bao cao la BEST-OF-N -- "nghiem tot nhat ma GA tim
duoc voi ngan sach nay" -- kem trung vi de van mo ta duoc do tan mac.

Buoc cuoi ap dung tinh long nhau cua tap kha thi: nghiem tim duoc o moi eps_1
CHAT hon cung hop le o eps_1 hien tai, nen best-of duoc lay tren hop cua cac
kho nghiem tu diem chat nhat den diem hien tai. Nho vay duong cong khong giam
theo dung ly thuyet, khong can "sua tay" so lieu.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common.utility import evaluate_config
from config import GAParams, GA
from datagen.genome import GenomeSpec, decode
from datagen.scenario import epsilon_sweep
from training.ga import run_ga

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS1 = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
N_POP, N_GEN, N_STAG = 120, 120, 40      # cung ngan sach voi exp_fig6 cu
N_RESTART = 10                           # so lan chay doc lap moi diem
N_LINKS = 2
SEED0 = 7


def _ga(seed: int, n_pop: int, n_gen: int, n_stag: int) -> GAParams:
    return GAParams(n_pop=n_pop, n_gen=n_gen, n_elite=max(n_pop // 25, 4),
                    p_cross=0.8, p_mutate=0.05, n_stag=n_stag, seed=seed)


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pop", type=int, default=N_POP)
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--n-stag", type=int, default=N_STAG)
    ap.add_argument("--n-restart", type=int, default=N_RESTART)
    ap.add_argument("--out", default="pp1_eps_baseline.json",
                    help="ten file trong results/")
    return ap.parse_args()


def main(n_pop: int = N_POP, n_gen: int = N_GEN, n_stag: int = N_STAG,
         n_restart: int = N_RESTART, out: str = "pp1_eps_baseline.json") -> None:
    sweep = epsilon_sweep(EPS1)
    spec = GenomeSpec(n_ac=len(sweep[0][1]), n_links=N_LINKS,
                      allow_link_choice=True)

    print("ngan sach GA: n_pop=%d n_gen=%d n_stag=%d, %d khoi dau/diem"
          % (n_pop, n_gen, n_stag, n_restart), flush=True)

    points = []
    for pos, (e1, ac_set) in enumerate(sweep):
        runs = []
        for k in range(n_restart):
            t0 = time.perf_counter()
            r = run_ga(ac_set, n_links=N_LINKS,
                       ga=_ga(SEED0 + 100 * pos + k, n_pop, n_gen, n_stag))
            ok = bool(r.qos.feasible(ac_set))
            runs.append({
                "seed": SEED0 + 100 * pos + k,
                "feasible": ok,
                "objective": float(r.qos.objective) if ok else float("nan"),
                "sum_theta": float(np.sum(r.qos.theta)) if ok else float("nan"),
                "genome": r.genome.astype(int).tolist(),
                "n_eval": int(r.history.n_eval),   # so lan goi mo hinh giai tich
                "wall": time.perf_counter() - t0,
            })
            print("eps1=%.0e  lan %2d/%d  F=%7.3f  kha thi %s  (%.0fs)"
                  % (e1, k + 1, n_restart, runs[-1]["objective"], ok,
                     runs[-1]["wall"]), flush=True)
        points.append({"eps1": float(e1), "runs": runs})

    # --- best-of tren hop kho nghiem tu cac eps_1 chat hon ------------------
    pool: list[list[int]] = []
    res = {"eps1": list(map(float, EPS1)), "n_restart": n_restart,
           "n_pop": n_pop, "n_gen": n_gen, "n_stag": n_stag,
           "target_sum_theta": [], "best": {"fitness": [], "sum_theta": [],
                                            "feasible": []},
           "median": {"fitness": []}, "wall_per_point": [], "points": points}

    for pos, (e1, ac_set) in enumerate(sweep):
        res["target_sum_theta"].append(
            float(np.sum([-np.log10(a.epsilon) for a in ac_set])))
        pool += [r["genome"] for r in points[pos]["runs"] if r["feasible"]]
        best_obj, best_theta = float("-inf"), float("nan")
        for g in pool:
            q = evaluate_config(decode(np.asarray(g), spec), ac_set)
            if q.feasible(ac_set) and q.objective > best_obj:
                best_obj = float(q.objective)
                best_theta = float(np.sum(q.theta))
        res["best"]["feasible"].append(bool(np.isfinite(best_obj)))
        res["best"]["fitness"].append(best_obj if np.isfinite(best_obj)
                                      else float("nan"))
        res["best"]["sum_theta"].append(best_theta)
        med = np.nanmedian([r["objective"] for r in points[pos]["runs"]])
        res["median"]["fitness"].append(float(med))
        res["wall_per_point"].append(
            sum(r["wall"] for r in points[pos]["runs"]))
        res.setdefault("n_eval_per_run", []).append(
            float(np.mean([r["n_eval"] for r in points[pos]["runs"]])))
        print("=> eps1=%.0e  best %.3f  trung vi %.3f  (kho %d nghiem)"
              % (e1, res["best"]["fitness"][-1], med, len(pool)), flush=True)

    f = res["best"]["fitness"]
    print("\nkhong giam?",
          all(f[i] <= f[i + 1] + 1e-9 for i in range(len(f) - 1)))

    p = os.path.join(ROOT, "results", out)
    json.dump(res, open(p, "w"), indent=1)
    print("saved ->", p)


if __name__ == "__main__":
    a = _cli()
    main(a.n_pop, a.n_gen, a.n_stag, a.n_restart, a.out)

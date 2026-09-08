"""training/pp1_eps_methods.py -- quet eps_1 x ngan sach x PHUONG PHAP.

    py -3 -m training.pp1_eps_methods --n-pop 40 --n-gen 40 --n-stag 13 \
        --out pp1_methods_40x40.json

Mo rong `pp1_eps_baseline` sang ca ba phuong phap cua bai (cung ba duong da
dung o Fig. 3b va Bang I), de duong cong chi phi -- chat luong co du cac muc
cua phep boc tach (ablation):

  ga             GA tren mo hinh giai tich (baseline Yi et al.)
  ga_gnn         GA + mang danh gia (surrogate) sang loc truoc
  ga_gnn_policy  GA + mang danh gia + mang de xuat gieo quan the ban dau

Moi (eps_1, phuong phap, ngan sach) chay N_RESTART lan doc lap. Bao cao
best-of-N tren HOP kho nghiem tu cac eps_1 chat hon -- xem giai thich trong
`pp1_eps_baseline`; nho tinh long nhau cua tap kha thi ma duong cong khong
giam theo dung ly thuyet.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from common.utility import evaluate_config
from config import GAParams
from datagen.genome import GenomeSpec, decode
from datagen.scenario import epsilon_sweep
from training.compare_seeding import policy_seeds
from training.ga import run_ga
from training.ga_surrogate import run_ga_surrogate
from training.gnn import Surrogate
from training.train_policy import load_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS1 = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
POLICY = os.path.join(ROOT, "results", "var_1500_2027.pt")
METHODS = ("ga", "ga_gnn", "ga_gnn_policy")
N_LINKS = 2
SEED0 = 7


def _ga(seed: int, n_pop: int, n_gen: int, n_stag: int) -> GAParams:
    return GAParams(n_pop=n_pop, n_gen=n_gen, n_elite=max(n_pop // 25, 4),
                    p_cross=0.8, p_mutate=0.05, n_stag=n_stag, seed=seed)


def _run(method, ac_set, ga, sur, pol):
    """Mot lan chay cua mot phuong phap. Tra ve (GAResult, so lan goi GNN)."""
    if method == "ga":
        return run_ga(ac_set, n_links=N_LINKS, ga=ga), 0
    seeds = None
    if method == "ga_gnn_policy":
        seeds = policy_seeds(pol, sur, ac_set, N_LINKS,
                             n_seed=ga.n_pop, temperature=1.6)
    r = run_ga_surrogate(ac_set, sur, n_links=N_LINKS, ga=ga, seed=ga.seed,
                         verify_top=8, seed_genomes=seeds)
    return r, 0


def main(n_pop: int, n_gen: int, n_stag: int, n_restart: int,
         out: str, methods=METHODS) -> None:
    sweep = epsilon_sweep(EPS1)
    spec = GenomeSpec(n_ac=len(sweep[0][1]), n_links=N_LINKS,
                      allow_link_choice=True)
    sur = Surrogate.load(os.path.join(ROOT, "results", "gnn_model.pt"), DEV)
    pol = load_policy(POLICY, DEV)

    print("ngan sach: n_pop=%d n_gen=%d n_stag=%d, %d khoi dau/diem, PP: %s"
          % (n_pop, n_gen, n_stag, n_restart, ", ".join(methods)), flush=True)

    res = {"eps1": list(map(float, EPS1)), "n_restart": n_restart,
           "n_pop": n_pop, "n_gen": n_gen, "n_stag": n_stag,
           "target_sum_theta": [float(np.sum([-np.log10(a.epsilon)
                                              for a in acs]))
                                for _, acs in sweep],
           "methods": {}}

    for method in methods:
        points, pool = [], []
        best_f, best_t, med_f, wall_pt, ncall = [], [], [], [], []
        for pos, (e1, ac_set) in enumerate(sweep):
            runs = []
            for k in range(n_restart):
                t0 = time.perf_counter()
                ga = _ga(SEED0 + 100 * pos + k, n_pop, n_gen, n_stag)
                r, _ = _run(method, ac_set, ga, sur, pol)
                ok = bool(r.qos.feasible(ac_set))
                runs.append({"seed": ga.seed, "feasible": ok,
                             "objective": float(r.qos.objective) if ok
                                          else float("nan"),
                             "genome": r.genome.astype(int).tolist(),
                             "n_eval": int(r.history.n_eval),
                             "wall": time.perf_counter() - t0})
            points.append({"eps1": float(e1), "runs": runs})

            pool += [r["genome"] for r in runs if r["feasible"]]
            bo, bt = float("-inf"), float("nan")
            for g in pool:
                q = evaluate_config(decode(np.asarray(g), spec), ac_set)
                if q.feasible(ac_set) and q.objective > bo:
                    bo, bt = float(q.objective), float(np.sum(q.theta))
            best_f.append(bo if np.isfinite(bo) else float("nan"))
            best_t.append(bt)
            med_f.append(float(np.nanmedian([r["objective"] for r in runs])))
            wall_pt.append(sum(r["wall"] for r in runs))
            ncall.append(float(np.mean([r["n_eval"] for r in runs])))
            print("  %-14s eps1=%.0e  best %6.2f  trung vi %6.2f  "
                  "(%4.0f goi, %5.1fs)"
                  % (method, e1, best_f[-1], med_f[-1], ncall[-1],
                     wall_pt[-1] / n_restart), flush=True)

        mono = all(best_f[i] <= best_f[i + 1] + 1e-9
                   for i in range(len(best_f) - 1))
        res["methods"][method] = {
            "best": {"fitness": best_f, "sum_theta": best_t},
            "median": {"fitness": med_f}, "wall_per_point": wall_pt,
            "n_eval_per_run": ncall, "monotone": mono, "points": points}
        print("  %-14s khong giam? %s" % (method, mono), flush=True)

    p = os.path.join(ROOT, "results", out)
    json.dump(res, open(p, "w"), indent=1)
    print("saved ->", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pop", type=int, required=True)
    ap.add_argument("--n-gen", type=int, required=True)
    ap.add_argument("--n-stag", type=int, required=True)
    ap.add_argument("--n-restart", type=int, default=10)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.n_pop, a.n_gen, a.n_stag, a.n_restart, a.out,
         tuple(a.methods.split(",")))

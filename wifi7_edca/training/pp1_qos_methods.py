"""training/pp1_qos_methods.py -- QoS tung AC cho BON cau hinh.

    py -3 -m training.pp1_qos_methods

Hinh QoS cua paper goc doi chieu bang tham so mac dinh voi nghiem cua GA. O day
them ca hai muc boc tach cua pipeline, de doc duoc cau hoi "muc nao cua pipeline
lam nen chat luong nghiem":

  Default EDCA         bang tham so IEEE 802.11e
  GA baseline          GA tren mo hinh giai tich
  + evaluation GNN     GA co g_phi sang loc ung vien
  + proposal GNN       them pi_psi gieo quan the ban dau (pipeline day du)

CACH CHON CAU HINH DEM SO SANH
------------------------------
Mot lan chay GA la mot mau cua bien ngau nhien, khong phai "nghiem cua GA":
tren kich ban nay do tan mac giua cac hat giong rong hon chenh lech giua cac
phuong phap. Neu moi phuong phap chi chay mot lan thi hinh QoS ve ra chu yeu la
nhieu hat giong. Vi vay ba phuong phap deu chay N_SEED lan doc lap voi CUNG mot
ngan sach, va cau hinh dem ve la cau hinh TOT NHAT theo muc tieu chinh xac --
tuc moi phuong phap deu duoc do o trang thai tot nhat ma no dat toi.

Moi gia tri bao cao deu do mo hinh giai tich tinh, khong phai surrogate.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from config import GAParams
from datagen.scenario import main_scenario
from training.compare_seeding import policy_seeds
from training.ga import run_ga
from training.ga_surrogate import run_ga_surrogate
from training.gnn import Surrogate
from training.train_policy import load_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NL = 2
POLICY = os.path.join(ROOT, "results", "var_1500_2027.pt")
# Cung ngan sach cho ca ba, de chenh lech chi den tu viec co mang hay khong.
N_POP, N_GEN, N_STAG = 120, 120, 40
N_SEED = 10
SEED0 = 2025


def _ga(seed: int) -> GAParams:
    return GAParams(n_pop=N_POP, n_gen=N_GEN, n_elite=max(N_POP // 25, 4),
                    p_cross=0.8, p_mutate=0.05, n_stag=N_STAG, seed=seed)


def _solve(tag, acs, sur, pol, seed):
    ga = _ga(seed)
    if tag == "ga":
        return run_ga(acs, n_links=NL, ga=ga)
    seeds = None
    if tag == "ga_gnn_policy":
        seeds = policy_seeds(pol, sur, acs, NL, n_seed=N_POP, temperature=1.6)
    return run_ga_surrogate(acs, sur, n_links=NL, ga=ga, seed=seed,
                            verify_top=8, seed_genomes=seeds)


def main() -> None:
    acs = main_scenario()
    sur = Surrogate.load(os.path.join(ROOT, "results", "gnn_model.pt"), DEV)
    pol = load_policy(POLICY, DEV)

    out = {"ac": [a.name for a in acs], "eps": [a.epsilon for a in acs],
           "n_seed": N_SEED, "n_pop": N_POP, "n_gen": N_GEN, "configs": {}}

    F = json.load(open(os.path.join(ROOT, "results", "fig45.json"),
                       encoding="utf-8"))
    c = F["configs"]["Default EDCA"]
    out["configs"]["default"] = {"violation": c["violation"],
                                 "p_loss": c["p_loss"],
                                 "objective": c["objective"],
                                 "feasible": c["feasible"]}

    for tag in ("ga", "ga_gnn", "ga_gnn_policy"):
        best, runs = None, []
        for k in range(N_SEED):
            t0 = time.perf_counter()
            r = _solve(tag, acs, sur, pol, SEED0 + k)
            ok = bool(r.qos.feasible(acs))
            obj = float(r.qos.objective) if ok else float("nan")
            runs.append({"seed": SEED0 + k, "feasible": ok, "objective": obj,
                         "n_eval": int(r.history.n_eval),
                         "wall": time.perf_counter() - t0})
            if ok and (best is None or obj > best[0]):
                best = (obj, r)
            print("  %-14s hat giong %2d  F = %7.3f  kha thi %s"
                  % (tag, SEED0 + k, obj, ok), flush=True)
        obj, r = best
        objs = [x["objective"] for x in runs]
        out["configs"][tag] = {
            "violation": r.qos.violation.tolist(),
            "p_loss": r.qos.p_loss.tolist(),
            "objective": obj, "feasible": True,
            "objective_median": float(np.nanmedian(objs)),
            "objective_min": float(np.nanmin(objs)),
            "n_eval_median": float(np.median([x["n_eval"] for x in runs])),
            "wall_median": float(np.median([x["wall"] for x in runs])),
            "runs": runs}
        print("%-14s tot nhat %.3f  trung vi %.3f"
              % (tag, obj, out["configs"][tag]["objective_median"]), flush=True)

    print()
    for tag, c in out["configs"].items():
        print("%-14s Pr(D>=Dmax): %s" % (
            tag, " ".join("%.1e" % v for v in c["violation"])))

    p = os.path.join(ROOT, "results", "pp1_qos_methods.json")
    json.dump(out, open(p, "w"), indent=1)
    print("saved ->", p)


if __name__ == "__main__":
    main()

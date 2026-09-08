"""training/pp1_anytime.py -- ba phuong phap tren CUNG mot truc ngan sach.

    py -3 -m training.pp1_anytime --n-seed 20

TAI SAO CAN THI NGHIEM NAY
--------------------------
Bang I so sanh ba phuong phap o cung (N_pop, N_gen). O phep so sanh do, viec
mang danh gia g_phi KHONG cho muc tieu cao hon GA thuan la dung nhu mong doi:
g_phi chi thay bo danh gia dat bang mot bo xep hang re, nen o cung so ung vien
duoc xet, no cung lam khong hon -- tham chi hoi kem, vi no co sai so. Cai no
doi lai la chi phi: 1.483 lan goi mo hinh giai tich thay vi 29.309.

Nhung "cung (N_pop, N_gen)" khong phai "cung ngan sach tinh toan". Cung mot
luoi tham so tieu ton cua ba phuong phap 12.426, 1.070 va 474 lan goi mo hinh
giai tich. Cau hoi dung de hoi la:

    VOI CUNG MOT LUONG TINH TOAN, phuong phap nao dat muc tieu cao hon?

Neu g_phi that su huu ich thi voi cung t giay no phai sang loc duoc nhieu ung
vien hon, va do do dat muc tieu cao hon GA thuan:

    F_{g_phi}(t) > F_{GA}(t).

Do moi la bang chung rang do chinh xac du bao (rho = 0.9996) chuyen hoa thanh
loi ich toi uu hoa, chu khong dung lai o mot chi so hoi quy.

CACH LAM
--------
Moi phuong phap chay N_SEED lan doc lap voi ngan sach ROI RAI (n_gen lon, khong
dung som), va moi the he ghi lai (thoi gian, so lan goi chinh xac, nghiem tot
nhat den luc do) -- xem `GAHistory.wall` / `.evals`. Sau do voi moi moc ngan
sach t ta doc ra gia tri ma tung lan chay dat duoc TAI thoi diem do, roi bao
cao trung binh va ty le thanh cong tren cac seed.

Moc ngan sach duoc quet tren CA HAI truc:
  * thoi gian thuc  -- de doc, nhung phu thuoc may;
  * so lan goi mo hinh giai tich -- khong phu thuoc may, la truc chuyen duoc.
"""
from __future__ import annotations

import argparse
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
POLICY = os.path.join(ROOT, "results", "var_1500_2027.pt")
METHODS = ("ga", "ga_gnn", "ga_gnn_policy")
NL = 2
SEED0 = 2025
SUCCESS = 49.21          # nguong "thanh cong" dung xuyen suot bai

# Ngan sach roi rai va KHONG dung som: muon doc duong cong den het thi phai de
# no chay het. n_stag lon vo hieu hoa tieu chi dung som.
N_POP, N_GEN, N_STAG = 200, 300, 10 ** 6


def _trace(r, t_total):
    """(thoi gian, so lan goi, nghiem tot nhat) tai moi the he, da lam don dieu.

    `best` trong lich su la nghiem tot nhat DEN thoi diem do nen von da khong
    giam; np.maximum.accumulate chi de phong truong hop bo tinh chinh cuoi cung
    day them mot bac sau vong lap.
    """
    n = min(len(r.history.wall), len(r.history.evals), len(r.history.best))
    w = np.asarray(r.history.wall[:n], float)
    e = np.asarray(r.history.evals[:n], float)
    b = np.maximum.accumulate(np.asarray(r.history.best[:n], float))
    # Buoc tinh chinh cuc bo chay SAU vong lap, nen diem cuoi cung mang tong
    # chi phi that cua ca lan chay.
    w = np.append(w, t_total)
    e = np.append(e, r.history.n_eval)
    b = np.append(b, max(b[-1], float(r.fitness)))
    return w, e, b


def _at(x, y, grid):
    """Gia tri y tai moc x = grid (giu bac thang: gia tri cua moc gan nhat da qua).

    Truoc moc dau tien phuong phap chua co gi de bao cao, nen tra ve NaN thay vi
    ngoai suy nguoc -- ngoai suy o day se tang gia tri cho phuong phap cham.
    """
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        m = x <= g
        if m.any():
            out[i] = y[m][-1]
    return out


def main(n_seed: int, out: str) -> None:
    acs = main_scenario()
    sur = Surrogate.load(os.path.join(ROOT, "results", "gnn_model.pt"), DEV)
    pol = load_policy(POLICY, DEV)

    runs = {m: [] for m in METHODS}
    for m in METHODS:
        for k in range(n_seed):
            seed = SEED0 + k
            ga = GAParams(n_pop=N_POP, n_gen=N_GEN, n_elite=max(N_POP // 25, 4),
                          p_cross=0.8, p_mutate=0.05, n_stag=N_STAG, seed=seed)
            t0 = time.perf_counter()
            if m == "ga":
                r = run_ga(acs, n_links=NL, ga=ga)
            else:
                seeds = (policy_seeds(pol, sur, acs, NL, n_seed=N_POP,
                                      temperature=1.6)
                         if m == "ga_gnn_policy" else None)
                r = run_ga_surrogate(acs, sur, n_links=NL, ga=ga, seed=seed,
                                     verify_top=8, seed_genomes=seeds)
            t = time.perf_counter() - t0
            w, e, b = _trace(r, t)
            runs[m].append({"seed": seed, "wall": w.tolist(),
                            "evals": e.tolist(), "best": b.tolist(),
                            "final": float(r.fitness), "total_wall": t,
                            "n_eval": int(r.history.n_eval),
                            "frac_feasible_init": float(r.history.frac_feasible[0])})
            print("  %-14s seed %d  F = %7.3f  %6.1fs  %6d goi"
                  % (m, seed, r.fitness, t, r.history.n_eval), flush=True)

    # --- doc ra tren luoi ngan sach chung -----------------------------------
    t_max = max(max(x["total_wall"] for x in runs[m]) for m in METHODS)
    e_max = max(max(x["n_eval"] for x in runs[m]) for m in METHODS)
    t_grid = np.logspace(np.log10(0.1), np.log10(t_max), 60)
    e_grid = np.logspace(np.log10(50), np.log10(e_max), 60)

    # Cac moc dau tien nam truoc khi bat ky lan chay nao kip bao cao, nen ca
    # lat cat deu la NaN; do la dung y do, chi can dung canh bao lai.
    import warnings
    warnings.filterwarnings("ignore", r".*empty slice.*")
    warnings.filterwarnings("ignore", r".*Degrees of freedom.*")

    res = {"n_seed": n_seed, "success": SUCCESS, "n_pop": N_POP,
           "n_gen": N_GEN, "t_grid": t_grid.tolist(),
           "e_grid": e_grid.tolist(), "methods": {}}

    for m in METHODS:
        by_t = np.array([_at(np.asarray(x["wall"]), np.asarray(x["best"]), t_grid)
                         for x in runs[m]])
        by_e = np.array([_at(np.asarray(x["evals"]), np.asarray(x["best"]), e_grid)
                         for x in runs[m]])
        res["methods"][m] = {
            "wall": {"mean": np.nanmean(by_t, 0).tolist(),
                     "std": np.nanstd(by_t, 0).tolist(),
                     "success": np.nanmean(by_t >= SUCCESS, 0).tolist(),
                     "n_done": (~np.isnan(by_t)).sum(0).tolist()},
            "evals": {"mean": np.nanmean(by_e, 0).tolist(),
                      "std": np.nanstd(by_e, 0).tolist(),
                      "success": np.nanmean(by_e >= SUCCESS, 0).tolist(),
                      "n_done": (~np.isnan(by_e)).sum(0).tolist()},
            "final": [x["final"] for x in runs[m]],
            "total_wall": [x["total_wall"] for x in runs[m]],
            "n_eval": [x["n_eval"] for x in runs[m]],
            "frac_feasible_init": [x["frac_feasible_init"] for x in runs[m]],
            "runs": runs[m]}
        f = res["methods"][m]["final"]
        print("%-14s cuoi cung: trung vi %.3f  thanh cong %d/%d  "
              "%.1fs  %d goi  kha thi ban dau %.0f%%"
              % (m, float(np.median(f)), sum(v >= SUCCESS for v in f), n_seed,
                 float(np.median(res["methods"][m]["total_wall"])),
                 int(np.median(res["methods"][m]["n_eval"])),
                 100 * float(np.mean(res["methods"][m]["frac_feasible_init"]))),
              flush=True)

    p = os.path.join(ROOT, "results", out)
    json.dump(res, open(p, "w"), indent=1)
    print("saved ->", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seed", type=int, default=20)
    ap.add_argument("--out", default="pp1_anytime.json")
    a = ap.parse_args()
    main(a.n_seed, a.out)

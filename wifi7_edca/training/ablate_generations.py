"""training/ablate_generations.py -- vong lap tien hoa dong gop bao nhieu?

    py -3 -m training.ablate_generations

Cau hoi: sau khi GNN policy da de xuat quan the va GNN surrogate da xep hang,
CON LAI bao nhieu gia tri den tu phep tien hoa?

Do bang cach quet N_gen tu 0 tro len voi MOI THU KHAC giu nguyen:

    N_gen = 0  ->  chi co GNN: policy de xuat 80 cau hinh, surrogate xep hang,
                   lay cai tot nhat. Khong co chon loc / lai ghep / dot bien.
    N_gen > 0  ->  them dung N_gen the he tinh chinh.

Moi fitness deu do MO HINH GIAI TICH tinh. 5 seed moi diem.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from common.utility import evaluate_config, fitness
from config import GA, GAParams
from datagen.genome import GenomeSpec, decode
from datagen.scenario import main_scenario
from training.compare_seeding import policy_seeds
from training.ga_surrogate import run_ga_surrogate
from training.gnn import Surrogate
from training.train_policy import load_policy

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NL = 2
N_POP = 80
SEEDS = [2025, 2026, 2027, 2028, 2029]
GENS = [0, 1, 2, 3, 5, 8, 12, 20, 40, 80]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    acs = main_scenario()
    spec = GenomeSpec(n_ac=len(acs), n_links=NL, allow_link_choice=True)
    sur = Surrogate.load(os.path.join(ROOT, "results", "gnn_model.pt"), DEV)
    pol = load_policy(os.path.join(ROOT, "results", "policy.pt"), DEV)

    out = {"n_pop": N_POP, "seeds": SEEDS, "gens": GENS, "rows": []}
    print("N_gen = 0 means GNN only: policy proposes, surrogate ranks, no evolution.")
    print("%6s %10s %10s %10s %9s %9s" %
          ("N_gen", "fit_med", "fit_min", "fit_max", "wall_s", "success"))
    print("-" * 60)

    for g in GENS:
        fits, walls = [], []
        for s in SEEDS:
            t0 = time.perf_counter()
            seeds = policy_seeds(pol, sur, acs, NL, n_seed=N_POP, temperature=1.6)
            if g == 0:
                # policy_seeds returns them surrogate-ranked, best first; the
                # exact model then scores that single winner.
                res = evaluate_config(decode(np.asarray(seeds[0], int), spec), acs)
                f = fitness(res, acs)
            else:
                ga = GAParams(n_pop=N_POP, n_gen=g, n_elite=3,
                              p_cross=GA.p_cross, p_mutate=GA.p_mutate,
                              n_stag=10 ** 6, seed=s)
                r = run_ga_surrogate(acs, sur, n_links=NL, ga=ga, seed=s,
                                     verify_top=8, seed_genomes=seeds)
                f = float(r.fitness)
            walls.append(time.perf_counter() - t0)
            fits.append(f)
        thr = 49.207331
        row = {"n_gen": g, "fit": fits, "wall": walls,
               "fit_med": float(np.median(fits)),
               "wall_med": float(np.median(walls)),
               "success": int(sum(f >= thr for f in fits))}
        out["rows"].append(row)
        print("%6d %10.3f %10.3f %10.3f %9.2f %6d/%d" %
              (g, row["fit_med"], min(fits), max(fits), row["wall_med"],
               row["success"], len(SEEDS)), flush=True)

    p = os.path.join(ROOT, "results", "ablate_generations.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()

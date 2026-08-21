"""
training/compare_seeding.py -- Policy lam DIEM KHOI TAO cho GA (muc 9.9).

Muc 9.6 tang toc GA bang cach thay ham danh gia; muc 9.8 bo han GA nhung mat
1.94 diem vi bien an toan. Muc nay gop ca hai: policy sinh quan the ban dau
trong vai mili-giay, GA tinh chinh phan con lai.

Y tuong: `training.ga.structured_seeds` la mot bo QUY TAC THU CONG do nguoi viet
ra sau khi quet tham so ("CW_min lon, CW_max ~ CW_min, R = 7, AC nen day sang
AIFSN lon"). Policy da hoc chinh nhung quy luat do tu du lieu, va con dieu chinh
duoc theo tung kich ban cu the -- dieu ma mot bo quy tac tinh khong lam duoc.

Bon cau hinh duoc do, cung seed va cung ngan sach GA:

    1. GA goc                       -- mo hinh giai tich + structured_seeds
    2. GA + GNN surrogate           -- muc 9.6
    3. GA + GNN + gieo bang policy  -- muc 9.9 (day)
    4. Policy don thuan             -- muc 9.8

Chay:
    python -m training.compare_seeding --model results/gnn_model.pt \
        --policy results/policy.pt
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from common.utility import summarize
from config import GA, ACConfig, GAParams
from datagen.genome import describe
from datagen.scenario import main_scenario
from training.batch_graph import LevelTables
from training.ga import run_ga
from training.ga_surrogate import run_ga_surrogate
from training.gnn import Surrogate
from training.policy import (PolicyGNN, encode_scenarios, sample_levels)
from training.train_policy import load_policy, score_levels


@torch.no_grad()
def policy_seeds(policy: PolicyGNN, sur: Surrogate, ac_set: Sequence[ACConfig],
                 n_links: int, n_seed: int = 80, temperature: float = 1.6
                 ) -> List[np.ndarray]:
    """Sinh `n_seed` genome bang policy, xep hang bang surrogate, tot nhat truoc.

    Dung nhiet do > 1 de co DA DANG: sau khi entropy giam dan trong huan luyen,
    policy gan nhu tat dinh, ma mot quan the ban dau gom 80 ban sao giong het
    nhau thi khong con la quan the. Ca the greedy luon duoc giu o vi tri dau.
    """
    dev = sur.device
    scen, inp = encode_scenarios([ac_set], n_links, dev)
    logits = policy(inp)

    greedy = sample_levels(logits, k=1, greedy=True)                 # (1,1,N,6)
    sampled = sample_levels(logits, k=max(n_seed - 1, 1),
                            temperature=temperature)
    levels = torch.cat([greedy, sampled], dim=1)                     # (1,n,N,6)

    f = score_levels(levels, scen, LevelTables(dev), n_links, sur.model)[0]
    order = torch.argsort(f, descending=True).cpu().numpy()
    lv = levels[0].cpu().numpy()
    n_ac = len(ac_set)
    return [lv[int(i), :n_ac].ravel().astype(int) for i in order]


def compare(model_path: str, policy_path: str, n_links: int = 2,
            ga: GAParams = GA, verify_top: int = 8, seed: int = 2025,
            n_seed: int = 80, temperature: float = 1.6,
            device: Optional[str] = None, skip_base: bool = False):
    acs = main_scenario()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sur = Surrogate.load(model_path, dev)
    pol = load_policy(policy_path, dev)
    print(f"thiet bi: {dev}\n")

    rows = []

    if not skip_base:
        print("=== [1/3] GA goc (giai tich + structured_seeds) ===")
        t0 = time.time()
        r = run_ga(acs, n_links=n_links, ga=ga, seed=seed, verbose=True)
        rows.append(("GA goc", time.time() - t0, r, 0))

    print("\n=== [2/3] GA + GNN surrogate (structured_seeds) ===")
    t0 = time.time()
    r = run_ga_surrogate(acs, sur, n_links=n_links, ga=ga, seed=seed,
                         verify_top=verify_top, verbose=True)
    rows.append(("GA + GNN (structured_seeds)", time.time() - t0, r,
                 getattr(r, "n_surrogate", 0)))

    print("\n=== [3/3] GA + GNN + gieo bang POLICY ===")
    t0 = time.time()
    seeds = policy_seeds(pol, sur, acs, n_links, n_seed=n_seed,
                         temperature=temperature)
    t_seed = time.time() - t0
    print(f"    policy sinh {len(seeds)} genome trong {t_seed*1000:.1f} ms")
    r = run_ga_surrogate(acs, sur, n_links=n_links, ga=ga, seed=seed,
                         verify_top=verify_top, verbose=True,
                         seed_genomes=seeds)
    rows.append(("GA + GNN + gieo bang policy", time.time() - t0, r,
                 getattr(r, "n_surrogate", 0)))

    # --- bang tong hop ------------------------------------------------------
    print("\n" + "=" * 88)
    print("KET QUA -- kich ban Sec. V · moi fitness deu do MO HINH GIAI TICH tinh")
    print("=" * 88)
    print(f"{'phuong phap':32}{'thoi gian':>11}{'fitness':>10}{'kha thi':>9}"
          f"{'giai tich':>11}{'the he':>8}{'KT dau':>8}")
    print("-" * 88)
    for tag, dt, r, ns in rows:
        f0 = r.history.frac_feasible[0] if r.history.frac_feasible else 0.0
        print(f"{tag:32}{dt:9.1f} s{r.fitness:10.3f}"
              f"{str(r.qos.feasible(acs)):>9}{r.history.n_eval:11d}"
              f"{r.history.stopped_at:8d}{f0*100:7.0f}%")
    print(f"{'Policy don thuan (muc 9.8)':32}{0.0138:9.4f} s{47.760:10.3f}"
          f"{'True':>9}{1:11d}{0:>8}{'-':>8}")
    print("-" * 88)
    print("KT dau = ty le ca the KHA THI trong quan the BAN DAU")

    if len(rows) >= 2:
        base, best = rows[-2], rows[-1]
        print(f"\ngieo bang policy so voi structured_seeds: "
              f"fitness {base[2].fitness:.3f} -> {best[2].fitness:.3f}  ·  "
              f"thoi gian {base[1]:.1f}s -> {best[1]:.1f}s")

    print("\n--- Nghiem cua GA + GNN + gieo bang policy ---")
    print(describe(rows[-1][2].params))
    print(summarize(rows[-1][2].qos, acs))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Policy lam diem khoi tao cho GA")
    ap.add_argument("--model", type=str, default="results/gnn_model.pt")
    ap.add_argument("--policy", type=str, default="results/policy.pt")
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--pop", type=int, default=0)
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--verify-top", type=int, default=8)
    ap.add_argument("--n-seed", type=int, default=80)
    ap.add_argument("--temp", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--skip-base", action="store_true",
                    help="bo qua GA goc (dung so da do o muc 9.6)")
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()

    g = GA
    if a.pop or a.gen:
        g = GAParams(n_pop=a.pop or GA.n_pop, n_gen=a.gen or GA.n_gen,
                     n_elite=GA.n_elite, p_cross=GA.p_cross,
                     p_mutate=GA.p_mutate, n_stag=GA.n_stag, seed=GA.seed)
    compare(a.model, a.policy, n_links=a.links, ga=g, verify_top=a.verify_top,
            seed=a.seed, n_seed=a.n_seed, temperature=a.temp,
            device=a.device, skip_base=a.skip_base)

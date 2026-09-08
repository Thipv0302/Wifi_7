"""training/pp1_paper_figs.py -- so lieu cho 3 hinh ket qua cua Paper 1.

    py -3 -m training.pp1_paper_figs

Ba phep do, tuong ung ba hinh:

  E1  duong hoi tu cua MANG DE XUAT (proposal network) -- doc lai tu log huan
      luyen. GA la thuat toan heuristic, khong co "duong hoi tu theo epoch" so
      sanh duoc, nen no vao hinh duoi dang DUONG NGANG (trung vi va gia tri tot
      nhat cua 10 seed).

  E2  QoS tung AC cho ba cau hinh: Default EDCA, nghiem cua GA goc, nghiem cua
      pipeline. Cung cau truc Fig. 5 cua paper goc.

  E3  quet nguong eps_1: pipeline chay lai o moi gia tri, doi chieu voi ket qua
      GA da co trong results/fig6.json. Cung cau truc Fig. 6 cua paper goc.

Moi gia tri deu do `common.utility.evaluate_config` (mo hinh giai tich) tinh.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import torch

from common.utility import evaluate_config, fitness
from config import DEFAULT_EDCA, GA, GAParams
from datagen.genome import GenomeSpec, decode
from datagen.scenario import epsilon_sweep, main_scenario
from training.compare_seeding import policy_seeds
from training.ga_surrogate import run_ga_surrogate
from training.gnn import Surrogate
from training.train_policy import load_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NL = 2
POLICY = os.path.join(ROOT, "results", "var_1500_2027.pt")   # seeded, 5/5 length
N_POP, N_GEN = 160, 1                                        # depth from ablation


def run_pipeline(acs, n_links, sur, pol, seed):
    """Mot lan chay pipeline: de xuat -> xep hang -> 1 the he -> kiem chung."""
    t0 = time.perf_counter()
    seeds = policy_seeds(pol, sur, acs, n_links, n_seed=N_POP, temperature=1.6)
    ga = GAParams(n_pop=N_POP, n_gen=N_GEN, n_elite=3, p_cross=GA.p_cross,
                  p_mutate=GA.p_mutate, n_stag=10 ** 6, seed=seed)
    r = run_ga_surrogate(acs, sur, n_links=n_links, ga=ga, seed=seed,
                         verify_top=8, seed_genomes=seeds)
    return r, time.perf_counter() - t0


def main():
    acs = main_scenario()
    spec = GenomeSpec(n_ac=len(acs), n_links=NL, allow_link_choice=True)
    sur = Surrogate.load(os.path.join(ROOT, "results", "gnn_model.pt"), DEV)
    pol = load_policy(POLICY, DEV)
    # benchmark.json now lives with the manuscript it belongs to.
    bpath = os.path.join(ROOT, "results", "benchmark.json")
    if not os.path.exists(bpath):
        bpath = os.path.join(os.path.dirname(ROOT),
                             "Paper1_GNN_Accelerated_GA", "data",
                             "benchmark.json")
    B = json.load(open(bpath, encoding="utf-8"))
    i200 = B["pops"].index(200)
    # Ca hai duong tham chieu cua hinh hoi tu deu lay tu MOT ngan sach duy nhat
    # (N_pop = 200), tren dung 10 hat giong. Ban truoc lay `max` tren CA 70 lan
    # chay (7 kich thuoc quan the x 10 hat giong) nhung ghi nhan la "best of 10
    # seeds" -- vua sai nhan, vua dem best-of-70 dat canh mot lan chay duy nhat
    # cua mang. Duong so sanh dung la trung vi: mot lan chay doi mot lan chay.
    ga200 = sorted(r["fitness"] for r in B["sweep"]["ga"][i200]["runs"])
    out = {"ac": [a.name for a in acs],
           "eps": [a.epsilon for a in acs],
           "ga_runs": ga200,
           "ga_fit_med": float(np.median(ga200)),
           "ga_fit_min": float(ga200[0]),
           "ga_fit_max": float(ga200[-1])}

    # ================================================================== E1
    # Duong hoi tu: doc lai log huan luyen cua mang de xuat.
    log = os.path.join(ROOT, "results", "pp1_train_log.txt")
    if os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="replace").read()
        step, exact, feas = [], [], []
        for m in re.finditer(r"buoc\s+(\d+)/\d+.*?CHINH XAC\s+([-\d.]+)\s+"
                             r"kha thi\s+(\w+)", txt):
            step.append(int(m.group(1)))
            exact.append(float(m.group(2)))
            feas.append(m.group(3) == "True")
        out["e1"] = {"step": step, "exact": exact, "feasible": feas}
        print("E1: doc %d moc huan luyen" % len(step))
    else:
        print("E1: KHONG co %s -- chay lai train_policy voi log truoc" % log)

    # ================================================================== E2
    # QoS tung AC: Default EDCA / GA goc / pipeline.
    F = json.load(open(os.path.join(ROOT, "results", "fig45.json"),
                       encoding="utf-8"))
    e2 = {}
    for tag in ("Default EDCA", "Opt. MLO EDCA"):
        c = F["configs"][tag]
        e2[tag] = {"violation": c["violation"], "p_loss": c["p_loss"]}
    r, _ = run_pipeline(acs, NL, sur, pol, seed=2025)
    e2["GNN pipeline"] = {"violation": r.qos.violation.tolist(),
                          "p_loss": r.qos.p_loss.tolist()}
    print("E2: pipeline F = %.3f  kha thi %s"
          % (r.fitness, r.qos.feasible(acs)))
    out["e2"] = e2

    # ================================================================== E3
    # Quet eps_1, doi chieu voi ket qua GA da luu.
    G = json.load(open(os.path.join(ROOT, "results", "fig6.json"),
                       encoding="utf-8"))
    # Duong GA baseline: uu tien ban chay lai nhieu khoi dau
    # (training/pp1_eps_baseline.py). Ban cu trong fig6.json chi co MOT lan
    # chay moi diem, nen no tut o eps_1 = 1e-5 -- dieu khong the xay ra voi
    # nghiem toi uu, vi noi long eps_1 chi lam tap kha thi no ra.
    bpath2 = os.path.join(ROOT, "results", "pp1_eps_baseline.json")
    if os.path.exists(bpath2):
        Gb = json.load(open(bpath2, encoding="utf-8"))
        assert Gb["eps1"] == G["eps1"], "luoi eps_1 khong khop"
        ga_mlo = {"fitness": Gb["best"]["fitness"],
                  "sum_theta": Gb["best"]["sum_theta"],
                  "feasible": Gb["best"]["feasible"],
                  "fitness_median": Gb["median"]["fitness"],
                  "n_restart": Gb["n_restart"],
                  "wall": Gb["wall_per_point"]}
        print("E3: dung GA baseline best-of-%d tu pp1_eps_baseline.json"
              % Gb["n_restart"])
    else:
        ga_mlo = G["MLO EDCA"]
        print("E3: KHONG co pp1_eps_baseline.json -- dung fig6.json (1 lan chay)")
    e3 = {"eps1": G["eps1"], "target_sum_theta": G["target_sum_theta"],
          "ga_edca": G["EDCA"], "ga_mlo": ga_mlo,
          "pipeline": {"fitness": [], "sum_theta": [], "feasible": [],
                       "wall": []}}
    for pos, (e1, ac_set) in enumerate(epsilon_sweep(tuple(G["eps1"]))):
        r, w = run_pipeline(ac_set, NL, sur, pol, seed=2025 + pos)
        ok = bool(r.qos.feasible(ac_set))
        e3["pipeline"]["feasible"].append(ok)
        e3["pipeline"]["fitness"].append(float(r.qos.objective) if ok
                                         else float("nan"))
        e3["pipeline"]["sum_theta"].append(float(np.sum(r.qos.theta)) if ok
                                           else float("nan"))
        e3["pipeline"]["wall"].append(w)
        print("E3: eps_1 = %.0e  pipeline obj %.2f  kha thi %s  (%.1fs)"
              % (e1, r.qos.objective, ok, w), flush=True)
    out["e3"] = e3

    p = os.path.join(ROOT, "results", "pp1_paper_figs.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()

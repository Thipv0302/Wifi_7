"""training/train_variance.py -- phuong sai giua cac lan huan luyen policy.

    py -3 -m training.train_variance

Chay N lan huan luyen DOC LAP o moi do dai, roi danh gia CHINH XAC bang mo
hinh giai tich. Ly do phai do: `train_policy` chi seed numpy (`--seed` -> lich
lay kich ban), KHONG seed torch, nen khoi tao trong so va phep lay mau la ngau
nhien. Hai lan chay cung cau hinh da cho 48.081 (kha thi) va -0.604 (VI PHAM).

Mot lan chay khong noi len dieu gi -- day dung la sai lam da bat duoc truoc do
voi con so "tang toc 10.03x" cua mot seed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from common.utility import evaluate_config, fitness
from datagen.genome import GenomeSpec, decode
from datagen.scenario import main_scenario
from training.gnn import Surrogate
from training.policy import encode_scenarios, sample_levels
from training.train_policy import load_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NL = 2
STEPS = [1500, 3000, 6000]
# Moi lan chay mot SEED KHAC NHAU. Sau khi `train_policy` seed ca torch, cung
# mot seed cho ket qua giong het (da kiem: trong so trung khop tung bit), nen
# bang nay vua do duoc phuong sai vua tai lap duoc.
SEEDS = [2025, 2026, 2027, 2028, 2029]
N_RUN = len(SEEDS)


def evaluate(ckpt, acs, spec, sur):
    pol = load_policy(ckpt, DEV)
    _, inp = encode_scenarios([acs], NL, DEV)
    with torch.no_grad():
        lv = sample_levels(pol(inp), k=1, greedy=True)
    g = np.asarray(lv[0, 0].cpu().numpy()[:len(acs)]).reshape(-1).astype(int)
    res = evaluate_config(decode(g, spec), acs)
    return float(fitness(res, acs)), bool(res.feasible(acs)), float(res.c.mean())


def main():
    acs = main_scenario()
    spec = GenomeSpec(n_ac=len(acs), n_links=NL, allow_link_choice=True)
    sur = Surrogate.load(os.path.join(ROOT, "results", "gnn_model.pt"), DEV)
    out = {}
    for steps in STEPS:
        rows = []
        for seed in SEEDS:
            ckpt = os.path.join(ROOT, "results", "var_%d_%d.pt" % (steps, seed))
            t0 = time.time()
            subprocess.run([sys.executable, "-m", "training.train_policy",
                            "--surrogate", "results/gnn_model.pt",
                            "--steps", str(steps), "--seed", str(seed),
                            "--out", ckpt],
                           cwd=ROOT, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
            f, ok, cbar = evaluate(ckpt, acs, spec, sur)
            rows.append({"seed": seed, "fit": f, "feasible": ok,
                         "c_mean": cbar, "train_s": time.time() - t0})
            print("  steps=%5d seed %d: F = %8.3f  feasible %-5s  c_mean %.3f "
                  " (%.0fs)" % (steps, seed, f, ok, cbar, rows[-1]["train_s"]),
                  flush=True)
        fits = [x["fit"] for x in rows]
        ok = sum(x["feasible"] for x in rows)
        out[str(steps)] = {"runs": rows, "median": float(np.median(fits)),
                           "min": float(min(fits)), "max": float(max(fits)),
                           "n_feasible": ok, "n_run": N_RUN}
        print("steps=%5d | median %.3f | range [%.3f, %.3f] | feasible %d/%d\n"
              % (steps, np.median(fits), min(fits), max(fits), ok, N_RUN),
              flush=True)

    p = os.path.join(ROOT, "results", "train_variance.json")
    json.dump(out, open(p, "w"), indent=1)
    print("saved ->", p)


if __name__ == "__main__":
    main()

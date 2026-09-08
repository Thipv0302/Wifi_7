"""training/compare_policies.py -- doi chieu nhieu checkpoint policy.

    py -3 -m training.compare_policies results/policy.pt results/policy_12k.pt

Moi checkpoint duoc do BANG CUNG MOT quy trinh va CUNG MOT bo seed:

  * greedy decode  -> danh gia bang MO HINH GIAI TICH (khong phai surrogate)
  * best-of-512    -> surrogate xep hang, mo hinh giai tich cham ca the duoc chon
  * 2000 mau tho   -> ti le khai thi, khong xep hang
  * bien rang buoc tung AC cua nghiem greedy

Moi con so bao cao deu do `common.utility.evaluate_config` tinh. Surrogate chi
duoc dung de XEP HANG, khong bao gio de bao cao.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from common.utility import evaluate_config, fitness
from datagen.genome import GenomeSpec, decode
from datagen.scenario import main_scenario
from training.batch_graph import LevelTables
from training.gnn import Surrogate
from training.policy import encode_scenarios, sample_levels
from training.train_policy import load_policy, score_levels

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NL = 2
N_RAW = 2000
K_BEST = 512


def main(paths):
    acs = main_scenario()
    spec = GenomeSpec(n_ac=len(acs), n_links=NL, allow_link_choice=True)
    eps = np.array([a.epsilon for a in acs])
    sur = Surrogate.load("results/gnn_model.pt", DEV)
    tab = LevelTables(DEV)
    scen, inp = encode_scenarios([acs], NL, DEV)

    def exact(row):
        g = np.asarray(row[:len(acs)]).reshape(-1).astype(int)
        res = evaluate_config(decode(g, spec), acs)
        return fitness(res, acs), res.feasible(acs), res

    print("thiet bi %s | %d AC | %d link | moi checkpoint dung cung seed\n"
          % (DEV, len(acs), NL))
    hdr = ("%-26s %9s %9s %9s %11s %12s"
           % ("checkpoint", "greedy F", "best512", "raw feas", "worst Pr/eps", "min headroom"))
    print(hdr)
    print("-" * len(hdr))

    for p in paths:
        pol = load_policy(p, DEV)
        with torch.no_grad():
            logits = pol(inp)

            lv_g = sample_levels(logits, k=1, greedy=True)
            f_g, ok_g, res_g = exact(lv_g[0, 0].cpu().numpy())

            torch.manual_seed(1234)
            lv_k = sample_levels(logits, k=K_BEST, temperature=1.0)
            s = score_levels(lv_k, scen, tab, NL, sur.model)[0]
            f_k, ok_k, _ = exact(lv_k[0, int(torch.argmax(s))].cpu().numpy())

            torch.manual_seed(7)
            lv_r = sample_levels(logits, k=N_RAW, temperature=1.0)[0].cpu().numpy()
        feas = sum(exact(lv_r[i])[1] for i in range(N_RAW))

        ratio = res_g.violation / eps
        print("%-26s %9.3f %9.3f %8.2f%% %11.3f %11.1fx"
              % (p.split("/")[-1] + ("" if ok_g else " (INFEASIBLE)"),
                 f_g, f_k, 100.0 * feas / N_RAW, ratio.max(), 1.0 / ratio.max()))
        print("    per-AC Pr/eps : " + "  ".join("%s %.2e" % (a.name, r)
                                                 for a, r in zip(acs, ratio)))
        # c near 1 with P_loss near 1 is the degenerate corner: the constraint
        # holds only because almost nothing is delivered.
        print("    per-AC c      : " + "  ".join("%s %.4f" % (a.name, c)
                                                 for a, c in zip(acs, res_g.c)))
        cfg = decode(np.asarray(lv_g[0, 0].cpu().numpy()[:len(acs)])
                     .reshape(-1).astype(int), spec)
        print("    CWmin         : " + "  ".join(str(x.cw_min) for x in cfg))


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/policy.pt"])

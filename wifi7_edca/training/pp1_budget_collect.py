"""training/pp1_budget_collect.py -- gop ket qua quet ngan sach vao MOT file.

    py -3 -m training.pp1_budget_collect

Doc cac file roi do hai script sinh ra:

  pp1_eps_baseline*.json   GA thuan (training/pp1_eps_baseline.py)
  pp1_methods_*.json       ga_gnn + ga_gnn_policy (training/pp1_eps_methods.py)

va ghi `results/pp1_budget_methods.json` -- dinh dang phang ma make_figures.py
cua manuscript doc truc tiep. Tach ra lam hai buoc vi phan GA thuan chay mat
gan mot gio con hai nhanh GNN chi vai phut; gop lai o day de khong phai chay
lai phan dat khi them mot muc ngan sach.
"""
from __future__ import annotations

import json
import os

import numpy as np

from common.utility import evaluate_config
from datagen.genome import GenomeSpec, decode
from datagen.scenario import epsilon_sweep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SWEEP = epsilon_sweep((1e-8, 1e-7, 1e-6, 1e-5, 1e-4))

# (nhan, file GA thuan, file hai nhanh GNN)
BUDGETS = [
    ("16x9",    "pp1_eps_baseline_iso.json",     "pp1_methods_iso.json"),
    ("24x24",   "pp1_eps_baseline_24x24.json",   "pp1_methods_24x24.json"),
    ("40x40",   "pp1_eps_baseline_40x40.json",   "pp1_methods_40x40.json"),
    ("60x60",   "pp1_eps_baseline_60x60.json",   "pp1_methods_60x60.json"),
    ("80x80",   "pp1_eps_baseline_80x80.json",   "pp1_methods_80x80.json"),
    ("120x120", "pp1_eps_baseline.json",         "pp1_methods_120x120.json"),
    # Hai muc duoi chay 40 khoi dau moi diem thay vi 10, dung cho hinh quet
    # nguong: voi n = 10 sai so chuan cua trung binh la 1-3 diem muc tieu, du
    # de tao ra cac cho lom khong co that. Moi muc chi co MOT nhanh, nen phia
    # con lai de None.
    ("16x9r40",    "pp1_eps_baseline_iso_r40.json", None),
    ("40x40r40",   None,             "pp1_methods_40x40_r40.json"),
    ("120x120r40", None,             "pp1_methods_120x120_r40.json"),
    # Ba muc cua hinh quet nguong trong ban thao, 40 khoi dau moi diem.
    ("60x60r40",     "pp1_eps_baseline_60x60_r40.json", None),
    ("120x120gr40",  None,            "pp1_methods_120x120_gnn_r40.json"),
]
# GA thuan chay truoc khi script ghi lai n_eval, nen do rieng mot lan
# (training/pp1_eps_baseline.py --n-restart 1 tren mot diem eps bat ky).
GA_CALLS = {"16x9": 526, "24x24": 795, "40x40": 1622,
            "60x60": 3362, "80x80": 6060, "120x120": 12426,
            "16x9r40": 526, "60x60r40": 3362}


def _sum_theta(genome, ac_set, spec) -> float:
    """sum_i theta_i cua mot genome, tinh lai bang mo hinh giai tich."""
    q = evaluate_config(decode(np.asarray(genome), spec), ac_set)
    return float(np.sum(q.theta))


def main() -> None:
    spec = GenomeSpec(n_ac=len(SWEEP[0][1]), n_links=2, allow_link_choice=True)
    out = {"budgets": [b[0] for b in BUDGETS], "methods": {}}
    for tag, f_ga, f_gnn in BUDGETS:
        G = (json.load(open(os.path.join(RES, f_ga), encoding="utf-8"))
             if f_ga else None)
        D = (json.load(open(os.path.join(RES, f_gnn), encoding="utf-8"))
             if f_gnn else None)
        ref = G or D
        out.setdefault("eps1", ref["eps1"])
        out.setdefault("target_sum_theta", ref["target_sum_theta"])
        out.setdefault("n_restart", ref["n_restart"])
        n_rst = ref["n_restart"]

        src = {}
        if G is not None:
            src["ga"] = {"best": G["best"], "median": G["median"],
                         "wall_per_point": G["wall_per_point"],
                         "n_eval_per_run": [GA_CALLS[tag]] * len(G["eps1"]),
                         "points": G["points"]}
        if D is not None:
            src.update(D["methods"])

        for m, d in src.items():
            e = out["methods"].setdefault(
                m, {"budget": [], "calls": [], "wall_per_run": [],
                    "best_mean": [], "median_mean": [], "mean_mean": [],
                    "best": [], "median": [], "mean": [], "std": [],
                    "sum_theta": [], "sum_theta_mean": [],
                    "sum_theta_std": []})
            e["budget"].append(tag)
            e.setdefault("n_restart", []).append(n_rst)
            e["calls"].append(float(np.mean(d["n_eval_per_run"])))
            e["wall_per_run"].append(
                float(np.mean(d["wall_per_point"])) / n_rst)
            e["best"].append(d["best"]["fitness"])
            e["median"].append(d["median"]["fitness"])
            e["sum_theta"].append(d["best"].get("sum_theta"))
            e["best_mean"].append(float(np.mean(d["best"]["fitness"])))
            e["median_mean"].append(float(np.mean(d["median"]["fitness"])))

            # Mean and spread over the restarts at each threshold. `best` is the
            # solution the method can reach; the mean is what one run of it
            # returns, and the two answer different questions. Only the first is
            # bounded by the monotonicity of F* -- see pp1_eps_baseline.
            mu, sd, tmu, tsd = [], [], [], []
            for pos, pt in enumerate(d["points"]):
                obj = [r["objective"] for r in pt["runs"]]
                mu.append(float(np.nanmean(obj)))
                sd.append(float(np.nanstd(obj)))
                th = [r.get("sum_theta") for r in pt["runs"]]
                if any(t is None for t in th):
                    # pp1_eps_methods stores the genome but not sum_theta;
                    # recover it exactly, which costs one analytical call each.
                    th = [_sum_theta(r["genome"], SWEEP[pos][1], spec)
                          if r["feasible"] else np.nan for r in pt["runs"]]
                tmu.append(float(np.nanmean(th)))
                tsd.append(float(np.nanstd(th)))
            e["mean"].append(mu)
            e["std"].append(sd)
            e["sum_theta_mean"].append(tmu)
            e["sum_theta_std"].append(tsd)
            e["mean_mean"].append(float(np.mean(mu)))

    p = os.path.join(RES, "pp1_budget_methods.json")
    json.dump(out, open(p, "w"), indent=1)
    for m, e in out["methods"].items():
        print("%-14s %s" % (m, e["budget"]))
        print("%-14s calls %s" % ("", [round(c) for c in e["calls"]]))
        print("%-14s best  %s" % ("", [round(x, 2) for x in e["best_mean"]]))
        print("%-14s mean  %s" % ("", [round(x, 2) for x in e["mean_mean"]]))
    print("saved ->", p)


if __name__ == "__main__":
    main()

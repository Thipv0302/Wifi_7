"""export_data.py -- Xuat toan bo so lieu do duoc ra CSV (results/export/).

Cac file JSON trong results/ la dinh dang lam viec cua chuong trinh: long nhau
nhieu tang, mang dai, kho doc bang mat. Module nay trai chung ra dang BANG PHANG
(mot dong = mot phep do) de mo bang Excel / pandas / R ma khong phai viet code
boc tach.

    python -m export_data                 (chay tu thu muc wifi7_edca)

Nguon:
    results/benchmark.json   -- PP1: 210 lan chay GA (7 N_pop x 10 seed x 3 pp)
    results/pp2_policy.json  -- PP2: do policy (best-of-K, mau tho, bien rang buoc)
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "export")
METHODS = ["ga", "ga_gnn", "ga_gnn_policy"]


def _load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def _write(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("  %-22s %5d dong" % (name, len(rows)))
    return path


# ---------------------------------------------------------------------------
# PP1 -- GA / GA + GNN surrogate / GA + GNN + policy seeding
# ---------------------------------------------------------------------------


def export_pp1(B):
    pops, seeds, lab = B["pops"], B["seeds"], B["labels"]
    best = max(r["fitness"] for m in METHODS
               for e in B["sweep"][m] for r in e["runs"])
    thr = 0.99 * best

    rows = []
    for m in METHODS:
        for e in B["sweep"][m]:
            for k, r in enumerate(e["runs"]):
                rows.append([m, lab[m], e["n_pop"],
                             seeds[k] if k < len(seeds) else "",
                             "%.6f" % r["fitness"], int(r["feasible"]),
                             "%.4f" % r["wall"], "%.3f" % r["seed_ms"],
                             r["n_eval"], r["n_surrogate"], r["stopped_at"],
                             "%.4f" % r["frac_feasible_init"],
                             int(r["fitness"] >= thr)])
    _write("pp1_runs.csv",
           ["method", "method_label", "n_pop", "seed", "fitness", "feasible",
            "wall_s", "seed_ms", "n_exact_eval", "n_surrogate_eval",
            "stopped_at_gen", "frac_feasible_init", "success"], rows)

    rows = []
    for m in METHODS:
        for e in B["sweep"][m]:
            f = np.array([r["fitness"] for r in e["runs"]])
            ok = int((f >= thr).sum())
            rows.append([m, lab[m], e["n_pop"], len(f),
                         "%.6f" % e["fit_med"], "%.6f" % e["fit_min"],
                         "%.6f" % e["fit_max"], "%.4f" % e["wall_med"],
                         e["eval_med"], ok, "%.1f" % (100.0 * ok / len(f)),
                         "%.4f" % np.mean([r["frac_feasible_init"]
                                           for r in e["runs"]])])
    _write("pp1_summary.csv",
           ["method", "method_label", "n_pop", "n_runs", "fitness_median",
            "fitness_min", "fitness_max", "wall_s_median",
            "n_exact_eval_median", "n_success", "success_rate_pct",
            "frac_feasible_init_mean"], rows)

    rows = []
    for m in METHODS:
        a = B["anytime"][m]
        for i, (w, ev, bf) in enumerate(zip(a["curve_wall"], a["curve_evals"],
                                            a["curve_best"])):
            rows.append([m, lab[m], i, "%.6f" % w, ev, "%.6f" % bf])
    _write("pp1_anytime.csv",
           ["method", "method_label", "point_index", "wall_s",
            "n_exact_eval", "best_fitness"], rows)

    i200 = pops.index(200)
    rows = []
    for m in METHODS:
        e = B["sweep"][m][i200]
        f = np.array([r["fitness"] for r in e["runs"]])
        ok = int((f >= thr).sum())
        rows.append([m, lab[m], "%.4f" % e["wall_med"], e["eval_med"],
                     "%.6f" % e["fit_med"], "%d/%d" % (ok, len(f)),
                     "%.2f" % (B["sweep"]["ga"][i200]["wall_med"] / e["wall_med"])])
    p = B["policy"]
    rows.append(["policy", B["labels"]["policy"], "%.6f" % p["wall"],
                 p["n_eval"], "%.6f" % p["fitness"], "",
                 "%.1f" % (B["sweep"]["ga"][i200]["wall_med"] / p["wall"])])
    _write("pp1_headline_npop200.csv",
           ["method", "method_label", "wall_s_median", "n_exact_eval_median",
            "fitness_median", "success", "speedup_vs_ga"], rows)
    return thr, best


# ---------------------------------------------------------------------------
# PP2 -- GNN policy
# ---------------------------------------------------------------------------


def export_pp2(P):
    e1 = P["e1"]
    rows = []
    for src in ("policy", "random"):
        for K in e1["ks"]:
            d = e1[src][str(K)]
            for rep, (f, ok, w) in enumerate(zip(d["fit"], d["feas"], d["wall"])):
                rows.append([src, K, rep, "%.6f" % f, int(ok), "%.6f" % w])
    g = e1["greedy"]
    rows.append(["policy_greedy", 1, 0, "%.6f" % g["fit"], int(g["feas"]),
                 "%.6f" % g["wall"]])
    _write("pp2_best_of_k.csv",
           ["source", "k_samples", "rep", "fitness", "feasible", "wall_s"], rows)

    rows = []
    for src, d in P["e2"].items():
        for i, (f, ok) in enumerate(zip(d["fit"], d["feas"])):
            rows.append([src, i, "%.6f" % f, int(ok)])
    _write("pp2_raw_samples.csv",
           ["source", "index", "fitness", "feasible"], rows)

    eps = np.array(P["eps"])
    rows = []
    for tag, d in P["e3"].items():
        viol = np.array(d["viol"])
        ploss = np.array(d["p_loss"])
        contrib = -np.log10(np.maximum(ploss, 1e-10))
        for i, ac in enumerate(P["ac"]):
            rows.append([tag, "%.6f" % d["fit"], ac, "%.3e" % eps[i],
                         "%.6e" % viol[i], "%.6f" % (viol[i] / eps[i]),
                         "%.6e" % ploss[i], "%.6f" % contrib[i],
                         int(viol[i] < eps[i])])
    _write("pp2_constraint_margin.csv",
           ["solution", "total_fitness", "ac", "epsilon", "violation_prob",
            "violation_over_epsilon", "p_loss", "fitness_contribution",
            "constraint_met"], rows)

    e4 = P["e4"]
    rows = [[s, "%.4f" % el, "%.6f" % ex, int(ok)] for s, el, ex, ok
            in zip(e4["step"], e4["elite"], e4["exact"], e4["feas"])]
    _write("pp2_training_curve.csv",
           ["step", "elite_surrogate_score", "exact_fitness", "feasible"], rows)


# ---------------------------------------------------------------------------
# Cac phep do bo sung cho ban thao (quet nguong, anytime, QoS, huan luyen)
# ---------------------------------------------------------------------------


def export_sweep(BM):
    """Quet eps_1 x ngan sach x phuong phap -- mot dong moi (pp, ngan sach, eps)."""
    rows = []
    for m, e in BM["methods"].items():
        for i, b in enumerate(e["budget"]):
            for j, eps in enumerate(BM["eps1"]):
                rows.append([m, b, e["n_restart"][i], round(e["calls"][i]),
                             round(e["wall_per_run"][i], 3), eps,
                             round(BM["target_sum_theta"][j], 4),
                             round(e["mean"][i][j], 4), round(e["std"][i][j], 4),
                             round(e["median"][i][j], 4),
                             round(e["best"][i][j], 4),
                             round(e["sum_theta_mean"][i][j], 4)])
    _write("pp1_eps_sweep.csv",
           ["method", "budget", "n_restart", "exact_calls", "wall_s_per_run",
            "eps_1", "target_sum_theta", "fitness_mean", "fitness_std",
            "fitness_median", "fitness_best_of_n", "sum_theta_mean"], rows)



def export_anytime(A):
    """Muc tieu dat duoc theo ngan sach chung -- ca hai truc."""
    rows = []
    for axis, gk in (("wall_s", "t_grid"), ("exact_calls", "e_grid")):
        for m, d in A["methods"].items():
            a = d["wall" if axis == "wall_s" else "evals"]
            for i, g in enumerate(A[gk]):
                mu = a["mean"][i]
                if mu != mu:            # NaN: chua lan chay nao toi moc nay
                    continue
                rows.append([axis, m, round(g, 4), round(mu, 4),
                             round(a["std"][i], 4), round(a["success"][i], 4),
                             a["n_done"][i]])
    _write("pp1_anytime_budget.csv",
           ["axis", "method", "budget", "fitness_mean", "fitness_std",
            "success_rate", "n_runs_reporting"], rows)
    print("      (n = %d seed doc lap)" % A["n_seed"])

    rows = [[m, r["seed"], round(r["final"], 4), round(r["total_wall"], 3),
             r["n_eval"], round(r["frac_feasible_init"], 4)]
            for m, d in A["methods"].items() for r in d["runs"]]
    _write("pp1_anytime_runs.csv",
           ["method", "seed", "final_fitness", "wall_s", "exact_calls",
            "frac_feasible_init"], rows)



def export_qos(Q):
    """QoS tung AC cho bon cau hinh."""
    rows = []
    for tag, c in Q["configs"].items():
        for i, ac in enumerate(Q["ac"]):
            rows.append([tag, ac, Q["eps"][i], "%.6e" % c["violation"][i],
                         "%.6e" % c["p_loss"][i],
                         round(c["objective"], 4),
                         c["violation"][i] < Q["eps"][i]])
    _write("pp1_qos_per_ac.csv",
           ["config", "ac", "eps", "violation", "p_loss", "objective",
            "feasible"], rows)



def export_train_curve(T):
    """Duong hoi tu cua mang danh gia, tren thang muc tieu."""
    rows = [[r["epoch"], round(r["elapsed"], 2), round(r["exact"], 4),
             round(r["exact_best"], 4), round(r["rmse_fit"], 4),
             round(r["rho_fit"], 6), round(r["feas_acc"], 6)]
            for r in T["curve"]]
    _write("gnn_eval_training.csv",
           ["epoch", "elapsed_s", "topv_mean_exact_F", "topv_best_exact_F",
            "rmse_fitness", "spearman_rho", "feasibility_accuracy"], rows)
    print("      (tran cua mot thu tu hoan hao: %.3f)" % T["pool"]["oracle_topv"])


def main():
    os.makedirs(OUT, exist_ok=True)
    B, P = _load("benchmark.json"), _load("pp2_policy.json")
    print("PP1 (results/benchmark.json):")
    thr, best = export_pp1(B)
    print("PP2 (results/pp2_policy.json):")
    export_pp2(P)

    # Cac phep do bo sung. Bo qua cai nao chua chay thay vi hong ca lan xuat:
    # khong phai ban sao nao cua repo cung co du moi file ket qua.
    for name, fn in (("pp1_budget_methods.json", export_sweep),
                     ("pp1_anytime.json", export_anytime),
                     ("pp1_qos_methods.json", export_qos),
                     ("gnn_train_curve.json", export_train_curve)):
        if not os.path.exists(os.path.join(RES, name)):
            print("  (bo qua %s -- chua co)" % name)
            continue
        print("%s:" % name)
        fn(_load(name))

    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Du lieu do duoc, xuat tu results/*.json bang `python -m export_data`.\n"
            "Moi gia tri fitness deu do MO HINH GIAI TICH Eqs.(1)-(18) tinh,\n"
            "khong bao gio do surrogate tinh.\n\n"
            "PP1 -- GA / GA + GNN surrogate / GA + GNN + policy seeding\n"
            "  pp1_runs.csv               210 lan chay: 3 pp x 7 N_pop x 10 seed\n"
            "  pp1_summary.csv            gop theo (pp, N_pop)\n"
            "  pp1_anytime.csv            duong fitness theo thoi gian / so lan danh gia\n"
            "  pp1_headline_npop200.csv   bang chinh o N_pop = 200\n\n"
            "Quet nguong / anytime / QoS (ban thao Paper 1)\n"
            "  pp1_eps_sweep.csv          eps_1 x ngan sach x phuong phap\n"
            "  pp1_anytime_budget.csv     muc tieu theo ngan sach chung, 2 truc\n"
            "  pp1_anytime_runs.csv       tung lan chay cua thi nghiem anytime\n"
            "  pp1_qos_per_ac.csv         QoS tung AC, bon cau hinh\n"
            "  gnn_eval_training.csv      hoi tu mang danh gia tren thang muc tieu\n\n"
            "PP2 -- GNN policy\n"
            "  pp2_best_of_k.csv          chat luong theo so mau K (policy vs ngau nhien)\n"
            "  pp2_raw_samples.csv        6000 mau tho da danh gia chinh xac\n"
            "  pp2_constraint_margin.csv  bien rang buoc tung AC\n"
            "  pp2_training_curve.csv     16 moc trong 3000 buoc huan luyen\n\n"
            "Nguong 'success' = 1%% cua gia tri tot nhat quan sat duoc tren toan bo\n"
            "phep do: fitness >= %.6f (gia tri tot nhat = %.6f).\n" % (thr, best))
    print("\nDa xuat -> %s" % OUT)


if __name__ == "__main__":
    main()

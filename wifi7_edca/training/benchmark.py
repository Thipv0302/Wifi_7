"""
training/benchmark.py -- Do doi chieu CONG BANG giua GA goc va cac bien the GNN.

Vi sao can module nay: bang so sanh o muc 9.6/9.9 cho ca hai phuong phap CUNG
mot cau hinh GA (N_pop = 200, N_gen = 300) roi bao cao thoi gian. Cach trinh bay
do vo tinh LAM MO loi the cua surrogate, vi truc hoanh cua duong hoi tu la THE HE
-- ma mot the he cua GA goc dat gap ~19 lan mot the he cua GA + GNN. Ve theo the
he thi hai duong gan nhu trung nhau; ve theo THOI GIAN THUC hoac theo SO LAN GOI
MO HINH GIAI TICH thi khoang cach that moi hien ra.

Khong ha ngan sach cua GA goc de no thua. Nguoc lai: cho MOI phuong phap chay
tren CUNG MOT DAI ngan sach, roi hoi "voi ngan sach t giay, phuong phap nao dat
fitness cao hon?". Do la phep so sanh anytime tieu chuan, va no chiu duoc phan
bien vi khong co tham so nao duoc chon rieng cho tung phuong phap.

Hai phep do:

  A. DUONG ANYTIME  -- fitness tot nhat DA KIEM CHUNG theo thoi gian thuc / theo
     so lan goi mo hinh giai tich, cung mot cau hinh GA cho ca bon phuong phap.

  B. QUET NGAN SACH -- N_pop chay tren mot luoi, nhieu seed doc lap. Tra loi
     "de dat fitness F thi moi phuong phap can bao nhieu giay / bao nhieu lan
     danh gia?" -- day la hinh danh cho paper.

Chay:
    python -m training.benchmark --model results/gnn_model.pt \
        --policy results/policy.pt --seeds 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from config import GA, GAParams, RESULT_DIR
from datagen.scenario import main_scenario
from training.batch_graph import LevelTables
from training.compare_seeding import policy_seeds
from training.ga import run_ga
from training.ga_surrogate import run_ga_surrogate
from training.gnn import Surrogate
from training.policy import encode_scenarios, sample_levels
from training.train_policy import load_policy, score_levels

# Nhan hien thi + thu tu ve. Khoa dung xuyen suot benchmark va phan ve hinh.
METHODS = ["ga", "ga_gnn", "ga_gnn_policy", "policy"]
LABELS = {
    "ga": "GA (analytical model only)",
    "ga_gnn": "GA + GNN surrogate",
    "ga_gnn_policy": "GA + GNN + policy seeding",
    "policy": "Policy only (no GA)",
}


def _budget(n_pop: int) -> GAParams:
    """Cung mot cau hinh GA cho moi phuong phap, chi doi N_pop."""
    return GAParams(n_pop=n_pop, n_gen=GA.n_gen, n_elite=max(n_pop // 25, 2),
                    p_cross=GA.p_cross, p_mutate=GA.p_mutate,
                    n_stag=GA.n_stag, seed=GA.seed)


def _run_one(method: str, acs, n_links: int, ga: GAParams, seed: int,
             sur: Surrogate, pol, verify_top: int, n_seed: int,
             temperature: float) -> Dict:
    """Chay mot phuong phap mot lan, tra ve lich su + ket qua cuoi."""
    t0 = time.perf_counter()

    if method == "ga":
        r = run_ga(acs, n_links=n_links, ga=ga, seed=seed)
        seed_ms = 0.0
    elif method == "ga_gnn":
        r = run_ga_surrogate(acs, sur, n_links=n_links, ga=ga, seed=seed,
                             verify_top=verify_top)
        seed_ms = 0.0
    elif method == "ga_gnn_policy":
        ts = time.perf_counter()
        seeds = policy_seeds(pol, sur, acs, n_links, n_seed=n_seed,
                             temperature=temperature)
        seed_ms = (time.perf_counter() - ts) * 1e3
        r = run_ga_surrogate(acs, sur, n_links=n_links, ga=ga, seed=seed,
                             verify_top=verify_top, seed_genomes=seeds)
    else:
        raise ValueError(method)

    wall = time.perf_counter() - t0
    h = r.history
    # `wall` trong history do tu luc vao run_*; cong them phan gieo bang policy
    # de truc thoi gian cua moi phuong phap deu tinh tu cung mot moc.
    off = seed_ms / 1e3
    return {
        "fitness": float(r.fitness),
        "feasible": bool(r.qos.feasible(acs)),
        "wall": float(wall),
        "seed_ms": float(seed_ms),
        "n_eval": int(h.n_eval),
        "n_surrogate": int(getattr(r, "n_surrogate", 0)),
        "stopped_at": int(h.stopped_at),
        "frac_feasible_init": float(h.frac_feasible[0] if h.frac_feasible else 0.0),
        "curve_wall": [float(t) + off for t in h.wall],
        "curve_evals": [int(e) for e in h.evals],
        "curve_best": [float(b) for b in h.best],
    }


@torch.no_grad()
def _run_policy_only(pol, sur, acs, n_links: int, n_sample: int = 1) -> Dict:
    """Policy don thuan: mot forward pass, khong co vong lap tien hoa."""
    from common.utility import evaluate_config, fitness
    from datagen.genome import GenomeSpec, decode

    dev = sur.device
    t0 = time.perf_counter()
    scen, inp = encode_scenarios([acs], n_links, dev)
    logits = pol(inp)
    levels = sample_levels(logits, k=n_sample, greedy=n_sample == 1)
    if n_sample > 1:
        f = score_levels(levels, scen, LevelTables(dev), n_links, sur.model)[0]
        pick = int(torch.argmax(f))
    else:
        pick = 0
    g = levels[0, pick, :len(acs)].reshape(-1).cpu().numpy().astype(int)
    if dev != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    spec = GenomeSpec(n_ac=len(acs), n_links=n_links,
                      allow_link_choice=n_links > 1)
    res = evaluate_config(decode(g, spec), acs)
    return {"fitness": float(fitness(res, acs)),
            "feasible": bool(res.feasible(acs)),
            "wall": float(wall), "n_eval": 1, "n_surrogate": n_sample}


def run(model_path: str, policy_path: str, n_links: int = 2,
        pops: Optional[List[int]] = None, seeds: int = 3,
        verify_top: int = 8, n_seed: int = 80, temperature: float = 1.6,
        device: Optional[str] = None, out: str = "benchmark.json") -> Dict:
    acs = main_scenario()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sur = Surrogate.load(model_path, dev)
    pol = load_policy(policy_path, dev)
    pops = pops or [25, 50, 100, 200]
    seed_list = [GA.seed + i for i in range(seeds)]

    print(f"thiet bi: {dev}  ·  {n_links} link  ·  N_pop {pops}  ·  "
          f"{seeds} seed  ·  N_gen = {GA.n_gen}, N_stag = {GA.n_stag}\n",
          flush=True)

    data: Dict = {"n_links": n_links, "pops": pops, "seeds": seed_list,
                  "n_gen": GA.n_gen, "n_stag": GA.n_stag,
                  "labels": LABELS, "sweep": {}, "anytime": {}, "policy": {}}

    # --- A. duong anytime: mot lan chay day du cho moi phuong phap ----------
    print("=== [A] duong anytime (N_pop = %d, seed = %d) ===" % (pops[-1], seed_list[0]))
    ga_full = _budget(pops[-1])
    for m in METHODS[:3]:
        r = _run_one(m, acs, n_links, ga_full, seed_list[0], sur, pol,
                     verify_top, n_seed, temperature)
        data["anytime"][m] = r
        print(f"    {LABELS[m]:34} {r['wall']:7.1f} s  "
              f"fitness {r['fitness']:7.3f}  giai tich {r['n_eval']:6d}",
              flush=True)

    p = _run_policy_only(pol, sur, acs, n_links)
    data["policy"] = p
    print(f"    {LABELS['policy']:34} {p['wall']*1e3:6.1f} ms  "
          f"fitness {p['fitness']:7.3f}  giai tich {p['n_eval']:6d}", flush=True)

    # --- B. quet ngan sach --------------------------------------------------
    # Luu SAU MOI muc ngan sach, khong doi den cuoi. Mot lan quet day du mat
    # hang tieng; neu chi ghi mot lan o cuoi thi bat ky lan dung nao -- het gio,
    # loi, nguoi dung huy -- deu xoa sach toan bo cong da bo ra.
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, out)

    def flush_json() -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)

    print("\n=== [B] quet ngan sach ===", flush=True)
    for m in METHODS[:3]:
        data["sweep"][m] = []
        for n_pop in pops:
            ga = _budget(n_pop)
            runs = [_run_one(m, acs, n_links, ga, s, sur, pol, verify_top,
                             n_seed, temperature) for s in seed_list]
            for r in runs:                    # duong cong khong can luu o day
                r.pop("curve_wall"), r.pop("curve_evals"), r.pop("curve_best")
            fit = [r["fitness"] for r in runs]
            wall = [r["wall"] for r in runs]
            ev = [r["n_eval"] for r in runs]
            data["sweep"][m].append({"n_pop": n_pop, "runs": runs,
                                     "fit_med": float(np.median(fit)),
                                     "fit_min": float(np.min(fit)),
                                     "fit_max": float(np.max(fit)),
                                     "wall_med": float(np.median(wall)),
                                     "eval_med": float(np.median(ev))})
            n_ok = sum(f >= 0.99 * max(fit) for f in fit)
            print(f"    {LABELS[m]:34} N_pop {n_pop:4d}  "
                  f"{np.median(wall):7.1f} s  "
                  f"fitness {np.median(fit):7.3f} "
                  f"[{np.min(fit):.3f}, {np.max(fit):.3f}]  "
                  f"giai tich {np.median(ev):7.0f}  "
                  f"dat {n_ok}/{len(fit)}", flush=True)
            flush_json()

    flush_json()
    print(f"\nDa luu -> {path}")
    return data


def merge(*paths: str, out: str = "benchmark.json") -> Dict:
    """Ghep nhieu lan quet ngan sach thanh mot tap du lieu.

    Quet dai N_pop rong ton hang chuc phut, ma thuong ta chi muon NOI THEM vai
    muc ngan sach chu khong chay lai tu dau. Phan `sweep` duoc gop va sap theo
    N_pop; phan `anytime`/`policy` lay tu file DAU TIEN -- do la lan chay o ngan
    sach lon nhat, tuc duong anytime co y nghia doi chieu.
    """
    merged: Optional[Dict] = None
    by_pop: Dict[str, Dict[int, Dict]] = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        if merged is None:
            merged = dict(d)
        for m, rows in d["sweep"].items():
            for r in rows:
                by_pop.setdefault(m, {})[int(r["n_pop"])] = r

    assert merged is not None
    merged["sweep"] = {m: [v[k] for k in sorted(v)] for m, v in by_pop.items()}
    merged["pops"] = sorted({int(r["n_pop"])
                             for rows in merged["sweep"].values() for r in rows})
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, out)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=1)
    print(f"da ghep {len(paths)} file · N_pop = {merged['pops']} -> {path}")
    return merged


def latex_table(d: Dict) -> str:
    """Bang so sanh o dinh dang LaTeX (booktabs), dan thang vao paper.

    Chi bao cao ngan sach LON NHAT cua moi phuong phap -- do la cau hinh ma ca
    ba deu dat fitness bao hoa, nen so sanh chi phi tai do la so sanh "cung chat
    luong nghiem", khong phai "cung tham so".
    """
    lab = d["labels"]
    top = d["pops"][-1]
    n_seed = len(d["seeds"])
    ref = None

    # Nguong "thanh cong" lay chung cho moi phuong phap: 1 % tuong doi so voi
    # gia tri tot nhat quan sat duoc tren TOAN BO phep do. Neu moi phuong phap
    # tu lay gia tri tot nhat cua rieng no lam chuan thi cot nay vo nghia.
    best = max(run["fitness"] for rows in d["sweep"].values()
               for r in rows for run in r["runs"])
    thr = 0.99 * best

    out = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Cost and reliability of reaching the optimised MLO EDCA "
        r"configuration (Sec.~V scenario, $M=2$ links, $N_{pop}=%d$, "
        r"$N_{gen}=%d$, $N_{stag}=50$, %d independent seeds). \emph{Success} "
        r"counts runs whose final fitness is within $1\%%$ of the best value "
        r"observed across all methods ($\geq %.2f$); fitness is the median "
        r"over seeds. All fitness values are computed by the analytical model "
        r"of Eqs.~(1)--(18), never by the surrogate.}"
        % (top, d["n_gen"], n_seed, thr),
        r"\label{tab:gnn_speedup}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & Time (s) & Exact evals & Fitness & Success & Speed-up \\",
        r"\midrule",
    ]
    for m in METHODS[:3]:
        rows = [r for r in (d["sweep"].get(m) or []) if r["n_pop"] == top]
        if not rows:
            continue
        r = rows[0]
        if ref is None:
            ref = r["wall_med"]
        ok = sum(run["fitness"] >= thr for run in r["runs"])
        out.append(r"%s & %.1f & %d & %.3f & %d/%d & %s \\"
                   % (lab[m], r["wall_med"], int(r["eval_med"]), r["fit_med"],
                      ok, len(r["runs"]),
                      r"---" if r["wall_med"] == ref
                      else r"$%.1f\times$" % (ref / r["wall_med"])))
    p = d.get("policy") or {}
    if p and ref:
        out.append(r"%s & %.4f & %d & %.3f & %s & $%.0f\times$ \\"
                   % (lab["policy"], p["wall"], p["n_eval"], p["fitness"],
                      r"---" if p["fitness"] < thr else r"1/1",
                      ref / p["wall"]))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Do doi chieu cong bang GA vs GNN")
    ap.add_argument("--model", type=str, default="results/gnn_model.pt")
    ap.add_argument("--policy", type=str, default="results/policy.pt")
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--pops", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--verify-top", type=int, default=8)
    ap.add_argument("--n-seed", type=int, default=80)
    ap.add_argument("--temp", type=float, default=1.6)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default="benchmark.json")
    ap.add_argument("--merge", type=str, nargs="+", default=None,
                    help="ghep cac file json da do san, khong chay lai phep do")
    a = ap.parse_args()

    if a.merge:
        data = merge(*a.merge, out=a.out)
        tex = latex_table(data)
        path = os.path.join(RESULT_DIR, os.path.splitext(a.out)[0] + "_table.tex")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(tex + "\n")
        print("\n" + tex + f"\n\nDa luu bang -> {path}")
        raise SystemExit(0)

    data = run(a.model, a.policy, n_links=a.links, pops=a.pops, seeds=a.seeds,
               verify_top=a.verify_top, n_seed=a.n_seed, temperature=a.temp,
               device=a.device, out=a.out)
    tex = latex_table(data)
    path = os.path.join(RESULT_DIR, os.path.splitext(a.out)[0] + "_table.tex")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print("\n" + tex + f"\n\nDa luu bang -> {path}")

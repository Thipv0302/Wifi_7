"""Do lai PHUONG PHAP 2 (GNN policy) de co du lieu ve hinh -- tuong tu PP1.

    py -3 -m training.pp2_measure [POLICY_CKPT] [OUT_JSON] [TRAIN_LOG]
"""
import sys
import json, os, time
import numpy as np, torch

from common.utility import evaluate_config, fitness
from config import GA, GAParams
from datagen.genome import GenomeSpec, decode
from datagen.scenario import main_scenario
from training.batch_graph import LevelTables
from training.gnn import Surrogate
from training.policy import encode_scenarios, sample_levels, MAX_AC
from training.train_policy import load_policy, score_levels
from training.compare_seeding import policy_seeds
from training.ga_surrogate import run_ga_surrogate

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NL = 2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
acs = main_scenario()
sur = Surrogate.load(os.path.join(ROOT, "results", "gnn_model.pt"), DEV)
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "policy.pt")
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "pp2_policy.json")
TRAIN_LOG = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "policy_log.txt")
print("checkpoint:", CKPT, flush=True)
pol = load_policy(CKPT, DEV)
tab = LevelTables(DEV)
spec = GenomeSpec(n_ac=len(acs), n_links=NL, allow_link_choice=True)
upper = spec.upper().reshape(len(acs), 6)
eps = np.array([a.epsilon for a in acs])
print(f"thiet bi {DEV} · {len(acs)} AC · {NL} link", flush=True)

scen, inp = encode_scenarios([acs], NL, DEV)
with torch.no_grad():
    logits = pol(inp)

def exact(lv_row):
    """lv_row: (MAX_AC,6) muc -> (fitness, kha thi, violation, p_loss)."""
    g = np.asarray(lv_row[:len(acs)]).reshape(-1).astype(int)
    res = evaluate_config(decode(g, spec), acs)
    return fitness(res, acs), res.feasible(acs), res.violation.copy(), res.p_loss.copy(), g

def rand_levels(k, rng):
    lv = np.zeros((k, MAX_AC, 6), dtype=np.int64)
    for i in range(len(acs)):
        for j in range(6):
            lv[:, i, j] = rng.integers(0, upper[i, j], size=k)
    return torch.as_tensor(lv, device=DEV).unsqueeze(0)   # (1,k,N,6)

def pick_best(levels):
    """levels (1,k,N,6) -> (chi so tot nhat theo surrogate, thoi gian GPU)."""
    with torch.no_grad():
        f = score_levels(levels, scen, tab, NL, sur.model)[0]
        i = int(torch.argmax(f))
    return i

OUT = {"eps": eps.tolist(), "ac": [a.name for a in acs]}

# ===================================================================== E1
# Chat luong theo SO MAU K -- "duong anytime" cua PP2, o thang mili-giay.
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
REPS = 7
e1 = {"ks": KS, "policy": {}, "random": {}}
for src in ("policy", "random"):
    for K in KS:
        fits, feas, walls = [], [], []
        for r in range(REPS):
            torch.manual_seed(1000 + r); rng = np.random.default_rng(1000 + r)
            if DEV != "cpu": torch.cuda.synchronize()
            t0 = time.perf_counter()
            if src == "policy":
                with torch.no_grad():
                    lg = pol(inp)
                    lv = sample_levels(lg, k=K, temperature=1.0)
            else:
                lv = rand_levels(K, rng)
            i = pick_best(lv)
            if DEV != "cpu": torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
            f, ok, *_ = exact(lv[0, i].cpu().numpy())
            fits.append(f); feas.append(bool(ok))
        e1[src][str(K)] = {"fit": fits, "feas": feas, "wall": walls}
        print(f"  E1 {src:6s} K={K:5d}  fit_med={np.median(fits):8.3f} "
              f"feas={sum(feas)}/{REPS}  {np.median(walls)*1e3:7.2f} ms", flush=True)

# diem tham chieu: policy greedy (1 mau tat dinh)
gw = []
for r in range(REPS):
    if DEV != "cpu": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        lg = pol(inp)
        lv = sample_levels(lg, k=1, greedy=True)
    if DEV != "cpu": torch.cuda.synchronize()
    gw.append(time.perf_counter() - t0)
gf, gok, gviol, gploss, ggen = exact(lv[0, 0].cpu().numpy())
e1["greedy"] = {"fit": gf, "feas": bool(gok), "wall": float(np.median(gw))}
print(f"  E1 greedy       fit={gf:.3f} feas={gok} {np.median(gw)*1e3:.2f} ms", flush=True)
OUT["e1"] = e1

# ===================================================================== E2
# Phan bo chat luong cua MAU THO -- policy vs ngau nhien (khong xep hang).
N = 2000
e2 = {}
for name, temp in (("policy_t1.0", 1.0), ("policy_t1.6", 1.6), ("random", None)):
    torch.manual_seed(7); rng = np.random.default_rng(7)
    if temp is None:
        lv = rand_levels(N, rng)[0].cpu().numpy()
    else:
        with torch.no_grad():
            lv = sample_levels(logits, k=N, temperature=temp)[0].cpu().numpy()
    t0 = time.perf_counter()
    fs, ok = [], []
    for i in range(N):
        f, o, *_ = exact(lv[i])
        fs.append(f); ok.append(bool(o))
    e2[name] = {"fit": fs, "feas": ok}
    print(f"  E2 {name:12s} kha thi {100*np.mean(ok):5.1f} %  "
          f"fit_med {np.median(fs):8.3f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
OUT["e2"] = e2

# ===================================================================== E3
# Bien rang buoc theo tung AC: Pr(D>=Dmax)/eps. > 1 la VI PHAM.
best_i = pick_best(sample_levels(logits, k=512, temperature=1.0))
with torch.no_grad():
    lv512 = sample_levels(logits, k=512, temperature=1.0)
bi = pick_best(lv512)
bf, bok, bviol, bploss, bgen = exact(lv512[0, bi].cpu().numpy())
print(f"  E3 policy best-of-512 fit={bf:.3f} feas={bok}", flush=True)

t0 = time.perf_counter()
seeds = policy_seeds(pol, sur, acs, NL, n_seed=80, temperature=1.6)
ga = GAParams(n_pop=80, n_gen=GA.n_gen, n_elite=3, p_cross=GA.p_cross,
              p_mutate=GA.p_mutate, n_stag=GA.n_stag, seed=2025)
r = run_ga_surrogate(acs, sur, n_links=NL, ga=ga, seed=2025, verify_top=8,
                     seed_genomes=seeds)
print(f"  E3 GA+GNN+policy fit={r.fitness:.3f} feas={r.qos.feasible(acs)} "
      f"({time.perf_counter()-t0:.1f}s)", flush=True)
OUT["e3"] = {
    "greedy":  {"fit": gf,  "viol": gviol.tolist(),  "p_loss": gploss.tolist()},
    "best512": {"fit": bf,  "viol": bviol.tolist(),  "p_loss": bploss.tolist()},
    "ga":      {"fit": float(r.fitness), "viol": r.qos.violation.tolist(),
                "p_loss": r.qos.p_loss.tolist()},
}

# ===================================================================== E4
# Duong huan luyen policy -- doc lai tu policy_log.txt (da co san).
import re
steps, elite, ex_f, ex_ok = [], [], [], []
txt = open(TRAIN_LOG, encoding="utf-8-sig").read()
for m in re.finditer(r"buoc\s+(\d+)/\d+\s+elite\(surrogate\)\s+([-\d.]+)\s+\|"
                     r".*?CHINH XAC\s+([-\d.]+)\s+kha thi\s+(\w+)", txt):
    steps.append(int(m.group(1))); elite.append(float(m.group(2)))
    ex_f.append(float(m.group(3))); ex_ok.append(m.group(4) == "True")
OUT["e4"] = {"step": steps, "elite": elite, "exact": ex_f, "feas": ex_ok}
print(f"  E4 doc duoc {len(steps)} moc huan luyen", flush=True)

json.dump(OUT, open(os.path.join(ROOT, "results", "pp2_policy.json"), "w"), indent=1)
print("\nDa luu -> results/pp2_policy.json")

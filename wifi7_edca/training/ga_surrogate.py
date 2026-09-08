"""
training/ga_surrogate.py -- GA duoc HO TRO boi GNN surrogate (Algorithm 1 tang toc).

Y tuong: GNN KHONG thay the mo hinh giai tich, no chi SANG LOC.

    moi the he:
        1. danh gia CA QUAN THE bang GNN -- MOT forward pass duy nhat
        2. chon K ca the tot nhat theo surrogate
        3. chi K ca the do moi duoc mo hinh giai tich kiem chung (chinh xac)
        4. fitness dung de chon loc = chinh xac neu co, surrogate neu chua

Vi sao phai co buoc 3: README muc 6.4 chi ro GA luon day nghiem toi uu toi SAT
bien rang buoc, ma o do sai so +-2x cua bat ky mo hinh nao cung la quyet dinh.
Neu tin surrogate hoan toan, GA se "khai thac" chinh sai so cua surrogate va tra
ve nghiem thuc te vi pham rang buoc. Voi buoc kiem chung, NGHIEM CUOI CUNG luon
duoc common.utility.evaluate_config xac nhan -- ket qua bao cao van chinh xac
tuyet doi, chi co qua trinh tim kiem la duoc tang toc.

Chay so sanh voi GA goc:
    python -m training.ga_surrogate --model results/gnn_model.pt
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.utility import evaluate_config, fitness
from config import GA, ACConfig, GAParams
from datagen.genome import GenomeSpec, decode, encode, random_genome
from datagen.graph import build_graph
from training.ga import (GAHistory, GAResult, _crossover, _mutate, _tournament,
                         coordinate_polish, structured_seeds)
from training.gnn import Surrogate


def polish_screened(genome: np.ndarray, spec: GenomeSpec, upper: np.ndarray,
                    ac_set: Sequence[ACConfig], surrogate: Surrogate,
                    eval_exact, n_pass: int = 3, n_verify: int = 3):
    """Tinh chinh theo toa do, nhung GNN sang loc truoc.

    training.ga.coordinate_polish phai danh gia CHINH XAC ~6 muc lan can cho moi
    gen -- voi 30 gen va 3 luot la ~540 lan nghich dao Fourier, ap dao ca ngan
    sach danh gia cua GA. O day GNN cham diem TOAN BO cac muc cua mot gen trong
    mot lan goi, roi chi `n_verify` muc hua hen nhat moi duoc kiem chung. Vua re
    hon vua tot hon: duyet het moi muc thay vi chi lan can +-2.
    """
    g = genome.copy()
    best = eval_exact(g)
    for _ in range(n_pass):
        improved = False
        for j in range(len(g)):
            cur = int(g[j])
            cands = [v for v in range(int(upper[j])) if v != cur]
            if not cands:
                continue
            trial = np.tile(g, (len(cands), 1))
            trial[:, j] = cands
            f_sur = surrogate.fitness(
                [build_graph(decode(t, spec), ac_set) for t in trial])
            for k in np.argsort(-f_sur)[:n_verify]:
                g[j] = cands[int(k)]
                f = eval_exact(g)
                if f > best + 1e-9:
                    best, cur, improved = f, cands[int(k)], True
                else:
                    g[j] = cur
        if not improved:
            break
    return g, best


def run_ga_surrogate(ac_set: Sequence[ACConfig], surrogate: Surrogate,
                     n_links: int = 1, ga: GAParams = GA,
                     seed: Optional[int] = None, verify_top: int = 8,
                     polish: bool = True, verbose: bool = False,
                     seed_genomes: Optional[Sequence[np.ndarray]] = None
                     ) -> GAResult:
    """Algorithm 1 voi ham thich nghi duoc GNN sang loc truoc.

    `seed_genomes` -- neu duoc cung cap, cac genome nay THAY THE
    `training.ga.structured_seeds` khi gieo quan the ban dau. Dung de gieo bang
    GNN policy (muc 9.9): policy sinh mot loat cau hinh trong vai mili-giay,
    thay cho bo quy tac thu cong `structured_seeds`.
    """
    rng = np.random.default_rng(ga.seed if seed is None else seed)
    spec = GenomeSpec(n_ac=len(ac_set), n_links=n_links,
                      allow_link_choice=n_links > 1)
    upper = spec.upper()

    exact: Dict[bytes, Tuple[float, bool]] = {}     # genome -> (fitness, kha thi)
    hist = GAHistory()
    t_start = time.perf_counter()
    n_sur = 0

    def eval_exact(g: np.ndarray) -> float:
        """Danh gia CHINH XAC bang mo hinh giai tich (co nho dem)."""
        key = g.tobytes()
        hit = exact.get(key)
        if hit is None:
            res = evaluate_config(decode(g, spec), ac_set)
            hit = (fitness(res, ac_set), res.feasible(ac_set))
            exact[key] = hit
            hist.n_eval += 1
        return hit[0]

    def eval_population(pop: List[np.ndarray]) -> np.ndarray:
        """Sang loc bang GNN, kiem chung `verify_top` ca the tot nhat."""
        nonlocal n_sur
        keys = [g.tobytes() for g in pop]
        known = np.array([k in exact for k in keys])

        f = np.empty(len(pop))
        f[known] = [exact[keys[i]][0] for i in np.where(known)[0]]

        todo = np.where(~known)[0]
        if len(todo):
            graphs = [build_graph(decode(pop[i], spec), ac_set) for i in todo]
            f[todo] = surrogate.fitness(graphs)
            n_sur += len(todo)

            # kiem chung cac ca the hua hen nhat bang mo hinh giai tich
            n_ver = min(verify_top, len(todo))
            for i in todo[np.argsort(-f[todo])[:n_ver]]:
                f[i] = eval_exact(pop[i])
        return f

    def record(fit_: np.ndarray, best_f: float):
        hist.wall.append(time.perf_counter() - t_start)
        hist.evals.append(hist.n_eval)
        hist.best.append(best_f)
        hist.mean.append(float(np.mean(fit_)))
        hist.frac_feasible.append(float(np.mean(fit_ >= 0.0)))
        ok = fit_ >= 0.0
        hist.mean_feasible.append(float(np.mean(fit_[ok])) if ok.any() else np.nan)

    # --- khoi tao quan the (giong het training/ga.run_ga) -------------------
    pop = [random_genome(spec, rng) for _ in range(ga.n_pop)]
    n_seed = max(ga.n_pop // 3, 1)
    if seed_genomes is not None:
        for j, g in enumerate(list(seed_genomes)[:n_seed]):
            pop[j] = np.asarray(g, dtype=int)
    else:
        seeds = structured_seeds(ac_set, n_links)
        if len(seeds) > n_seed:
            pick = np.linspace(0, len(seeds) - 1, n_seed).astype(int)
            seeds = [seeds[i] for i in pick]
        for j, ind in enumerate(seeds[:n_seed]):
            pop[j] = encode(ind, spec)

    fit = eval_population(pop)
    best_idx = int(np.argmax(fit))
    best_g = pop[best_idx].copy()
    best_f = eval_exact(best_g)          # ca the tot nhat LUON duoc kiem chung
    stagnant = 0
    record(fit, best_f)

    for gen in range(ga.n_gen):
        elite_idx = np.argsort(-fit)[:ga.n_elite]
        children: List[np.ndarray] = [pop[i].copy() for i in elite_idx]

        while len(children) < ga.n_pop:
            pa = pop[_tournament(fit, rng)]
            pb = pop[_tournament(fit, rng)]
            c1, c2 = _crossover(pa, pb, rng, ga.p_cross)
            children.append(_mutate(c1, upper, rng, ga.p_mutate))
            if len(children) < ga.n_pop:
                children.append(_mutate(c2, upper, rng, ga.p_mutate))

        pop = children
        fit = eval_population(pop)

        # chi cap nhat "best" bang gia tri DA KIEM CHUNG
        cand = int(np.argmax(fit))
        f_cand = eval_exact(pop[cand])
        fit[cand] = f_cand
        if f_cand > best_f + 1e-9:
            best_f, best_g, stagnant = f_cand, pop[cand].copy(), 0
        else:
            stagnant += 1

        if verbose and (gen % 10 == 0 or gen == ga.n_gen - 1):
            print(f"    gen {gen:3d}/{ga.n_gen}  best = {best_f:8.3f}  "
                  f"(giai tich: {hist.n_eval}, GNN: {n_sur})", end="\r")

        record(fit, best_f)
        if stagnant >= ga.n_stag:
            hist.stopped_at = gen + 1
            break
    else:
        hist.stopped_at = ga.n_gen

    if verbose:
        print()

    if polish:
        best_g, best_f = polish_screened(best_g, spec, upper, ac_set, surrogate,
                                         eval_exact)
        # The polish is a search step like any other, so it gets its own entry
        # in every history list -- including the two timing ones, or the lists
        # go out of step and an anytime read-out lands on the wrong generation.
        for lst, val in ((hist.best, best_f), (hist.mean, hist.mean[-1]),
                         (hist.frac_feasible, hist.frac_feasible[-1]),
                         (hist.mean_feasible, hist.mean_feasible[-1]),
                         (hist.wall, time.perf_counter() - t_start),
                         (hist.evals, hist.n_eval)):
            lst.append(val)

    best_params = decode(best_g, spec)
    qos = evaluate_config(best_params, ac_set)      # nghiem cuoi: CHINH XAC
    res = GAResult(genome=best_g, params=best_params, qos=qos,
                   fitness=best_f, history=hist, spec=spec)
    res.n_surrogate = n_sur                        # type: ignore[attr-defined]
    return res


# ---------------------------------------------------------------------------
# So sanh truc tiep voi GA goc
# ---------------------------------------------------------------------------


def compare(model_path: str, n_links: int = 2, ga: GAParams = GA,
            verify_top: int = 8, seed: int = 2025, device: Optional[str] = None):
    from common.utility import summarize
    from datagen.genome import describe
    from datagen.scenario import main_scenario
    from training.ga import run_ga

    acs = main_scenario()
    sur = Surrogate.load(model_path, device)
    print(f"surrogate chay tren: {sur.device}\n")

    print("=== [1/2] GA goc (chi mo hinh giai tich) ===")
    t0 = time.time()
    r_base = run_ga(acs, n_links=n_links, ga=ga, seed=seed, verbose=True)
    t_base = time.time() - t0

    print("\n=== [2/2] GA + GNN surrogate ===")
    t0 = time.time()
    r_sur = run_ga_surrogate(acs, sur, n_links=n_links, ga=ga, seed=seed,
                             verify_top=verify_top, verbose=True)
    t_sur = time.time() - t0

    n_sur_eval = getattr(r_sur, "n_surrogate", 0)
    print("\n" + "=" * 62)
    print(f"{'':22} {'GA goc':>16} {'GA + GNN':>16}")
    print("-" * 62)
    print(f"{'thoi gian (s)':22} {t_base:16.1f} {t_sur:16.1f}")
    print(f"{'danh gia giai tich':22} {r_base.history.n_eval:16d} "
          f"{r_sur.history.n_eval:16d}")
    print(f"{'danh gia bang GNN':22} {0:16d} {n_sur_eval:16d}")
    print(f"{'fitness (da kiem chung)':22} {r_base.fitness:16.3f} "
          f"{r_sur.fitness:16.3f}")
    print(f"{'kha thi':22} {str(r_base.qos.feasible(acs)):>16} "
          f"{str(r_sur.qos.feasible(acs)):>16}")
    print(f"{'the he dung':22} {r_base.history.stopped_at:16d} "
          f"{r_sur.history.stopped_at:16d}")
    print("-" * 62)
    print(f"tang toc: {t_base/max(t_sur,1e-9):.2f}x  ·  "
          f"giam danh gia giai tich: "
          f"{r_base.history.n_eval/max(r_sur.history.n_eval,1):.2f}x")
    print("=" * 62)

    print("\n--- Nghiem cua GA + GNN (da kiem chung bang mo hinh giai tich) ---")
    print(describe(r_sur.params))
    print(summarize(r_sur.qos, acs))
    return r_base, r_sur


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GA ho tro boi GNN surrogate")
    ap.add_argument("--model", type=str, default="results/gnn_model.pt")
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--pop", type=int, default=0, help="0 = mac dinh config.GA")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--verify-top", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()

    g = GA
    if a.pop or a.gen:
        g = GAParams(n_pop=a.pop or GA.n_pop, n_gen=a.gen or GA.n_gen,
                     n_elite=GA.n_elite, p_cross=GA.p_cross,
                     p_mutate=GA.p_mutate, n_stag=GA.n_stag, seed=GA.seed)
    compare(a.model, n_links=a.links, ga=g, verify_top=a.verify_top,
            seed=a.seed, device=a.device)

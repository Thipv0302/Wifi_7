"""
training/ga.py -- MODULE HUAN LUYEN: thuat toan di truyen (Algorithm 1).

Tai hien Algorithm 1 "Genetic Algorithm for EDCA Parameter Optimization":

    1. Khoi tao moi truong va mien gia tri cua AIFSN, CW_min, CW_max, TXOP, R
    2. Dat tham so co dinh: I_all, n, D_max, eps
    3. Dat tham so GA: N_pop, N_gen, ty le lai, so ca the tinh hoa, nguong dung som
    4. Ham thich nghi: tinh moi P_loss,i va tra ve sum_i -log(P_loss,i)
    5. Rang buoc: Pr(D >= D_max) - eps <= 0
    6. Chay GA:
         khoi tao quan the trong mien tham so
         voi moi the he:
             danh gia thich nghi
             chon loc cha me
             lai ghep + dot bien sinh the he con
             giu lai ca the tinh hoa
             cap nhat quan the
    7. Giai ma nghiem toi uu -> P_loss, Pr(D >= D_max) va bo tham so

Rang buoc duoc dua vao ham thich nghi duoi dang PHAT (xem common/utility.fitness)
nen GA luon uu tien vung kha thi ma khong can toan tu sua chua rieng.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.utility import QoSResult, evaluate_config, fitness
from config import ACConfig, EDCAParams, GA, GAParams
from datagen.genome import GenomeSpec, decode, encode, random_genome


@dataclass
class GAHistory:
    """Lich su hoi tu (dung ve Fig. 4)."""

    best: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)          # trung binh toan quan the
    mean_feasible: List[float] = field(default_factory=list)  # trung binh ca the kha thi
    frac_feasible: List[float] = field(default_factory=list)
    n_eval: int = 0
    stopped_at: int = 0

    def as_dict(self) -> Dict:
        return {"best": self.best, "mean": self.mean,
                "mean_feasible": self.mean_feasible,
                "frac_feasible": self.frac_feasible,
                "n_eval": self.n_eval, "stopped_at": self.stopped_at}


@dataclass
class GAResult:
    genome: np.ndarray
    params: List[EDCAParams]
    qos: QoSResult
    fitness: float
    history: GAHistory
    spec: GenomeSpec


# ---------------------------------------------------------------------------
# Toan tu di truyen
# ---------------------------------------------------------------------------


def _tournament(fit: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    idx = rng.integers(0, len(fit), size=k)
    return int(idx[np.argmax(fit[idx])])


def _crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
               p_cross: float) -> Tuple[np.ndarray, np.ndarray]:
    """Lai ghep deu (uniform crossover) tren tung gen."""
    if rng.random() >= p_cross:
        return a.copy(), b.copy()
    mask = rng.random(len(a)) < 0.5
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1, c2


def _mutate(g: np.ndarray, upper: np.ndarray, rng: np.random.Generator,
            p_mut: float, p_local: float = 0.6) -> np.ndarray:
    """Dot bien hon hop.

    - Voi xac suat `p_local`: dich chuyen gen mot buoc (+-1 muc) -- giup tinh
      chinh nghiem quanh vung kha thi, noi rang buoc tre rat nhay cam.
    - Con lai: dat lai gen bang gia tri ngau nhien trong mien -- giu kha nang
      thoat khoi cuc tri dia phuong.
    """
    m = rng.random(len(g)) < p_mut
    if not m.any():
        return g
    out = g.copy()
    idx = np.where(m)[0]
    local = rng.random(len(idx)) < p_local
    step = rng.choice([-1, 1], size=len(idx))
    out[idx[local]] = np.clip(out[idx[local]] + step[local], 0,
                              upper[idx[local]] - 1)
    rest = idx[~local]
    if len(rest):
        out[rest] = rng.integers(0, upper[rest])
    return out


# ---------------------------------------------------------------------------
# Ca the "moi" co cau truc -- giup GA vao vung kha thi nhanh hon
# ---------------------------------------------------------------------------


def structured_seeds(ac_set: Sequence[ACConfig],
                     n_links: int) -> List[List[EDCAParams]]:
    """Sinh mot so cau hinh hop ly de gieo vao quan the ban dau.

    Cau truc cua nghiem tot (kiem chung bang quet tham so trong mo hinh):
      * P_loss,k = c_k^{R_k} chi nho khi c_k nho  ->  CW_min LON va R = 7;
      * nhung CW lon lai keo dai duoi tre, nen CW_max phai gan CW_min
        (m_k = 0 hoac 1) de tre co chan tren chat;
      * TXOP ngan giu cho moi slot backoff ngan  ->  duoi tre giam nhanh;
      * AC co eps chat nhat nen duoc TACH sang link rieng de bot tranh chap.
    Cac cau hinh nay chi la diem xuat phat; GA van tu do tim kiem.
    """
    order = np.argsort([a.epsilon for a in ac_set])       # chat -> long
    rank = np.argsort(order)                              # rank[i] = hang cua AC i
    n_ac = len(ac_set)
    seeds: List[List[EDCAParams]] = []

    for cw_base in (16, 32, 64, 128, 256):
        for span in (1, 2):
            for txop in (0.0, 512.0, 2048.0):
                for split in range(1, max(n_ac, 2)):
                    params = []
                    for i in range(n_ac):
                        rk = int(rank[i])
                        params.append(EDCAParams(
                            cw_min=cw_base,
                            cw_max=min(cw_base * span, 1023),
                            aifsn=2 + min(rk, 13),
                            txop_us=txop,
                            retry=7,
                            link=(0 if rk < split else 1) % max(n_links, 1)))
                    seeds.append(params)
    return seeds


# ---------------------------------------------------------------------------
# Tinh chinh cuc bo (memetic): ha xuong theo tung toa do
# ---------------------------------------------------------------------------


def coordinate_polish(genome: np.ndarray, spec: GenomeSpec, upper: np.ndarray,
                      evaluate, n_pass: int = 3) -> Tuple[np.ndarray, float]:
    """Duyet lan luot tung gen, thu moi muc lan can va giu muc tot nhat.

    GA tim duoc vung nghiem tot nhung thuong dung lai cach cuc tri dia phuong
    vai bac; mot vai luot ha theo toa do (chi phi ~ do_dai_gen x so_muc lan
    danh gia) day them duoc vai diem thich nghi ma khong lam thay doi cau truc
    nghiem. Day la buoc "tinh chinh" cua so do memetic, khong thay the GA.
    """
    g = genome.copy()
    best = evaluate(g)
    for _ in range(n_pass):
        improved = False
        for j in range(len(g)):
            cur = int(g[j])
            hi = int(upper[j])
            # thu ca lan can gan (+-1, +-2) lan toan bo mien neu mien nho
            cands = ([v for v in range(hi)] if hi <= 12 else
                     sorted({max(cur - 2, 0), max(cur - 1, 0),
                             min(cur + 1, hi - 1), min(cur + 2, hi - 1),
                             0, hi - 1}))
            for v in cands:
                if v == cur:
                    continue
                g[j] = v
                f = evaluate(g)
                if f > best + 1e-9:
                    best, cur, improved = f, v, True
                else:
                    g[j] = cur
        if not improved:
            break
    return g, best


# ---------------------------------------------------------------------------
# Vong lap chinh
# ---------------------------------------------------------------------------


def run_ga(ac_set: Sequence[ACConfig], n_links: int = 1,
           ga: GAParams = GA, seed: Optional[int] = None,
           seed_individuals: Optional[Sequence[Sequence[EDCAParams]]] = None,
           polish: bool = True, verbose: bool = False) -> GAResult:
    """Chay Algorithm 1 cho mot kich ban.

    n_links = 1 -> toi uu EDCA don link (baseline "Optimized EDCA")
    n_links > 1 -> toi uu dong thoi phan bo AC vao link + tham so EDCA
                   (de xuat cua paper: "Optimized MLO EDCA")
    """
    rng = np.random.default_rng(ga.seed if seed is None else seed)
    spec = GenomeSpec(n_ac=len(ac_set), n_links=n_links,
                      allow_link_choice=n_links > 1)
    upper = spec.upper()

    cache: Dict[bytes, Tuple[float, bool]] = {}
    hist = GAHistory()

    def evaluate(g: np.ndarray) -> float:
        key = g.tobytes()
        cached = cache.get(key)
        if cached is None:
            res = evaluate_config(decode(g, spec), ac_set)
            cached = (fitness(res, ac_set), res.feasible(ac_set))
            cache[key] = cached
            hist.n_eval += 1
        return cached[0]

    def is_feasible(g: np.ndarray) -> bool:
        return bool(cache[g.tobytes()][1])

    def record(pop_, fit_):
        ok = np.array([is_feasible(g) for g in pop_])
        hist.best.append(best_f)
        hist.mean.append(float(np.mean(fit_)))
        hist.frac_feasible.append(float(ok.mean()))
        hist.mean_feasible.append(float(np.mean(fit_[ok])) if ok.any() else np.nan)

    # --- khoi tao quan the -------------------------------------------------
    pop = [random_genome(spec, rng) for _ in range(ga.n_pop)]
    seeds = list(seed_individuals or []) + structured_seeds(ac_set, n_links)
    n_seed = max(ga.n_pop // 3, 1)
    if len(seeds) > n_seed:                     # lay mau deu tren danh sach seed
        pick = np.linspace(0, len(seeds) - 1, n_seed).astype(int)
        seeds = [seeds[i] for i in pick]
    for j, ind in enumerate(seeds[:n_seed]):
        pop[j] = encode(ind, spec)

    fit = np.array([evaluate(g) for g in pop])
    best_idx = int(np.argmax(fit))
    best_g, best_f = pop[best_idx].copy(), float(fit[best_idx])
    stagnant = 0

    record(pop, fit)
    for gen in range(ga.n_gen):

        # --- giu ca the tinh hoa ------------------------------------------
        elite_idx = np.argsort(-fit)[:ga.n_elite]
        children: List[np.ndarray] = [pop[i].copy() for i in elite_idx]

        # --- chon loc + lai ghep + dot bien -------------------------------
        while len(children) < ga.n_pop:
            pa = pop[_tournament(fit, rng)]
            pb = pop[_tournament(fit, rng)]
            c1, c2 = _crossover(pa, pb, rng, ga.p_cross)
            children.append(_mutate(c1, upper, rng, ga.p_mutate))
            if len(children) < ga.n_pop:
                children.append(_mutate(c2, upper, rng, ga.p_mutate))

        pop = children
        fit = np.array([evaluate(g) for g in pop])

        gen_best = int(np.argmax(fit))
        if fit[gen_best] > best_f + 1e-9:
            best_f = float(fit[gen_best])
            best_g = pop[gen_best].copy()
            stagnant = 0
        else:
            stagnant += 1

        if verbose and (gen % 10 == 0 or gen == ga.n_gen - 1):
            print(f"    gen {gen:3d}/{ga.n_gen}  best = {best_f:8.3f}  "
                  f"mean = {np.mean(fit):8.3f}  (danh gia: {hist.n_eval})",
                  end="\r")

        record(pop, fit)
        if stagnant >= ga.n_stag:                    # dung som (N_stag)
            hist.stopped_at = gen + 1
            break
    else:
        hist.stopped_at = ga.n_gen

    if verbose:
        print()

    if polish:
        best_g, best_f = coordinate_polish(best_g, spec, upper, evaluate)
        hist.best.append(best_f)
        hist.mean.append(hist.mean[-1] if hist.mean else best_f)
        hist.frac_feasible.append(hist.frac_feasible[-1]
                                  if hist.frac_feasible else 1.0)
        hist.mean_feasible.append(hist.mean_feasible[-1]
                                  if hist.mean_feasible else best_f)

    best_params = decode(best_g, spec)
    qos = evaluate_config(best_params, ac_set)
    return GAResult(genome=best_g, params=best_params, qos=qos,
                    fitness=best_f, history=hist, spec=spec)


if __name__ == "__main__":
    import time

    from common.utility import summarize
    from datagen.genome import describe
    from datagen.scenario import main_scenario

    acs = main_scenario()
    small = GAParams(n_pop=40, n_gen=25, n_elite=4, n_stag=15)

    for n_links, tag in ((1, "EDCA don link"), (2, "MLO EDCA (2 link)")):
        t0 = time.time()
        res = run_ga(acs, n_links=n_links, ga=small, verbose=True)
        print(f"=== {tag} === ({time.time()-t0:.0f}s, "
              f"{res.history.n_eval} lan danh gia)")
        print(describe(res.params))
        print(summarize(res.qos, acs))
        print()

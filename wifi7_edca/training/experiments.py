"""
training/experiments.py -- MODULE HUAN LUYEN / CHAY THI NGHIEM.

Moi ham `exp_figX(...)` chay mot thi nghiem cua paper va tra ve dict du lieu
tho (luu vao results/*.json); module plotting chi doc dict do va ve.

  exp_fig2  : minh hoa mo hinh vung AIFS (Fig. 2)
  exp_fig3  : do nhay tham so -- theta va P_loss theo AIFSN_2, TXOP_2 (Fig. 3)
  exp_fig4  : hoi tu cua GA cho MLO EDCA va EDCA don link (Fig. 4)
  exp_fig5  : so sanh QoS ba cau hinh (Fig. 5)
  exp_fig6  : anh huong cua nguong eps_1 (Fig. 6)
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Sequence

import numpy as np

from common.collision import solve_link
from common.utility import evaluate_config, fitness
from common.zones import build_zone_model
from config import (ACConfig, EDCAParams, GAParams, RESULT_DIR, SENSITIVITY,
                    SIGMA_US)
from datagen.genome import default_params, describe
from datagen.scenario import epsilon_sweep, main_scenario, sensitivity_scenario
from training.baselines import default_edca
from training.ga import GAResult, run_ga


# ===========================================================================
# Fig. 2 -- mo hinh vung AIFS
# ===========================================================================


def exp_fig2(aifsn: Sequence[int] = (2, 3, 3, 5),
             n_sta: Sequence[int] = (2, 4, 4, 3)) -> Dict:
    """Minh hoa cau truc vung sinh ra tu mot tap AIFS cu the (Fig. 2)."""
    from config import T_SIFS_US

    aifs = [T_SIFS_US + a * SIGMA_US for a in aifsn]
    zm = build_zone_model(aifs)
    params = [EDCAParams(cw_min=16, cw_max=1023, aifsn=int(a), txop_us=0.0,
                         retry=7) for a in aifsn]
    st = solve_link(params, list(n_sta))

    n_slots = int(zm.h.max()) + 3
    phi = [int(np.sum(zm.h <= (ell - 1))) for ell in range(1, n_slots + 1)]
    zone_of_slot = [int(np.searchsorted(zm.offsets, ell - 1, side="right") - 1)
                    for ell in range(1, n_slots + 1)]
    return {
        "aifsn": list(aifsn),
        "aifs_us": [float(a) for a in zm.aifs_us],
        "h": zm.h.tolist(),
        "offsets": zm.offsets.tolist(),
        "z_count": zm.z_count.tolist(),
        "zone_of_ac": zm.zone_of_ac.tolist(),
        "phi": phi,
        "zone_of_slot": zone_of_slot,
        "n_slots": n_slots,
        "pi": st.pi.tolist(),
        "q": st.q.tolist(),
    }


# ===========================================================================
# Fig. 3 -- do nhay tham so (Sec. IV.A)
# ===========================================================================


def exp_fig3(sc=SENSITIVITY, verbose: bool = True) -> Dict:
    """Quet AIFSN_2 x TXOP_2, do theta_1, theta_2 (Eq. 17) va P_loss (Eq. 6).

    AC_1 giu nguyen AIFSN_1 = 8, TXOP_1 = 4080 us; hai AC giong nhau ve luu
    luong (n = 4, L = 1000 B, D_max = 100 ms).
    """
    ac_set = sensitivity_scenario(sc)
    aifsn2 = list(sc.aifsn_2_values)
    txop2 = list(sc.txop_2_values)

    theta = np.zeros((2, len(aifsn2), len(txop2)))
    p_loss = np.zeros((2, len(aifsn2)))

    for i, a2 in enumerate(aifsn2):
        for j, t2 in enumerate(txop2):
            params = [
                EDCAParams(sc.cw_min, sc.cw_max, sc.aifsn_1, sc.txop_1_us,
                           sc.retry, link=0),
                EDCAParams(sc.cw_min, sc.cw_max, int(a2), float(t2),
                           sc.retry, link=0),
            ]
            res = evaluate_config(params, ac_set)
            theta[0, i, j] = res.theta[0]
            theta[1, i, j] = res.theta[1]
            if j == 0:
                p_loss[0, i] = res.p_loss[0]
                p_loss[1, i] = res.p_loss[1]
        if verbose:
            print(f"   Fig.3: AIFSN_2 = {a2:2d} xong", end="\r")
    if verbose:
        print()

    return {"aifsn2": aifsn2, "txop2": txop2,
            "theta_ac1": theta[0].tolist(), "theta_ac2": theta[1].tolist(),
            "ploss_ac1": p_loss[0].tolist(), "ploss_ac2": p_loss[1].tolist(),
            "aifsn1": sc.aifsn_1, "txop1": sc.txop_1_us}


# ===========================================================================
# Fig. 4 & Fig. 5 -- toi uu bang GA va so sanh QoS
# ===========================================================================


def _ga_config(n_pop: int, n_gen: int, stag: int, seed: int) -> GAParams:
    return GAParams(n_pop=n_pop, n_gen=n_gen, n_elite=max(n_pop // 25, 4),
                    p_cross=0.8, p_mutate=0.05, n_stag=stag, seed=seed)


def exp_fig4_fig5(n_pop: int = 200, n_gen: int = 300, n_links: int = 2,
                  seed: int = 2025, verbose: bool = True) -> Dict:
    """Chay GA cho ca hai cau hinh, tra ve lich su hoi tu (Fig. 4) va ket qua
    QoS de so sanh (Fig. 5)."""
    ac_set = main_scenario()
    ga = _ga_config(n_pop, n_gen, 50, seed)     # N_stag = 50 (TABLE I)

    out: Dict = {"ac_names": [a.name for a in ac_set],
                 "epsilon": [a.epsilon for a in ac_set]}

    if verbose:
        print("   Fig.4: GA cho EDCA don link ...")
    t0 = time.time()
    res_sl = run_ga(ac_set, n_links=1, ga=ga, verbose=verbose)
    t_sl = time.time() - t0

    if verbose:
        print("   Fig.4: GA cho MLO EDCA ...")
    t0 = time.time()
    res_ml = run_ga(ac_set, n_links=n_links, ga=ga, verbose=verbose)
    t_ml = time.time() - t0

    _, res_def = default_edca(ac_set)

    out["history"] = {"EDCA": res_sl.history.as_dict(),
                      "MLO EDCA": res_ml.history.as_dict()}
    out["runtime_s"] = {"EDCA": t_sl, "MLO EDCA": t_ml}
    out["configs"] = {}
    for name, params, res in (
            ("Default EDCA", default_params(len(ac_set)), res_def),
            ("Opt. EDCA", res_sl.params, res_sl.qos),
            ("Opt. MLO EDCA", res_ml.params, res_ml.qos)):
        out["configs"][name] = {
            "violation": res.violation.tolist(),
            "p_loss": res.p_loss.tolist(),
            "theta": res.theta.tolist(),
            "objective": res.objective,
            "fitness": fitness(res, ac_set),
            "feasible": bool(res.feasible(ac_set)),
            "params": [[p.cw_min, p.cw_max, p.aifsn, p.txop_us, p.retry, p.link]
                       for p in params],
            "table": describe(params),
        }
    return out


# ===========================================================================
# Fig. 6 -- anh huong cua nguong eps_1
# ===========================================================================


def exp_fig6(eps1_values=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
             n_pop: int = 120, n_gen: int = 120, n_links: int = 2,
             seed: int = 7, verbose: bool = True) -> Dict:
    """Voi moi gia tri eps_1, chay GA DOC LAP cho EDCA don link va MLO EDCA.

    Moi diem tren duong cong la mot lan chay Algorithm 1 rieng, dung nhu cach
    paper dung sinh Fig. 6 -- va do do duong cong KHONG don dieu:

        "As shown in Fig. 6a, the fitness value does not increase
         monotonically, reflecting a trade-off between fitness and reliability"

    (Mot phien ban truoc cua module nay gop chung kho nghiem giua cac eps_1 de
    khu nhieu cua GA. Cach do cho duong cong phang -- dung ve mat toi uu nhung
    khong con giong Fig. 6 cua paper, nen da bo.)
    """
    rows = {"EDCA": {"fitness": [], "sum_theta": [], "feasible": []},
            "MLO EDCA": {"fitness": [], "sum_theta": [], "feasible": []}}
    target = []

    for pos, (e1, ac_set) in enumerate(epsilon_sweep(eps1_values)):
        target.append(float(np.sum([-np.log10(a.epsilon) for a in ac_set])))
        for tag, nl in (("EDCA", 1), ("MLO EDCA", n_links)):
            # moi (eps_1, cau hinh) mot hat giong rieng -> cac lan chay doc lap
            ga = _ga_config(n_pop, n_gen, 40, seed + 100 * pos + (0 if nl == 1 else 1))
            res = run_ga(ac_set, n_links=nl, ga=ga)
            ok = res.qos.feasible(ac_set)
            rows[tag]["feasible"].append(bool(ok))
            rows[tag]["fitness"].append(res.qos.objective if ok else float("nan"))
            rows[tag]["sum_theta"].append(float(np.sum(res.qos.theta)) if ok
                                          else float("nan"))
        if verbose:
            print(f"   Fig.6: eps_1 = {e1:.0e} xong "
                  f"(EDCA {rows['EDCA']['fitness'][-1]:.2f}, "
                  f"MLO {rows['MLO EDCA']['fitness'][-1]:.2f})")

    return {"eps1": list(map(float, eps1_values)), "target_sum_theta": target,
            **rows}


# ===========================================================================
# Luu / doc ket qua
# ===========================================================================


def save(name: str, data: Dict) -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path


def load(name: str) -> Dict:
    with open(os.path.join(RESULT_DIR, f"{name}.json")) as fh:
        return json.load(fh)

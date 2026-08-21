"""
training/simulate.py -- BO MO PHONG EDCA MUC SLOT (dung de KIEM CHUNG mo hinh).

Paper chi dua ra mo hinh giai tich; module nay mo phong truc tiep co che EDCA
theo tung slot de doi chieu hai dai luong then chot:

  * P_loss,k  -- Eq.(6)
  * Pr(D_k >= x) -- CCDF lay tu Eq.(10)-(16)

Quy uoc duoc giu DUNG NHU mo hinh giai tich de phep so sanh co nghia:
  - kenh bao hoa (luon co goi cho phat)
  - moi lan chiem kenh gui N_k goi; goi DAU TIEN chiu toan bo tre cho + backoff
    + T_DATA, cac goi con lai chiu Delta_k (dung nhu Eq. 10)
  - AC_k chi duoc dem lui sau khi kenh ranh lien tuc AIFSN_k slot
  - va cham -> nhan doi cua so, qua R_k lan thi bo goi (Eq. 6)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from common.timing import ACTiming, T_CTS_US, T_RTS_US, ac_timing
from config import ACConfig, EDCAParams, SIGMA_US, T_SIFS_US


@dataclass
class SimResult:
    p_loss: np.ndarray                 # (I,) ty le mat goi do phat lai qua R lan
    delays_us: List[np.ndarray]        # (I,) mau tre cua tung AC
    n_success: np.ndarray
    n_drop: np.ndarray
    sim_time_us: float

    def violation(self, d_max_us: Sequence[float]) -> np.ndarray:
        return np.array([float(np.mean(d >= dm)) if len(d) else np.nan
                         for d, dm in zip(self.delays_us, d_max_us)])


def simulate(params: Sequence[EDCAParams], ac_set: Sequence[ACConfig],
             n_slots: int = 4_000_000, seed: int = 0,
             max_delay_samples: int = 200_000) -> SimResult:
    """Mo phong `n_slots` slot backoff (khong ke thoi gian truyen)."""
    rng = np.random.default_rng(seed)
    n_ac = len(params)

    tim: List[ACTiming] = [ac_timing(params[i], ac_set[i].payload_bytes)
                           for i in range(n_ac)]
    aifsn = np.array([p.aifsn for p in params])
    cw_min = np.array([p.cw_min for p in params], dtype=float)
    cw_max = np.array([max(p.cw_max, p.cw_min) for p in params], dtype=float)
    retry_max = np.array([p.retry for p in params])
    n_sta = np.array([a.n_sta for a in ac_set])

    # mo rong ra tung TRAM
    sta_ac = np.concatenate([[i] * n_sta[i] for i in range(n_ac)])
    n_dev = len(sta_ac)
    stage = np.zeros(n_dev, dtype=int)
    cw = cw_min[sta_ac].copy()
    backoff = np.floor(rng.random(n_dev) * cw).astype(int)
    head_time = np.zeros(n_dev)          # thoi diem goi len dau hang doi

    t_us = 0.0
    idle_slots = 0                       # so slot ranh lien tiep sau busy period
    n_success = np.zeros(n_ac)
    n_drop = np.zeros(n_ac)
    delays: List[List[float]] = [[] for _ in range(n_ac)]

    for _ in range(n_slots):
        eligible = idle_slots >= aifsn[sta_ac]
        ready = eligible & (backoff == 0)
        n_tx = int(np.sum(ready))

        if n_tx == 0:
            backoff[eligible] -= 1
            np.maximum(backoff, 0, out=backoff)
            t_us += SIGMA_US
            idle_slots += 1
            continue

        tx = np.where(ready)[0]
        if n_tx == 1:
            d = int(tx[0])
            k = sta_ac[d]
            # tre cua goi dau tien trong TXOP: cho + backoff + T_DATA
            wait = t_us - head_time[d]
            delay = wait + T_RTS_US + T_CTS_US + 2 * T_SIFS_US + tim[k].t_data_us
            if len(delays[k]) < max_delay_samples:
                delays[k].append(delay)
                # cac goi con lai trong TXOP chi chiu Delta_k (dung nhu Eq. 10)
                for _j in range(tim[k].n_pkt - 1):
                    if len(delays[k]) < max_delay_samples:
                        delays[k].append(tim[k].delta_us)
            n_success[k] += tim[k].n_pkt
            t_us += tim[k].t_s_us - tim[k].aifs_us
            stage[d] = 0
            cw[d] = cw_min[k]
            backoff[d] = int(rng.random() * cw[d])
            head_time[d] = t_us
        else:
            longest = 0.0
            for d in tx:
                k = sta_ac[d]
                longest = max(longest, tim[k].t_c_us - tim[k].aifs_us)
                stage[d] += 1
                if stage[d] > retry_max[k]:          # vuot gioi han phat lai
                    n_drop[k] += 1
                    stage[d] = 0
                    cw[d] = cw_min[k]
                    head_time[d] = t_us              # goi moi len dau hang doi
                else:
                    cw[d] = min(cw[d] * 2, cw_max[k])
                backoff[d] = int(rng.random() * cw[d])
            t_us += longest
        idle_slots = 0

    total = n_success + n_drop
    p_loss = np.where(total > 0, n_drop / np.maximum(total, 1), np.nan)
    return SimResult(p_loss=p_loss,
                     delays_us=[np.array(d) for d in delays],
                     n_success=n_success, n_drop=n_drop, sim_time_us=t_us)


def compare_with_model(params: Sequence[EDCAParams], ac_set: Sequence[ACConfig],
                       n_slots: int = 4_000_000, seed: int = 0) -> str:
    """So sanh P_loss va Pr(D >= D_max) giua mo phong va mo hinh giai tich."""
    from common.utility import evaluate_config

    sim = simulate(params, ac_set, n_slots=n_slots, seed=seed)
    mdl = evaluate_config(params, ac_set)
    viol_sim = sim.violation([a.d_max_us for a in ac_set])

    lines = [f"{'AC':>5} {'P_loss (mo phong)':>19} {'P_loss (mo hinh)':>18} "
             f"{'Pr(D>=Dmax) mo phong':>22} {'mo hinh':>12}"]
    for i, a in enumerate(ac_set):
        lines.append(f"{a.name:>5} {sim.p_loss[i]:19.4e} {mdl.p_loss[i]:18.4e} "
                     f"{viol_sim[i]:22.4e} {mdl.violation[i]:12.4e}")
    lines.append(f"thoi gian mo phong = {sim.sim_time_us/1e6:.2f} s | "
                 f"so mau tre = {[len(d) for d in sim.delays_us]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import time

    from config import ACConfig, EDCAParams

    # Kich ban kiem chung: 2 AC, tham so vua phai de mo phong hoi tu nhanh
    acs = [ACConfig("AC1", 3, 500, 20.0, 1e-3),
           ACConfig("AC2", 3, 500, 20.0, 1e-3)]
    prm = [EDCAParams(16, 256, 2, 0.0, 5, link=0),
           EDCAParams(16, 256, 5, 0.0, 5, link=0)]

    t0 = time.time()
    print(compare_with_model(prm, acs, n_slots=3_000_000, seed=1))
    print(f"({time.time()-t0:.0f}s)")

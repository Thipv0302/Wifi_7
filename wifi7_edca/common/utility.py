"""
common/utility.py -- Ham "utility" dung chung: danh gia QoS cua MOT cau hinh.

Day la diem vao duy nhat ma cac module khac (training, plotting) goi toi, nen
moi cong thuc muc tieu chi ton tai o mot cho:

  Eq.(6)  P_loss,i = c_i^{R_i}
  Eq.(17) theta_i  = -log Pr(D_i >= D_max,i)
  Eq.(18) P1: argmax_x sum_i -log(P_loss,i)
              s.t.     Pr(D_i >= D_max,i) < eps_i

Ham thich nghi (fitness) cua GA = muc tieu Eq.(18) tru mot luong PHAT ty le
voi so bac do lon ma rang buoc bi vi pham.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from common.collision import LinkState, solve_link
from common.delay import build_delay_model, delay_ccdf, reliability_index
from common.timing import ac_timing
from config import ACConfig, DELTA_US, EDCAParams, X_SLOTS_MAX

# Xu ly rang buoc theo kieu "uu tien kha thi" nhung KHONG can hang so phat
# khong lo: ham muc tieu Eq.(18) luon >= 0 (vi P_loss <= 1), nen chi can cho
# ca the vi pham mot gia tri <= 0 la moi nghiem kha thi da tu dong xep tren moi
# nghiem vi pham.
#
#     fit = sum_i -log10(P_loss,i)          neu Pr(D_i >= D_max,i) < eps_i, moi i
#     fit = -lambda * sum_i excess_i        neu con vi pham
#
# Cach nay tai hien dung dang cua Fig. 4 trong paper: duong "best" xuat phat tu
# ~0 (quan the ban dau chua co ca the kha thi), duong "mean" nam quanh -4, va
# ca hai nhay bac thang khi GA tim duoc vung kha thi dau tien.
VIOLATION_PENALTY = 0.5       # lambda -- diem phat cho moi BAC DO LON vuot eps

# San duoi cua P_loss khi tinh -log10: moi AC dong gop toi da 10 diem.
# Hai ly do chon 1e-10:
#   * Fig. 5b cua paper co truc tung ket thuc DUNG o 1e-10 va nhieu cot nam sat
#     day truc -- day chinh la san ma paper dung khi bao cao P_loss;
#   * doi chieu voi mo phong muc slot cho thay c_k bi mo hinh uoc luong lech
#     ~25 %, ma P_loss = c^R nen sai so bi khuech dai luy thua R; moi gia tri
#     P_loss duoi 1e-10 deu nam ngoai do phan giai cua chinh mo hinh.
# Doi san nay lam THAY DOI THANG cua ham thich nghi (voi san 1e-12 gia tri toi
# uu MLO len toi ~44 thay vi ~38), nen day la mot lua chon can ghi ro.
P_LOSS_FLOOR = 1e-10


@dataclass
class QoSResult:
    """Ket qua danh gia QoS cua mot cau hinh EDCA da link."""

    p_loss: np.ndarray            # (I,) P_loss,i theo THU TU AC GOC
    violation: np.ndarray         # (I,) Pr(D_i >= D_max,i)
    theta: np.ndarray             # (I,) theta_i
    c: np.ndarray                 # (I,) xac suat va cham
    p_tx: np.ndarray              # (I,) xac suat phat
    link_of_ac: np.ndarray        # (I,) link ma AC duoc gan
    link_states: Dict[int, LinkState] = field(default_factory=dict)

    @property
    def objective(self) -> float:
        """Eq.(18): sum_i -log(P_loss,i).

        Dung logarit CO SO 10: thang gia tri fitness cua paper (Fig. 4 dat toi
        ~35 voi 5 AC) chi khop khi -log10 duoc dung, doi chieu voi cac muc
        P_loss trong Fig. 5b (1e-10 .. 1e-1). P_loss duoc kep duoi tai
        `P_LOSS_FLOOR` = 1e-10 -- xem ghi chu o dau module.
        """
        return float(np.sum(-np.log10(np.maximum(self.p_loss, P_LOSS_FLOOR))))

    def feasible(self, ac_set: Sequence[ACConfig]) -> bool:
        return bool(np.all(self.violation < np.array([a.epsilon for a in ac_set])))

    def excess(self, ac_set: Sequence[ACConfig]) -> np.ndarray:
        """So bac do lon (log10) ma tung AC vuot nguong eps_i (0 neu dat)."""
        eps = np.array([a.epsilon for a in ac_set])
        return np.maximum(np.log10(np.maximum(self.violation, 1e-300) / eps), 0.0)


# ---------------------------------------------------------------------------
# Chon buoc luong tu hoa thoi gian cho tung AC
# ---------------------------------------------------------------------------


def choose_delta_us(d_max_us: float, delta_base: float = DELTA_US,
                    x_max: int = X_SLOTS_MAX) -> float:
    """Buoc delta nho nhat (boi cua delta_base) sao cho D_max/delta <= x_max.

    D_max lon (300 ms) se dung delta = 20 us thay vi 10 us: so diem DFT giam
    mot nua ma sai so lam tron thoi gian van duoi 10 us tren moi su kien.
    """
    factor = max(1, int(np.ceil(d_max_us / (delta_base * x_max))))
    return delta_base * factor


# ---------------------------------------------------------------------------
# Danh gia mot cau hinh
# ---------------------------------------------------------------------------


def evaluate_config(params: Sequence[EDCAParams], ac_set: Sequence[ACConfig],
                    compute_delay: bool = True) -> QoSResult:
    """Danh gia day du mot cau hinh EDCA da link.

    Cac AC duoc nhom theo `link`; moi link giai he diem co dinh rieng (cac link
    o bang tan khac nhau nen doc lap ve tranh chap).
    """
    n_ac = len(ac_set)
    p_loss = np.ones(n_ac)
    violation = np.ones(n_ac)
    c_all = np.zeros(n_ac)
    p_all = np.zeros(n_ac)
    link_of = np.array([p.link for p in params], dtype=int)
    states: Dict[int, LinkState] = {}

    for link in sorted(set(link_of)):
        idx = [i for i in range(n_ac) if params[i].link == link]
        sub_params = [params[i] for i in idx]
        sub_n = [ac_set[i].n_sta for i in idx]

        st = solve_link(sub_params, sub_n)
        states[link] = st
        order = st.zm.order                      # chi so trong `idx`, sap theo AIFS
        timings = [ac_timing(sub_params[j], ac_set[idx[j]].payload_bytes)
                   for j in order]

        for k in range(st.zm.n_ac):
            gi = idx[order[k]]                   # chi so AC goc
            p_loss[gi] = st.p_loss[k]
            c_all[gi] = st.c[k]
            p_all[gi] = st.p[k]
            if compute_delay:
                d_us = choose_delta_us(ac_set[gi].d_max_us)
                dm = build_delay_model(st, k, timings, delta_us=d_us)
                x = int(round(ac_set[gi].d_max_us / d_us))
                violation[gi] = delay_ccdf(dm, x)

    theta = np.array([reliability_index(v) for v in violation])
    return QoSResult(p_loss=p_loss, violation=violation, theta=theta,
                     c=c_all, p_tx=p_all, link_of_ac=link_of, link_states=states)


# ---------------------------------------------------------------------------
# Ham thich nghi cua GA
# ---------------------------------------------------------------------------


def fitness(res: QoSResult, ac_set: Sequence[ACConfig],
            penalty: float = VIOLATION_PENALTY) -> float:
    """Muc tieu Eq.(18) kem xu ly rang buoc theo kieu "uu tien kha thi".

        fit = sum_i -log10(P_loss,i)      neu moi rang buoc dat  (>= 0)
        fit = -lambda * sum_i excess_i     neu co vi pham          (<= 0)

    Vi Eq.(18) luon khong am, MOI nghiem kha thi deu tot hon MOI nghiem vi pham
    (dung nghia "s.t." cua Eq. 18) ma khong can hang so phat lon; trong nhom vi
    pham thi ca the gan kha thi hon duoc uu tien.
    """
    excess = res.excess(ac_set)
    if np.all(excess <= 0.0):
        return res.objective
    return -penalty * float(np.sum(excess))


def summarize(res: QoSResult, ac_set: Sequence[ACConfig]) -> str:
    lines = [f"{'AC':>5} {'link':>5} {'c':>8} {'P_loss':>11} "
             f"{'Pr(D>=Dmax)':>13} {'eps':>9} {'dat?':>5}"]
    for i, ac in enumerate(ac_set):
        ok = "OK" if res.violation[i] < ac.epsilon else "VI PHAM"
        lines.append(f"{ac.name:>5} {res.link_of_ac[i]:5d} {res.c[i]:8.4f} "
                     f"{res.p_loss[i]:11.3e} {res.violation[i]:13.3e} "
                     f"{ac.epsilon:9.1e} {ok:>7}")
    lines.append(f"muc tieu Eq.(18) = {res.objective:.3f} | "
                 f"fitness = {fitness(res, ac_set):.3f} | "
                 f"kha thi = {res.feasible(ac_set)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import time

    from config import AC_SET, DEFAULT_EDCA

    t0 = time.perf_counter()
    res = evaluate_config(DEFAULT_EDCA, AC_SET)
    dt = (time.perf_counter() - t0) * 1000
    print("=== EDCA mac dinh, tat ca AC tren mot link ===")
    print(summarize(res, AC_SET))
    print(f"thoi gian danh gia: {dt:.0f} ms")

    # thu chia AC ra 2 link
    mlo = [p.copy() for p in DEFAULT_EDCA]
    for i, lk in enumerate([0, 1, 1, 0, 0]):
        mlo[i].link = lk
    t0 = time.perf_counter()
    res2 = evaluate_config(mlo, AC_SET)
    print("\n=== Cung tham so nhung chia 2 link (MLO) ===")
    print(summarize(res2, AC_SET))
    print(f"thoi gian danh gia: {(time.perf_counter()-t0)*1000:.0f} ms")

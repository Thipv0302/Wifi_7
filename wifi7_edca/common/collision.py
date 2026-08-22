"""
common/collision.py -- Mo hinh va cham EDCA (Sec. III.A).

Tai hien he diem co dinh:
  Eq.(4)  c_k = sum_j [pi_j / sum_i pi_i] (1 - (prod_l r_l^{n_l}) / r_k)
  Eq.(5)  p_k = 2 / (eta_k * sum_{j=0}^{R_k-1} c_k^j (f_{k,j} - 1))
          voi eta_k = (1 - c_k) (1 - c_k^{R_k})^{-1},
              f_{k,j} = 2^{min(j, m_k)} CW_min,k,  m_k = log2(CW_max,k/CW_min,k)
  Eq.(6)  P_loss,k = c_k^{R_k}

He (3)-(5) duoc giai lap cho tung LINK (cac AC tren cung mot link tranh chap
voi nhau; cac link doc lap ve tan so nen giai rieng).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from common.zones import (ZoneModel, build_zone_model, zone_idle_prob,
                          zone_stationary)
from config import EDCAParams

EPS = 1e-300


@dataclass
class LinkState:
    """Nghiem diem co dinh cua mot link (theo THU TU SAP XEP AIFS tang dan)."""

    zm: ZoneModel
    n_sta: np.ndarray          # (I,) so tram cua tung AC
    p: np.ndarray              # (I,) p_k -- xac suat phat trong mot slot
    r: np.ndarray              # (I,) r_k = 1 - p_k
    c: np.ndarray              # (I,) c_k -- xac suat va cham
    q: np.ndarray              # (J,) q_j -- xac suat khong ai phat trong vung j
    pi: np.ndarray             # (J,) pi_j
    p_loss: np.ndarray         # (I,) P_loss,k = c_k^{R_k}
    retry: np.ndarray          # (I,) R_k
    cw_min: np.ndarray         # (I,) CW_min,k
    cw_max: np.ndarray         # (I,) CW_max,k
    converged: bool
    n_iter: int

    def cw_at_stage(self, k: int, j: int) -> int:
        """f_{k,j} = min(2^j CW_min,k , CW_max,k) -- cua so o buoc backoff j.

        Paper viet f_{k,j} = 2^{min(j, m_k)} CW_min,k voi
        m_k = log2(CW_max,k / CW_min,k). Hai cach viet TRUNG NHAU khi ty so
        CW_max/CW_min la luy thua cua 2; khi khong phai (mien [1,1023] cua
        Sec. IV.B cho phep), dang min(...) van cho SO NGUYEN -- dieu bat buoc
        vi f_{k,j} la so mu cua z trong Eq.(13) (z^f voi f khong nguyen se roi
        vao nhanh cat cua ham mu phuc).
        """
        return int(min(2 ** j * self.cw_min[k], self.cw_max[k]))

    def cw_stages(self, k: int) -> np.ndarray:
        """Vector f_{k,j}, j = 0..R_k-1 (khong giam, moi buoc gap doi den khi cham tran)."""
        return np.array([self.cw_at_stage(k, j) for j in range(int(self.retry[k]))],
                        dtype=np.int64)


# ---------------------------------------------------------------------------
# Eq. (5): p_k tu c_k
# ---------------------------------------------------------------------------


def transmit_prob(c_k: float, cw_min: float, cw_max: float, retry: int) -> float:
    """Eq.(5) -- xap xi gia tri trung binh cua xac suat phat."""
    c_k = float(np.clip(c_k, 0.0, 1.0 - 1e-12))
    j = np.arange(retry)
    f = np.minimum(2.0 ** j * cw_min, max(cw_max, cw_min))   # xem cw_at_stage
    eta = (1.0 - c_k) / (1.0 - c_k ** retry) if c_k > 0 else 1.0
    denom = eta * np.sum(c_k ** j * (f - 1.0))
    if denom <= 0:
        return 1.0
    return float(np.clip(2.0 / denom, 1e-12, 1.0))


def transmit_prob_vec(c: np.ndarray, cw_min: np.ndarray, cw_max: np.ndarray,
                      retry: np.ndarray) -> np.ndarray:
    """Eq.(5) cho TAT CA cac AC cua mot link cung luc (ban vector hoa cua
    `transmit_prob`) -- day la ham duoc goi trong vong lap diem co dinh nen
    viec bo vong lap Python o day rut ngan dang ke thoi gian chay GA."""
    c = np.clip(c, 0.0, 1.0 - 1e-12)
    r_max = int(np.max(retry))
    j = np.arange(r_max)[None, :]                       # (1, R_max)
    f = np.minimum(2.0 ** j * cw_min[:, None], np.maximum(cw_max, cw_min)[:, None])
    valid = j < retry[:, None]
    denom_terms = np.where(valid, c[:, None] ** j * (f - 1.0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(c > 0, (1.0 - c) / (1.0 - c ** retry), 1.0)
    denom = eta * denom_terms.sum(axis=1)
    return np.clip(np.where(denom > 0, 2.0 / np.maximum(denom, 1e-300), 1.0),
                   1e-12, 1.0)


# ---------------------------------------------------------------------------
# Eq. (4): c_k tu p
# ---------------------------------------------------------------------------


def collision_prob(zm: ZoneModel, r: np.ndarray, n_sta: np.ndarray,
                   q: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Eq.(4) -- xac suat va cham cua tung AC (thu tu da sap xep)."""
    c = np.zeros(zm.n_ac)
    for k in range(zm.n_ac):
        j0 = zm.zone_of_ac[k]
        w = pi[j0:]
        s = w.sum()
        if s <= 0:
            c[k] = 0.0
            continue
        # 1 - q_j / r_k : xac suat co it nhat mot tram KHAC phat trong vung j
        others_idle = q[j0:] / max(r[k], EPS)
        c[k] = float(np.dot(w / s, 1.0 - np.clip(others_idle, 0.0, 1.0)))
    return np.clip(c, 0.0, 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# Giai he diem co dinh (3)-(5)
# ---------------------------------------------------------------------------


def solve_link(params: Sequence[EDCAParams], n_sta: Sequence[int],
               tol: float = 1e-11, max_iter: int = 300,
               damping: float = 0.5) -> LinkState:
    """Giai he diem co dinh cho cac AC dang hoat dong tren MOT link.

    params : danh sach EDCAParams cua cac AC tren link (theo thu tu goc)
    n_sta  : so tram tuong ung
    """
    aifs = [pp.aifs_us for pp in params]
    zm = build_zone_model(aifs)
    o = zm.order                                     # sap xep theo AIFS tang dan

    n = np.array([n_sta[i] for i in o], dtype=float)
    cw_min = np.array([params[i].cw_min for i in o], dtype=float)
    cw_max = np.array([max(params[i].cw_max, params[i].cw_min) for i in o], dtype=float)
    retry = np.array([params[i].retry for i in o], dtype=int)

    p = transmit_prob_vec(np.zeros(len(o)), cw_min, cw_max, retry)
    converged, it = False, 0

    for it in range(1, max_iter + 1):
        r = 1.0 - p
        q = zone_idle_prob(zm, r, n)
        pi = zone_stationary(zm, q)
        c = collision_prob(zm, r, n, q, pi)
        p_new = transmit_prob_vec(c, cw_min, cw_max, retry)
        p_next = damping * p_new + (1.0 - damping) * p
        if np.max(np.abs(p_next - p)) < tol:
            p = p_next
            converged = True
            break
        p = p_next

    r = 1.0 - p
    q = zone_idle_prob(zm, r, n)
    pi = zone_stationary(zm, q)
    c = collision_prob(zm, r, n, q, pi)
    p_loss = c ** retry                                # Eq.(6)

    return LinkState(zm=zm, n_sta=n, p=p, r=r, c=c, q=q, pi=pi,
                     p_loss=p_loss, retry=retry, cw_min=cw_min, cw_max=cw_max,
                     converged=converged, n_iter=it)


if __name__ == "__main__":
    from config import DEFAULT_EDCA, AC_SET

    st = solve_link(DEFAULT_EDCA, [a.n_sta for a in AC_SET])
    names = [AC_SET[i].name for i in st.zm.order]
    print(f"hoi tu: {st.converged} sau {st.n_iter} vong lap")
    print(f"{'AC':>5} {'AIFS':>7} {'p':>9} {'c':>9} {'P_loss':>11}")
    for k, nm in enumerate(names):
        print(f"{nm:>5} {st.zm.aifs_us[k]:7.0f} {st.p[k]:9.5f} {st.c[k]:9.5f} "
              f"{st.p_loss[k]:11.3e}")

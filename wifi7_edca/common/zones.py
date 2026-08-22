"""
common/zones.py -- Mo hinh vung AIFS (AIFS zone model), Sec. II + III.A.

Tai hien:
  Eq.(1)  phi_j = max{k | h_k < j-1}   -- tap AC du dieu kien tranh chap o slot j
  Eq.(2)  h_k   = (AIFS_k - AIFS_1)/sigma
  Eq.(3)  pi_k  -- xac suat dung cua he o vung k

Y tuong: sau khoang AIFS ngan nhat (AIFS_1), thoi gian backoff duoc chia thanh
J "vung". Vung j gom cac slot ma DUNG mot tap AC nhat dinh duoc phep dem lui.
AC co AIFS lon hon phai cho them (h_k slot) nen chi tham gia tu vung Z_k^0.

Cai dat o day tong quat hon paper mot chut: cac AC co AIFS TRUNG NHAU duoc gop
vao cung mot vung (paper ngam gia dinh moi AC mot AIFS rieng, khi do J = I va
Z_j = j).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from config import SIGMA_US

EPS = 1e-300


@dataclass
class ZoneModel:
    """Cau truc vung AIFS cua MOT link.

    Thu tu cac AC o day la thu tu DA SAP XEP tang dan theo AIFS.
    """

    order: np.ndarray          # (I,) chi so AC goc, sap xep theo AIFS tang dan
    h: np.ndarray              # (I,) h_k -- do lech slot tuong doi, Eq.(2)
    offsets: np.ndarray        # (J,) cac gia tri h phan biet (bien vung)
    zone_of_ac: np.ndarray     # (I,) Z_k^0 -- vung dau tien AC k duoc tranh chap
    z_count: np.ndarray        # (J,) Z_j -- so AC du dieu kien trong vung j
    width: np.ndarray          # (J,) do rong vung (so slot); vung cuoi = inf
    aifs_us: np.ndarray        # (I,) AIFS cua tung AC (da sap xep)

    @property
    def n_ac(self) -> int:
        return len(self.h)

    @property
    def n_zone(self) -> int:
        return len(self.offsets)


def build_zone_model(aifs_us: Sequence[float]) -> ZoneModel:
    """Dung cau truc vung tu danh sach AIFS cua cac AC dang hoat dong."""
    aifs = np.asarray(aifs_us, dtype=float)
    order = np.argsort(aifs, kind="stable")
    a_sorted = aifs[order]

    h = np.rint((a_sorted - a_sorted[0]) / SIGMA_US).astype(int)   # Eq.(2)
    offsets = np.unique(h)
    zone_of_ac = np.searchsorted(offsets, h)                       # Z_k^0 (0-based)
    z_count = np.array([int(np.sum(h <= d)) for d in offsets])     # Z_j

    width = np.empty(len(offsets), dtype=float)
    width[:-1] = np.diff(offsets)
    width[-1] = np.inf                                             # vung cuoi mo
    return ZoneModel(order=order, h=h, offsets=offsets,
                     zone_of_ac=zone_of_ac, z_count=z_count,
                     width=width, aifs_us=a_sorted)


# ---------------------------------------------------------------------------
# Xac suat kenh ranh trong tung vung / tung slot
# ---------------------------------------------------------------------------


def zone_idle_prob(zm: ZoneModel, r: np.ndarray, n_sta: np.ndarray) -> np.ndarray:
    """q_j = prod_{i=1..Z_j} r_i^{n_i} -- khong tram nao trong vung j phat.

    r, n_sta duoc cho theo THU TU DA SAP XEP cua zone model.
    """
    log_r = np.log(np.clip(r, EPS, 1.0)) * n_sta
    cum = np.cumsum(log_r)                       # cum[k] = sum_{i<=k} n_i log r_i
    return np.exp(cum[zm.z_count - 1])


def slot_idle_prob(zm: ZoneModel, r: np.ndarray, n_sta: np.ndarray,
                   n_slots: int) -> np.ndarray:
    """Xac suat kenh ranh tai slot ell = 1..n_slots (Eq. 1: tap phi_ell).

    Tai slot ell, cac AC co h_k <= ell-1 da duoc phep tranh chap.
    """
    out = np.empty(n_slots)
    log_r = np.log(np.clip(r, EPS, 1.0)) * n_sta
    for i, ell in enumerate(range(1, n_slots + 1)):
        active = zm.h <= (ell - 1)
        out[i] = np.exp(np.sum(log_r[active]))
    return out


def active_mask_at_slot(zm: ZoneModel, ell: int) -> np.ndarray:
    """Mat na phi_ell -- cac AC du dieu kien tranh chap tai slot ell."""
    return zm.h <= (ell - 1)


# ---------------------------------------------------------------------------
# Eq. (3): xac suat dung cua tung vung
# ---------------------------------------------------------------------------


def zone_stationary(zm: ZoneModel, q: np.ndarray) -> np.ndarray:
    """pi_j -- xac suat he dang o vung j.

    Ky vong so slot nam trong vung j trong mot chu ky backoff:
        alpha_{j-1} * sum_{t=0}^{Delta_j - 1} q_j^t
    voi alpha_j = prod_{l<=j} q_l^{Delta_l} (xac suat moi slot cua cac vung
    truoc deu ranh). Vung cuoi cung mo vo han -> tong = 1/(1-q_J).
    """
    n_zone = zm.n_zone
    weight = np.zeros(n_zone)
    alpha = 1.0
    for j in range(n_zone):
        qj = min(max(q[j], 0.0), 1.0 - 1e-15)
        if np.isinf(zm.width[j]):                       # vung cuoi
            weight[j] = alpha / (1.0 - qj)
        else:
            d = int(zm.width[j])
            weight[j] = alpha * (1.0 - qj ** d) / (1.0 - qj) if d > 0 else 0.0
            alpha *= qj ** d
    total = weight.sum()
    return weight / total if total > 0 else np.full(n_zone, 1.0 / n_zone)


if __name__ == "__main__":
    zm = build_zone_model([30.0, 50.0, 50.0, 90.0])
    print("h        =", zm.h)
    print("offsets  =", zm.offsets)
    print("Z_j      =", zm.z_count)
    print("Z_k^0    =", zm.zone_of_ac)
    print("do rong  =", zm.width)
    r = np.array([0.9, 0.92, 0.95, 0.97])
    n = np.array([2, 4, 4, 3])
    q = zone_idle_prob(zm, r, n)
    print("q_j      =", np.round(q, 4))
    print("pi_j     =", np.round(zone_stationary(zm, q), 4), "tong =",
          round(zone_stationary(zm, q).sum(), 12))

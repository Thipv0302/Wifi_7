"""
common/timing.py -- Cac tham so thoi gian cua co che EDCA RTS/CTS (Sec. III.B).

  Eq.(7)  T_DATA,k = T_PHY_H + (L_MAC_H + L_k) / r_data
  Eq.(8)  N_k      = max{ floor(TXOP_k / Delta_k), 1 },  Delta_k = T_DATA,k + T_ACK + 2 SIFS
  Eq.(9)  T_k^C    = T_RTS + AIFS_k
          T_k^S    = T_RTS + T_CTS + N_k Delta_k + SIFS + AIFS_k

Moi thoi luong deu tinh bang micro-giay; ham `to_slots` quy doi sang bo so
nguyen lan `delta` de dung lam so mu trong ham sinh (Eq. 10-15).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (DELTA_US, EDCAParams, L_ACK, L_CTS, L_MAC_H, L_RTS, R_CTRL,
                    R_DATA, SIGMA_US, T_PHY_H_US, T_SIFS_US)


def _bits_to_us(bits: int, rate_bps: float) -> float:
    return bits / rate_bps * 1e6


# --- Thoi luong khung dieu khien (khong phu thuoc AC) ----------------------
T_RTS_US = T_PHY_H_US + _bits_to_us(L_RTS, R_CTRL)
T_CTS_US = T_PHY_H_US + _bits_to_us(L_CTS, R_CTRL)
T_ACK_US = T_PHY_H_US + _bits_to_us(L_ACK, R_CTRL)


@dataclass(frozen=True)
class ACTiming:
    """Thoi luong dac trung cua mot AC."""

    t_data_us: float      # Eq.(7)
    delta_us: float       # Delta_k
    n_pkt: int            # Eq.(8) N_k -- so goi trong mot TXOP
    t_c_us: float         # Eq.(9) T_k^C
    t_s_us: float         # Eq.(9) T_k^S
    aifs_us: float

    # Cac thanh phan "airtime thuan" (da bo phan AIFS) dung trong Eq.(15):
    @property
    def air_collision_us(self) -> float:
        """T_k^C - AIFS_k = T_RTS -- thoi gian mat khi RTS va cham."""
        return self.t_c_us - self.aifs_us

    @property
    def air_success_us(self) -> float:
        """T_k^S - AIFS_k - SIFS = T_RTS + T_CTS + N_k Delta_k."""
        return self.t_s_us - self.aifs_us - T_SIFS_US


def ac_timing(params: EDCAParams, payload_bytes: int) -> ACTiming:
    """Tinh toan bo thoi luong cua mot AC tu tham so EDCA va kich thuoc goi."""
    t_data = T_PHY_H_US + _bits_to_us(L_MAC_H + payload_bytes * 8, R_DATA)  # Eq.(7)
    delta_k = t_data + T_ACK_US + 2 * T_SIFS_US
    n_pkt = max(int(np.floor(params.txop_us / delta_k)), 1)                 # Eq.(8)
    aifs = params.aifs_us
    t_c = T_RTS_US + aifs                                                   # Eq.(9)
    t_s = T_RTS_US + T_CTS_US + n_pkt * delta_k + T_SIFS_US + aifs          # Eq.(9)
    return ACTiming(t_data_us=t_data, delta_us=delta_k, n_pkt=n_pkt,
                    t_c_us=t_c, t_s_us=t_s, aifs_us=aifs)


def to_slots(duration_us: float, delta_us: float = DELTA_US) -> int:
    """Quy doi thoi luong sang so nguyen lan delta (so mu trong ham sinh).

    Ham sinh Eq.(10)-(15) doi hoi so mu NGUYEN de z^a khong bi da tri tren mat
    phang phuc; vi vay moi thoi luong duoc lam tron ve boi cua delta (10 us),
    sai so toi da 5 us tren moi su kien.
    """
    return int(round(duration_us / delta_us))


SLOT_SIGMA = to_slots(SIGMA_US)     # sigma / delta = 2
SLOT_SIFS = to_slots(T_SIFS_US)     # SIFS  / delta = 1
SLOT_RTS = to_slots(T_RTS_US)


if __name__ == "__main__":
    from config import AC_SET, DEFAULT_EDCA

    print(f"T_RTS = {T_RTS_US:.1f} us | T_CTS = {T_CTS_US:.1f} us | "
          f"T_ACK = {T_ACK_US:.1f} us")
    print(f"{'AC':>5} {'L(B)':>6} {'T_DATA':>9} {'Delta':>9} {'N_k':>5} "
          f"{'T^C':>9} {'T^S':>10}")
    for ac, pp in zip(AC_SET, DEFAULT_EDCA):
        t = ac_timing(pp, ac.payload_bytes)
        print(f"{ac.name:>5} {ac.payload_bytes:6d} {t.t_data_us:9.1f} "
              f"{t.delta_us:9.1f} {t.n_pkt:5d} {t.t_c_us:9.1f} {t.t_s_us:10.1f}")

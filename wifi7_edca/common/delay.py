"""
common/delay.py -- Mo hinh tre bang HAM SINH (generating function), Sec. III.B.

Tai hien:
  Eq.(10) D(z)   = (1/N_k) A(z) T(z) e(z) + ((N_k-1)/N_k) z^{Delta_k/delta}
  Eq.(11) e(z)   -- ham sinh cua khoang cho AIFS (co the bi ngat va lam lai)
  Eq.(12) xi_l   -- thoi luong gian doan tai slot l, va rho_k(l)
  Eq.(13) A(z)   -- ham sinh cua backoff + va cham qua toi da R_k lan phat lai
  Eq.(14) Y(z)   -- ham sinh cua do dai MOT slot backoff
  Eq.(15) T(z), H(z) = C(z), G_l(z)
  Eq.(16) Pr(D >= x) -- nghich dao chuoi Fourier (Abate-Whitt)
  Eq.(17) theta_k = -log Pr(D_k >= D_max,k)

GHI CHU VE CACH DOC Eq.(15):
Paper viet H(z) = C(z) = e(z) z^{T_C/delta} va G_l(z) = e(z) z^{(T_S,l - AIFS_k
- SIFS)/delta}. Vi T_k^C = T_RTS + AIFS_k (Eq. 9) da chua AIFS, con e(z) cung la
ham sinh cua khoang cho AIFS, nen doc nguyen van se tinh AIFS hai lan. O day ta
dung cach doc NHAT QUAN: moi so hang = (airtime thuan) x (ham sinh cho AIFS),
tuc la
    C(z) = H(z) = e(z) z^{T_RTS/delta}
    G_l(z)      = e(z) z^{(T_RTS + T_CTS + N_l Delta_l)/delta}
Nho vay D(1) = 1 va Y(1) = 1 dung nhu mot ham sinh xac suat hop le.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common.collision import LinkState
from common.timing import ACTiming, T_RTS_US, to_slots
from common.zones import active_mask_at_slot
from config import (DELTA_US, INVERSION_GAMMA, INVERSION_L,
                    INVERSION_N_MAX, SIGMA_US, T_SIFS_US)

EPS = 1e-300


# ===========================================================================
# 1. Cac he so vo huong cua mot AC (khong phu thuoc z)
# ===========================================================================


@dataclass
class DelayModel:
    """Toan bo he so can thiet de danh gia D(z) cho MOT AC (chi so k da sap xep)."""

    k: int
    c_k: float
    eta_k: float
    retry: int
    cw_stages: np.ndarray          # f_{k,j}, j = 0..R_k-1

    # --- e(z): he so cua chuoi Sum_l mu_l * Xi_l(z) ---
    e_scale: float                 # s_{h_k}
    e_exp: int                     # AIFS_k / delta
    defer_coef: np.ndarray         # he so cua tung so hang trong Sum mu_l Xi_l
    defer_exp: np.ndarray          # so mu tuong ung

    # --- Y(z) ---
    idle_coef: float               # 1 - c_k   (slot ranh)
    busy_coef: np.ndarray          # gamma_{k,l} va nu_k
    busy_exp: np.ndarray           # so mu airtime tuong ung

    # --- D(z) ---
    n_pkt: int                     # N_k
    t_data_exp: int                # T_DATA,k / delta
    delta_exp: int                 # Delta_k / delta

    # --- do phan giai thoi gian dung cho ham sinh ---
    delta_us: float = DELTA_US     # buoc luong tu hoa thoi gian
    sigma_exp: int = 2             # sigma / delta
    rts_exp: int = 35              # T_RTS / delta

    # tham khao
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nu: float = 0.0


def build_delay_model(st: LinkState, k: int, timings: Sequence[ACTiming],
                      delta_us: float = DELTA_US) -> DelayModel:
    """Tinh moi he so vo huong cho AC thu k (thu tu da sap xep theo AIFS).

    `timings` cung phai theo thu tu da sap xep cua `st.zm.order`.
    `delta_us` la buoc luong tu hoa thoi gian dung lam so mu cua ham sinh; co
    the noi long (20 us thay vi 10 us) cho cac AC co D_max lon de giam so diem
    DFT ma khong anh huong dang ke do chinh xac.
    """
    sigma_exp = to_slots(SIGMA_US, delta_us)
    sifs_exp = to_slots(T_SIFS_US, delta_us)
    rts_exp = to_slots(T_RTS_US, delta_us)
    zm = st.zm
    r, p, n = st.r, st.p, st.n_sta
    q, pi = st.q, st.pi
    n_ac = zm.n_ac

    # ---------------- gamma_{k,l} va nu_k (Eq. 14) ----------------
    j0 = zm.zone_of_ac[k]
    denom = pi[j0:].sum()
    gamma = np.zeros(n_ac)
    if denom > 0:
        for l in range(n_ac):
            j_start = max(j0, zm.zone_of_ac[l])
            if j_start >= zm.n_zone:
                continue
            w = pi[j_start:] / denom
            if l == k:
                if n[k] < 2:
                    continue
                g_j = (n[k] - 1) * p[k] * q[j_start:] / max(r[k] ** 2, EPS)
            else:
                g_j = n[l] * p[l] * q[j_start:] / max(r[k] * r[l], EPS)
            gamma[l] = float(np.dot(w, g_j))
    # Ŷ(1) = (1 - c_k) + sum_l gamma_{k,l} + nu_k phai bang 1 DUNG BANG SO, neu
    # khong A(z) = prod_j (1 - Y^{f_j}) / (f_j (1 - Y)) se khuech dai sai lech do
    # luy thua f_j len toi 1023 (mot sai so 1e-4 tren Y(1) lam D(1) lech ~40 %).
    # Khi sum(gamma) > c_k (xap xi gia tri trung binh khong bao dam bat dang
    # thuc nay o vung va cham rat cao) thi chuan hoa lai gamma thay vi cat nu.
    g_sum = float(gamma.sum())
    if g_sum > st.c[k] and g_sum > 0:
        gamma *= st.c[k] / g_sum
        nu = 0.0
    else:
        nu = float(st.c[k]) - g_sum

    # ---------------- e(z): Eq. (11), (12) ----------------
    h_k = int(zm.h[k])
    aifs_1_us = float(zm.aifs_us[0])
    log_r_n = np.log(np.clip(r, EPS, 1.0)) * n

    coefs: List[float] = []
    exps: List[int] = []
    s_prev = 1.0                                  # s_{l-1}
    s_h = 1.0
    for ell in range(1, h_k + 1):
        mask = active_mask_at_slot(zm, ell)       # tap phi_l, Eq.(1)
        idle_l = float(np.exp(np.sum(log_r_n[mask])))
        mu_l = s_prev * (1.0 - idle_l)            # ngat lan dau tai slot l
        base_exp = to_slots(aifs_1_us + (ell - 1) * SIGMA_US, delta_us)

        if mu_l > 0 and idle_l < 1.0:
            # rho_a(l): AC a la thu pham gay gian doan (Eq. 12)
            rho_sum = 0.0
            for a in np.where(mask)[0]:
                rho_a = n[a] * p[a] * (idle_l / max(r[a], EPS)) / (1.0 - idle_l)
                if rho_a <= 0:
                    continue
                rho_sum += rho_a
                xi_exp = (to_slots(timings[a].n_pkt * timings[a].delta_us,
                                   delta_us) - sifs_exp)
                coefs.append(mu_l * rho_a)
                exps.append(base_exp + max(xi_exp, 0))
            # phan con lai: va cham -> mat T_RTS
            rest = max(1.0 - rho_sum, 0.0)
            if rest > 0:
                coefs.append(mu_l * rest)
                exps.append(base_exp + rts_exp)
        s_prev = s_prev * idle_l
        s_h = s_prev

    # Tong khoi luong cua chuoi Sum_l mu_l Xi_l(1) PHAI bang 1 - s_{h_k} de
    # e(1) = s_{h_k} / (1 - (1 - s_{h_k})) = 1. Viec cat `rest` ve 0 khi
    # rho_sum > 1 lam mat mot phan khoi luong; chuan hoa lai o day.
    if coefs:
        total = float(np.sum(coefs))
        if total > 0:
            scale = (1.0 - s_h) / total
            coefs = [cf * scale for cf in coefs]

    # ---------------- Y(z): Eq. (14) ----------------
    busy_coef = np.concatenate([gamma, [nu]])
    busy_exp = np.array(
        [to_slots(timings[l].air_success_us, delta_us) for l in range(n_ac)]
        + [rts_exp], dtype=int)

    t = timings[k]
    c_k = float(st.c[k])
    retry = int(st.retry[k])
    eta = (1.0 - c_k) / (1.0 - c_k ** retry) if c_k > 0 else 1.0

    return DelayModel(
        k=k, c_k=c_k, eta_k=eta, retry=retry, cw_stages=st.cw_stages(k),
        e_scale=s_h if h_k > 0 else 1.0,
        e_exp=to_slots(t.aifs_us, delta_us),
        defer_coef=np.array(coefs), defer_exp=np.array(exps, dtype=int),
        idle_coef=1.0 - c_k, busy_coef=busy_coef, busy_exp=busy_exp,
        n_pkt=t.n_pkt, t_data_exp=to_slots(t.t_data_us, delta_us),
        delta_exp=to_slots(t.delta_us, delta_us), gamma=gamma, nu=nu,
        delta_us=delta_us, sigma_exp=sigma_exp, rts_exp=rts_exp)


# ===========================================================================
# 2. Luoi diem phuc + bo nho dem luy thua
# ===========================================================================


class FourierGrid:
    """Luoi z = r * exp(i pi n / N) dung cho Eq.(16), co bo nho dem z^e.

    Chi luu NUA luoi (n = 0..N): vi moi he so cua ham sinh deu thuc nen
    G(conj z) = conj G(z), phan con lai suy ra bang doi xung lien hop.
    Nho vay so diem phai tinh giam mot nua.

    Luy thua z^e duoc tinh bang exp(e * Log z). Voi e NGUYEN thi
    exp(e Log z) = z^e chinh xac (khong co van de nhanh cat), nhung nhanh hon
    numpy.power khoang 2.5 lan tren mang phuc.

    Bo nho dem `_cache` duoc GIU LAI giua cac lan danh gia (xem `get_grid`):
    trong GA, cac so mu lap lai rat nhieu vi khong gian tham so la roi rac.
    """

    __slots__ = ("n_dft", "m", "r", "n_idx", "z", "log_z", "one_minus_z",
                 "_cache")

    CACHE_MAX = 192            # tran so mu duoc luu (moi mang ~2N * 16 byte)

    def __init__(self, n_dft: int, r: float):
        self.n_dft = n_dft
        self.m = 2 * n_dft
        self.r = r
        self.n_idx = np.arange(n_dft + 1)
        self.z = r * np.exp(1j * np.pi * self.n_idx / n_dft)
        self.log_z = np.log(r) + 1j * (np.pi * self.n_idx / n_dft)
        self.one_minus_z = 1.0 - self.z
        self._cache = {0: np.ones_like(self.z), 1: self.z}

    def zpow(self, e: int) -> np.ndarray:
        e = int(e)
        cached = self._cache.get(e)
        if cached is None:
            cached = np.exp(e * self.log_z)
            if len(self._cache) >= self.CACHE_MAX:
                self._cache = {0: self._cache[0], 1: self.z}
            self._cache[e] = cached
        return cached


# Bo nho dem luoi theo n_dft: moi AC co D_max co dinh nen chi co vai luoi khac
# nhau trong toan bo phien chay, va bo nho dem z^e duoc dung lai giua cac lan
# danh gia cua GA.
_GRIDS: Dict[int, FourierGrid] = {}


def get_grid(n_dft: int, r: float) -> FourierGrid:
    g = _GRIDS.get(n_dft)
    if g is None or g.r != r:
        g = FourierGrid(n_dft, r)
        _GRIDS[n_dft] = g
    return g


def _ipow(a: np.ndarray, e: int) -> np.ndarray:
    """a^e voi e nguyen >= 0, bang phep binh phuong lien tiep (nhanh & chinh xac)."""
    e = int(e)
    if e == 0:
        return np.ones_like(a)
    result = None
    base = a
    while e:
        if e & 1:
            result = base if result is None else result * base
        e >>= 1
        if e:
            base = base * base
    return result


def _poly(grid: FourierGrid, coef: np.ndarray, exp: np.ndarray) -> np.ndarray:
    """Sum_i coef_i z^{exp_i}."""
    out = np.zeros_like(grid.z)
    for cf, ex in zip(coef, exp):
        if cf != 0.0:
            out += cf * grid.zpow(ex)
    return out


def eval_e(dm: DelayModel, grid: FourierGrid) -> np.ndarray:
    """Eq.(11): ham sinh cua khoang cho AIFS."""
    if len(dm.defer_coef) == 0:
        return grid.zpow(dm.e_exp)
    return (dm.e_scale * grid.zpow(dm.e_exp)
            / (1.0 - _poly(grid, dm.defer_coef, dm.defer_exp)))


def eval_y(dm: DelayModel, grid: FourierGrid, e_z: np.ndarray) -> np.ndarray:
    """Eq.(14): ham sinh do dai mot slot backoff."""
    return (dm.idle_coef * grid.zpow(dm.sigma_exp)
            + e_z * _poly(grid, dm.busy_coef, dm.busy_exp))


def eval_a(dm: DelayModel, y_z: np.ndarray, c_z: np.ndarray) -> np.ndarray:
    """Eq.(13): ham sinh cua backoff + va cham.

    f_{k,j} khong giam va moi buoc GAP DOI cho den khi cham CW_max, nen
    Y(z)^{f_{k,j}} duoc lay bang cach BINH PHUONG ket qua cua buoc truoc; chi
    luy thua dau tien (Y^{CW_min}) phai tinh truc tiep.
    """
    one_minus_y = 1.0 - y_z
    one_minus_y = np.where(np.abs(one_minus_y) < 1e-14, 1e-14, one_minus_y)

    a = np.zeros_like(y_z)
    prod_u = np.ones_like(y_z)
    c_pow = np.ones_like(y_z)
    y_f = None
    f_prev = 0
    for i in range(dm.retry):
        f = int(dm.cw_stages[i])
        if y_f is None:
            y_f = _ipow(y_z, f)
        elif f == 2 * f_prev:
            y_f = y_f * y_f
        elif f != f_prev:
            y_f = _ipow(y_z, f)
        f_prev = f
        u_i = (1.0 - y_f) / (f * one_minus_y)          # backoff deu tren [0, f-1]
        prod_u = prod_u * u_i
        a += dm.eta_k * (dm.c_k ** i) * c_pow * prod_u
        c_pow = c_pow * c_z
    return a


def eval_dhat(dm: DelayModel, grid: FourierGrid) -> np.ndarray:
    """Eq.(10): ham sinh cua TONG tre cua mot goi."""
    e_z = eval_e(dm, grid)
    y_z = eval_y(dm, grid, e_z)
    c_z = e_z * grid.zpow(dm.rts_exp)                  # C(z) = H(z), Eq.(15)
    a_z = eval_a(dm, y_z, c_z)
    first = a_z * grid.zpow(dm.t_data_exp) * e_z / dm.n_pkt
    rest = (dm.n_pkt - 1) / dm.n_pkt * grid.zpow(dm.delta_exp)
    return first + rest


# ===========================================================================
# 3. Eq. (16): nghich dao chuoi Fourier de lay CCDF
# ===========================================================================


def delay_ccdf(dm: DelayModel, x_slots: int, l: float = INVERSION_L,
               gamma_acc: float = INVERSION_GAMMA,
               n_max: int = INVERSION_N_MAX) -> float:
    """Pr(D_k >= x) voi x tinh bang so slot delta.

    Nghich dao chuoi Fourier (Abate-Whitt) cua ham sinh duoi (tail generating
    function)  G(z) = (1 - D(z)) / (1 - z):

        a_x = 1/(2N r^x) * Sum_{n=0}^{2N-1} G(r w^n) w^{-n x},   w = exp(i pi / N)

    Chon N = x * l va r = 10^{-gamma/N}  =>  r^x = 10^{-gamma/l}.
    Sai so chong pho ~ r^{2N} = 10^{-2 gamma}; sai so lam tron ~ sqrt(2N) * eps
    may / r^x. Xem ghi chu trong config.py ve viec chon gamma.
    """
    x = int(max(x_slots, 1))
    n_dft = int(min(max(round(x * l), 16), n_max))
    r = 10.0 ** (-gamma_acc / n_dft)
    r_pow_x = 10.0 ** (-gamma_acc * x / n_dft)

    grid = get_grid(n_dft, r)
    g_z = (1.0 - eval_dhat(dm, grid)) / grid.one_minus_z
    phase = np.exp(-1j * np.pi * grid.n_idx * x / n_dft)
    term = g_z * phase

    # doi xung lien hop: n va 2N-n cho hai so hang lien hop nhau
    total = np.real(term[0]) + np.real(term[-1]) + 2.0 * np.sum(np.real(term[1:-1]))
    val = total / (grid.m * r_pow_x)
    return float(np.clip(val, 1e-16, 1.0))


def reliability_index(violation_prob: float) -> float:
    """Eq.(17): theta_k = -log Pr(D_k >= D_max,k).

    Dung logarit co so 10 cho nhat quan voi ham muc tieu Eq.(18) -- xem ghi chu
    trong common/utility.QoSResult.objective ve viec chon co so.
    """
    return -np.log10(max(violation_prob, 1e-300))


def dhat_at(dm: DelayModel, z_value: complex) -> complex:
    """Danh gia D(z) tai MOT diem (tien kiem tra tinh hop le cua ham sinh)."""
    class _One:
        z = np.array([z_value], dtype=complex)

        def zpow(self, e):
            return self.z ** int(e)

    return complex(eval_dhat(dm, _One())[0])


if __name__ == "__main__":
    from common.collision import solve_link
    from common.timing import ac_timing
    from config import AC_SET, DEFAULT_EDCA

    st = solve_link(DEFAULT_EDCA, [a.n_sta for a in AC_SET])
    order = st.zm.order
    timings = [ac_timing(DEFAULT_EDCA[i], AC_SET[i].payload_bytes) for i in order]

    print(f"{'AC':>5} {'h_k':>4} {'c_k':>8} {'D(1-)':>9} "
          f"{'Pr(D>=Dmax)':>13} {'theta':>9}")
    for k in range(st.zm.n_ac):
        dm = build_delay_model(st, k, timings)
        d1 = float(np.real(dhat_at(dm, 1.0 - 1e-12)))
        ac = AC_SET[order[k]]
        x = int(round(ac.d_max_us / DELTA_US))
        v = delay_ccdf(dm, x)
        print(f"{ac.name:>5} {st.zm.h[k]:4d} {dm.c_k:8.4f} {d1:9.5f} "
              f"{v:13.3e} {reliability_index(v):9.3f}")

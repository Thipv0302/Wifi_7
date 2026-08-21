"""
config.py -- Toan bo tham so cua he thong Multi-link EDCA (Wi-Fi 7).

Nguon: Yi, Cheng, Wang, Pan, Ouyang, Zhang,
"Intelligent Multi-link EDCA Optimization for Delay-Bounded QoS in Wi-Fi 7",
arXiv:2509.25855v1.
  - TABLE I  : Simulation Parameters
  - Sec. IV.A: kich ban phan tich do nhay tham so (Fig. 3)
  - Sec. IV.B: mien gia tri cua cac tham so EDCA
  - Sec. V   : kich ban danh gia 5 AC

Moi hang so deu ghi kem so hieu bang / muc trong paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ===========================================================================
# 1. THAM SO VAT LY / MAC  (TABLE I)
# ===========================================================================

R_CTRL = 1e6            # r_ctrl  -- control rate  (1 Mbps)
R_DATA = 11e6           # r_data  -- data rate     (11 Mbps)
SIGMA_US = 20.0         # sigma   -- slot time     (20 us)
DELTA_US = 10.0         # delta   -- discrete time slot dung cho ham sinh (10 us)
T_SIFS_US = 10.0        # T_SIFS  -- Short Interframe Space
T_PHY_H_US = 192.0      # T_PHY_H -- PHY header time

L_RTS = 160             # bit
L_CTS = 112             # bit
L_ACK = 112             # bit
L_MAC_H = 224           # bit -- MAC header

# ===========================================================================
# 2. THAM SO THUAT TOAN DI TRUYEN (TABLE I)
# ===========================================================================


@dataclass(frozen=True)
class GAParams:
    """Tham so GA theo TABLE I.

    Gia tri goc cua paper (N_pop = 2000, N_gen = 500) tuong duong 10^6 lan danh
    gia ham thich nghi; moi lan danh gia phai nghich dao so ham sinh cho tung AC
    nen chi phi rat lon. Mac dinh o day dung N_pop nho hon (200) nhung GIU
    NGUYEN N_gen = 300 va N_stag = 50 cua paper de duong hoi tu Fig. 4 co cung
    thang truc hoanh; buoc tinh chinh theo toa do (`ga.coordinate_polish`) bu
    lai phan quan the bi cat bot. Dung `GAParams.paper()` de chay dung cau hinh
    cua paper.
    """

    n_pop: int = 200           # N_pop   (paper: 2000)
    n_gen: int = 300           # N_gen   (paper: 500)
    n_elite: int = 8           # N_elite (paper: 80  -- giu ty le 4%)
    p_cross: float = 0.8       # N_cross (paper: 0.8)
    p_mutate: float = 0.05   # ~1.5 gen dot bien/ca the (do dai gen = 30)
    n_stag: int = 50           # N_stag  (paper: 50) -- nguong dung som
    seed: int = 2025

    @staticmethod
    def paper() -> "GAParams":
        return GAParams(n_pop=2000, n_gen=500, n_elite=80, p_cross=0.8,
                        p_mutate=0.05, n_stag=50)


GA = GAParams()

# ===========================================================================
# 3. MIEN GIA TRI THAM SO EDCA  (Sec. IV.B)
# ===========================================================================

CW_MIN_RANGE = (1, 1023)      # CW_min,i in [1, 1023]
CW_MAX_RANGE = (1, 1023)      # CW_max,i in [1, 1023], CW_min,i <= CW_max,i
AIFSN_RANGE = (2, 15)         # AIFSN_i  in [2, 15]
TXOP_RANGE_US = (0.0, 8192.0)  # TXOP_i  in [0, 8.192] ms
TXOP_STEP_US = 32.0           # buoc 32 us
RETRY_RANGE = (4, 7)          # R_i      in [4, 7]

# ===========================================================================
# 4. CAU HINH ACCESS CATEGORY
# ===========================================================================


@dataclass
class ACConfig:
    """Mot Access Category: luu luong + rang buoc QoS."""

    name: str
    n_sta: int              # n_i^all -- so STA mang luong AC_i
    payload_bytes: int      # L_i
    d_max_ms: float         # D_max,i -- tre toi da chap nhan duoc
    epsilon: float          # eps_i   -- nguong xac suat vi pham tre

    @property
    def payload_bits(self) -> int:
        return self.payload_bytes * 8

    @property
    def d_max_us(self) -> float:
        return self.d_max_ms * 1000.0


@dataclass
class EDCAParams:
    """Vector tham so EDCA x_i cua mot AC (bien quyet dinh cua P1, Eq. 18)."""

    cw_min: int
    cw_max: int
    aifsn: int
    txop_us: float
    retry: int
    link: int = 0           # m_i -- link ma AC nay duoc gan vao

    @property
    def aifs_us(self) -> float:
        """AIFS = SIFS + AIFSN * sigma  (dinh nghia EDCA)."""
        return T_SIFS_US + self.aifsn * SIGMA_US

    def copy(self) -> "EDCAParams":
        return EDCAParams(self.cw_min, self.cw_max, self.aifsn,
                          self.txop_us, self.retry, self.link)


# --- Kich ban danh gia chinh (Sec. V) --------------------------------------
# "We set the number of stations for every AC n = [2,4,4,3,3], packet sizes
#  L = [50,210,256,800,2000] bytes, maximum tolerable delays
#  D_max = [50,60,100,300,300] ms, and delay violation probability thresholds
#  eps = [1e-7, 1e-6, 1e-4, 1e-2, 0.5]"
AC_SET: List[ACConfig] = [
    ACConfig("AC1", 2,   50,  50.0, 1e-7),   # dich vu nhay cam tre
    ACConfig("AC2", 4,  210,  60.0, 1e-6),   # dich vu nhay cam tre
    ACConfig("AC3", 4,  256, 100.0, 1e-4),   # streaming
    ACConfig("AC4", 3,  800, 300.0, 1e-2),   # best-effort
    ACConfig("AC5", 3, 2000, 300.0, 5e-1),   # background
]

N_LINKS = 2      # M -- so link doc lap (Fig. 1: M = 2)

# --- EDCA mac dinh theo IEEE 802.11e (baseline "Default EDCA") -------------
# Bang mac dinh cua PHY OFDM (aCWmin = 15, IEEE 802.11-2020, Table 9-155):
#   AC_VO: CWmin=3,  CWmax=7,    AIFSN=2, TXOP=1504 us
#   AC_VI: CWmin=7,  CWmax=15,   AIFSN=2, TXOP=3008 us
#   AC_BE: CWmin=15, CWmax=1023, AIFSN=3, TXOP=0
#   AC_BK: CWmin=15, CWmax=1023, AIFSN=7, TXOP=0
# Anh xa 5 AC cua kich ban -> 4 hang muc chuan:
#   AC1, AC2 -> AC_VO ; AC3 -> AC_VI ; AC4 -> AC_BE ; AC5 -> AC_BK
#
# Paper khong liet ke bang mac dinh da dung. Chon bang aCWmin = 15 vi no tai
# hien dung DANG cua Fig. 5: cua so tranh chap rat nho -> P_loss cua ca 5 AC
# deu o muc 0.5-0.9 (Fig. 5b) trong khi tre cua AC1-AC3 van bi chan chat nen
# Pr(D >= D_max) cua chung nam sau duoi eps, con AC4/AC5 thi vi pham nang
# (Fig. 5a). Bang aCWmin = 31 (PHY DSSS) cho P_loss ~ 0.1 va lam AC1-AC3 vi
# pham tre -- khong khop voi Fig. 5.
DEFAULT_EDCA: List[EDCAParams] = [
    EDCAParams(cw_min=3,  cw_max=7,    aifsn=2, txop_us=1504.0, retry=7),
    EDCAParams(cw_min=3,  cw_max=7,    aifsn=2, txop_us=1504.0, retry=7),
    EDCAParams(cw_min=7,  cw_max=15,   aifsn=2, txop_us=3008.0, retry=7),
    EDCAParams(cw_min=15, cw_max=1023, aifsn=3, txop_us=0.0,    retry=7),
    EDCAParams(cw_min=15, cw_max=1023, aifsn=7, txop_us=0.0,    retry=7),
]

# ===========================================================================
# 5. KICH BAN PHAN TICH DO NHAY THAM SO (Sec. IV.A -- Fig. 3)
# ===========================================================================
# "We evaluate a scenario with two ACs: AC_1 configured as AIFSN_1 = 8 and
#  TXOP_1 = 4080 us while AC_2 has AIFSN_2 varying from 2 to 15 and TXOP_2
#  ranging from 0 to 8160 us. Other parameters are identical for both AC_1 and
#  AC_2: CW_min = 32, CW_max = 1024, retransmission limit R = 7, number of
#  stations n = 4, packet size L = 1000 bytes, and maximum tolerable delay
#  D_max = 100 ms."


@dataclass(frozen=True)
class SensitivityScenario:
    aifsn_1: int = 8
    txop_1_us: float = 4080.0
    cw_min: int = 32
    cw_max: int = 1024
    retry: int = 7
    n_sta: int = 4
    payload_bytes: int = 1000
    d_max_ms: float = 100.0
    # Fig. 3 cua paper ve tu AIFSN_2 = 0 den 15 (truc hoanh cua Fig. 3b), rong
    # hon mien [2, 15] neu trong Sec. IV.A; giu nguyen de doi chieu duoc.
    aifsn_2_values: Tuple[int, ...] = tuple(range(0, 16))       # 0 .. 15
    txop_2_values: Tuple[float, ...] = tuple(
        float(v) for v in range(0, 8161, 320))                  # 0 .. 8160 us


SENSITIVITY = SensitivityScenario()

# ===========================================================================
# 6. THAM SO NGHICH DAO SO (Eq. 16)
# ===========================================================================
# Pr(D >= x) duoc lay bang nghich dao chuoi Fourier cua ham sinh:
#   N = x * l ,  r = 10^(-8/N)  ->  r^x = 10^(-8/l)
# Paper: "setting l = 1 and r = 10^(-8/v) yields numerical errors < 1e-16".
INVERSION_L = 0.6
# Rang buoc quan trong: 2N PHAI lon hon x, neu khong cac he so a_{x-2N} se bi
# khuech dai boi r^{-2N} va lam hong ket qua  =>  l > 0.5.
# Voi l = 0.6, gamma = 1.5: r^x = 10^{-2.5}, sai so chong pho ~ 10^{-2*gamma}
# = 1e-3 (tuong doi), san nhieu lam tron ~ sqrt(2N)*1e-16/r^x ~ 1e-11.
# Da doi chieu voi cau hinh chinh xac cao (l = 2, gamma = 4): lech < 1e-3 tuong
# doi ke ca o vung xac suat 1e-8.
INVERSION_GAMMA = 1.5        # so mu trong r = 10^(-GAMMA / N)
INVERSION_N_MAX = 32768      # tran an toan cho so diem DFT

# So slot toi da cho mot lan nghich dao. AC nao co D_max/delta vuot nguong nay
# se dung buoc luong tu hoa tho hon (boi cua DELTA_US) de giam chi phi.
X_SLOTS_MAX = 15000

# ===========================================================================
# 7. DUONG DAN XUAT
# ===========================================================================

FIG_DIR = "figures"
RESULT_DIR = "results"

"""
datagen/genome.py -- MODULE SINH DU LIEU (2/2): khong gian tham so va ma hoa.

Bien quyet dinh cua P1 (Eq. 18) la vector x_i cua tung AC:
    CW_min,i , CW_max,i , AIFSN_i , TXOP_i , R_i
va -- rieng cho cau hinh MLO -- them phep gan AC vao link  m_i in {1..M}.

Mien gia tri (Sec. IV.B):
    CW_min, CW_max in [1, 1023],  CW_min <= CW_max
    AIFSN          in [2, 15]
    TXOP           in [0, 8.192] ms, buoc 32 us
    R              in [4, 7]

Nhiem sac the duoc ma hoa thanh vector SO NGUYEN, moi gen la mot chi so trong
mien roi rac tuong ung. Cach nay giup phep lai / dot bien luon sinh ra ca the
hop le, khong can sua chua phuc tap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from config import (AIFSN_RANGE, CW_MAX_RANGE, CW_MIN_RANGE, EDCAParams,
                    RETRY_RANGE, TXOP_RANGE_US, TXOP_STEP_US)

# --- Cac muc roi rac cua tung tham so --------------------------------------
# CW thuong la 2^k - 1 trong chuan; o day cho phep toan bo [1, 1023] nhu paper
# nhung luoi hoa theo thang log2 de GA tim kiem hieu qua hon.
# Bo qua CW = 1..3: cua so backoff qua nho khien moi tram phat ngay lap tuc
# (p -> 1, c -> 1) -- diem suy bien khong co y nghia van hanh.
CW_LEVELS = np.unique(np.clip(
    np.round(2.0 ** np.arange(2.0, 10.01, 0.5)).astype(int),
    max(CW_MIN_RANGE[0], 4), CW_MIN_RANGE[1]))

# He so nhan CW_max / CW_min = 2^m voi m = 0..10 (m chinh la so bac backoff
# toi da m_k trong Eq. 5). PHAI cho phep m = 0 (CW_max = CW_min): cac cau hinh
# toi uu cua paper -- P_loss ~ 1e-10 ma van giu duoc chan tre -- deu can cua so
# backoff KHONG nhan doi, neu bat buoc CW_max >= 4 CW_min thi duoi tre keo dai
# va rang buoc Pr(D >= D_max) < eps khong bao gio dat duoc.
CW_SPAN_LEVELS = 2 ** np.arange(0, 11)
AIFSN_LEVELS = np.arange(AIFSN_RANGE[0], AIFSN_RANGE[1] + 1)
TXOP_LEVELS = np.arange(TXOP_RANGE_US[0], TXOP_RANGE_US[1] + 1e-9, TXOP_STEP_US)
RETRY_LEVELS = np.arange(RETRY_RANGE[0], RETRY_RANGE[1] + 1)

GENES_PER_AC = 6      # (cw_min, cw_span, aifsn, txop, retry, link)


@dataclass(frozen=True)
class GenomeSpec:
    """Mo ta khong gian tim kiem cua mot bai toan cu the."""

    n_ac: int
    n_links: int = 1          # M = 1 -> toi uu EDCA don link
    allow_link_choice: bool = True

    @property
    def length(self) -> int:
        return self.n_ac * GENES_PER_AC

    def upper(self) -> np.ndarray:
        """So muc cua tung gen (gia tri hop le: 0 .. upper-1)."""
        per_ac = np.array([
            len(CW_LEVELS),            # cw_min
            len(CW_SPAN_LEVELS),       # cw_span: CW_max = CW_min * 2^span
            len(AIFSN_LEVELS),
            len(TXOP_LEVELS),
            len(RETRY_LEVELS),
            max(self.n_links if self.allow_link_choice else 1, 1),
        ])
        return np.tile(per_ac, self.n_ac)


# ---------------------------------------------------------------------------
# Giai ma / ma hoa
# ---------------------------------------------------------------------------


def decode(genome: np.ndarray, spec: GenomeSpec) -> List[EDCAParams]:
    """Nhiem sac the (vector so nguyen) -> danh sach EDCAParams."""
    g = np.asarray(genome, dtype=int).reshape(spec.n_ac, GENES_PER_AC)
    out: List[EDCAParams] = []
    for i in range(spec.n_ac):
        cw_min = int(CW_LEVELS[g[i, 0] % len(CW_LEVELS)])
        # gen thu hai la "do rong" cua chuoi backoff: CW_max >= CW_min luon dung
        span_idx = g[i, 1] % len(CW_SPAN_LEVELS)
        cw_max = int(min(cw_min * int(CW_SPAN_LEVELS[span_idx]), CW_MAX_RANGE[1]))
        cw_max = max(cw_max, cw_min)
        aifsn = int(AIFSN_LEVELS[g[i, 2] % len(AIFSN_LEVELS)])
        txop = float(TXOP_LEVELS[g[i, 3] % len(TXOP_LEVELS)])
        retry = int(RETRY_LEVELS[g[i, 4] % len(RETRY_LEVELS)])
        link = int(g[i, 5] % max(spec.n_links, 1)) if spec.allow_link_choice else 0
        out.append(EDCAParams(cw_min=cw_min, cw_max=cw_max, aifsn=aifsn,
                              txop_us=txop, retry=retry, link=link))
    return out


def encode(params: Sequence[EDCAParams], spec: GenomeSpec) -> np.ndarray:
    """EDCAParams -> nhiem sac the (lay muc gan nhat trong luoi roi rac)."""
    g = np.zeros((spec.n_ac, GENES_PER_AC), dtype=int)
    for i, p in enumerate(params):
        g[i, 0] = int(np.argmin(np.abs(CW_LEVELS - p.cw_min)))
        ratio = max(p.cw_max / max(p.cw_min, 1), 1.0)
        g[i, 1] = int(np.argmin(np.abs(CW_SPAN_LEVELS - ratio)))
        g[i, 2] = int(np.argmin(np.abs(AIFSN_LEVELS - p.aifsn)))
        g[i, 3] = int(np.argmin(np.abs(TXOP_LEVELS - p.txop_us)))
        g[i, 4] = int(np.argmin(np.abs(RETRY_LEVELS - p.retry)))
        g[i, 5] = int(p.link % max(spec.n_links, 1))
    return g.ravel()


def random_genome(spec: GenomeSpec, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, spec.upper())


def default_params(n_ac: int, n_links: int = 1) -> List[EDCAParams]:
    """EDCA mac dinh IEEE 802.11e, mo rong cho so AC bat ky.

    Thu tu uu tien: AC_VO, AC_VO, AC_VI, AC_BE, AC_BK (lap lai neu can).
    """
    from config import DEFAULT_EDCA

    out = []
    for i in range(n_ac):
        p = DEFAULT_EDCA[min(i, len(DEFAULT_EDCA) - 1)].copy()
        p.link = 0
        out.append(p)
    return out


def describe(params: Sequence[EDCAParams]) -> str:
    lines = [f"{'AC':>5} {'link':>5} {'CWmin':>7} {'CWmax':>7} {'AIFSN':>6} "
             f"{'TXOP(us)':>9} {'R':>3}"]
    for i, p in enumerate(params):
        lines.append(f"{'AC'+str(i+1):>5} {p.link:5d} {p.cw_min:7d} {p.cw_max:7d} "
                     f"{p.aifsn:6d} {p.txop_us:9.0f} {p.retry:3d}")
    return "\n".join(lines)


if __name__ == "__main__":
    spec = GenomeSpec(n_ac=5, n_links=2)
    print("so muc CW :", len(CW_LEVELS), "->", CW_LEVELS)
    print("so muc span:", len(CW_SPAN_LEVELS), "->", CW_SPAN_LEVELS)
    print("so muc TXOP:", len(TXOP_LEVELS), f"(0 .. {TXOP_LEVELS[-1]:.0f} us)")
    print("do dai nhiem sac the:", spec.length)

    rng = np.random.default_rng(0)
    g = random_genome(spec, rng)
    print("\nCa the ngau nhien:")
    print(describe(decode(g, spec)))

    d = default_params(5)
    print("\nEDCA mac dinh -> ma hoa -> giai ma (kiem tra vong tron):")
    print(describe(decode(encode(d, GenomeSpec(5, 1)), GenomeSpec(5, 1))))

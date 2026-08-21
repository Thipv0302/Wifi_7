"""
datagen/scenario.py -- MODULE SINH DU LIEU (1/2): kich ban luu luong.

Sinh cac tap Access Category (so tram, kich thuoc goi, rang buoc tre) dung lam
DAU VAO cho bai toan toi uu:

  * `main_scenario()`        -- kich ban 5 AC cua Sec. V
  * `sensitivity_scenario()` -- kich ban 2 AC cua Sec. IV.A (Fig. 3)
  * `epsilon_sweep()`        -- bien the cua kich ban chinh khi doi eps_1 (Fig. 6)
  * `random_scenario()`      -- sinh ngau nhien de kiem tra tinh on dinh
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from config import ACConfig, AC_SET, SENSITIVITY, SensitivityScenario


def main_scenario() -> List[ACConfig]:
    """Kich ban danh gia chinh (Sec. V): 5 AC voi yeu cau QoS rat khac nhau."""
    return [ACConfig(a.name, a.n_sta, a.payload_bytes, a.d_max_ms, a.epsilon)
            for a in AC_SET]


def sensitivity_scenario(sc: SensitivityScenario = SENSITIVITY) -> List[ACConfig]:
    """Kich ban do nhay tham so (Sec. IV.A): 2 AC giong het nhau ve luu luong.

    "Other parameters are identical for both AC_1 and AC_2: ... number of
     stations n = 4, packet size L = 1000 bytes, and maximum tolerable delay
     D_max = 100 ms."
    Nguong eps o day khong duoc paper dung (Fig. 3 chi ve theta va P_loss) nen
    dat bang 1.0 -- tuc la khong rang buoc.
    """
    return [ACConfig(f"AC{i+1}", sc.n_sta, sc.payload_bytes, sc.d_max_ms, 1.0)
            for i in range(2)]


def epsilon_sweep(eps1_values: Sequence[float]) -> List[Tuple[float, List[ACConfig]]]:
    """Bien the kich ban chinh khi thay doi rieng eps_1 (Fig. 6).

    "Figure 6 shows how the target reliability constraint of AC_1 (eps_1)
     influences the QoS performance."
    """
    out = []
    for e1 in eps1_values:
        acs = main_scenario()
        acs[0] = ACConfig(acs[0].name, acs[0].n_sta, acs[0].payload_bytes,
                          acs[0].d_max_ms, float(e1))
        out.append((float(e1), acs))
    return out


def random_scenario(n_ac: int = 5, seed: int = 0) -> List[ACConfig]:
    """Kich ban ngau nhien (dung de kiem tra do ben cua thuat toan)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_ac):
        out.append(ACConfig(
            name=f"AC{i+1}",
            n_sta=int(rng.integers(2, 6)),
            payload_bytes=int(rng.choice([50, 210, 256, 512, 800, 1500, 2000])),
            d_max_ms=float(rng.choice([30, 50, 60, 100, 200, 300])),
            epsilon=float(10.0 ** rng.uniform(-7, -1)),
        ))
    return out


def describe(ac_set: Sequence[ACConfig]) -> str:
    lines = [f"{'AC':>5} {'n_sta':>6} {'L (B)':>7} {'D_max (ms)':>11} {'eps':>10}"]
    for a in ac_set:
        lines.append(f"{a.name:>5} {a.n_sta:6d} {a.payload_bytes:7d} "
                     f"{a.d_max_ms:11.0f} {a.epsilon:10.1e}")
    lines.append(f"tong so tram = {sum(a.n_sta for a in ac_set)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Kich ban chinh (Sec. V) ===")
    print(describe(main_scenario()))
    print("\n=== Kich ban do nhay tham so (Sec. IV.A) ===")
    print(describe(sensitivity_scenario()))

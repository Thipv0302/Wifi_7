"""
training/baselines.py -- Cac cau hinh doi chung (Sec. V).

Paper so sanh ba cau hinh:
  1. "Default EDCA"    -- tham so EDCA mac dinh IEEE 802.11e, TAT CA AC tren
                          MOT link (khong co MLO)
  2. "Opt. EDCA"       -- GA toi uu tham so EDCA nhung van chi mot link
  3. "Opt. MLO EDCA"   -- GA toi uu DONG THOI phan bo AC vao link va tham so
                          EDCA cua tung link (de xuat cua paper)

Ngoai ra co them mot bo tim kiem ngau nhien de doi chieu chat luong cua GA.
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common.utility import QoSResult, evaluate_config, fitness
from config import ACConfig, EDCAParams, GA, GAParams
from datagen.genome import GenomeSpec, decode, default_params, random_genome


def default_edca(ac_set: Sequence[ACConfig]) -> Tuple[List[EDCAParams], QoSResult]:
    """Cau hinh 1: EDCA mac dinh, don link."""
    params = default_params(len(ac_set), n_links=1)
    return params, evaluate_config(params, ac_set)


def random_search(ac_set: Sequence[ACConfig], n_links: int, n_trials: int,
                  seed: int = 0) -> Tuple[List[EDCAParams], QoSResult, float]:
    """Tim kiem ngau nhien cung ngan sach danh gia -- moc so sanh cho GA."""
    rng = np.random.default_rng(seed)
    spec = GenomeSpec(n_ac=len(ac_set), n_links=n_links,
                      allow_link_choice=n_links > 1)
    best = None
    for _ in range(n_trials):
        params = decode(random_genome(spec, rng), spec)
        res = evaluate_config(params, ac_set)
        f = fitness(res, ac_set)
        if best is None or f > best[2]:
            best = (params, res, f)
    return best


def compare_configs(ac_set: Sequence[ACConfig],
                    configs: Dict[str, List[EDCAParams]]) -> Dict[str, QoSResult]:
    return {name: evaluate_config(p, ac_set) for name, p in configs.items()}


if __name__ == "__main__":
    from common.utility import summarize
    from datagen.scenario import main_scenario

    acs = main_scenario()
    params, res = default_edca(acs)
    print("=== Default EDCA (don link) ===")
    print(summarize(res, acs))

    t0 = time.time()
    p2, r2, f2 = random_search(acs, n_links=2, n_trials=300, seed=1)
    print(f"\n=== Tim kiem ngau nhien 300 lan, 2 link ({time.time()-t0:.0f}s) ===")
    print(summarize(r2, acs))

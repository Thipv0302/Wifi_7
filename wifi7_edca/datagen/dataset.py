"""
datagen/dataset.py -- Sinh TAP HUAN LUYEN cho GNN surrogate.

Nhan cua tap huan luyen do CHINH MO HINH GIAI TICH trong common/ sinh ra, nen
du lieu la "mien phi": khong can ns-3, khong can do dac. Chi phi = so mau x 16 ms,
chia deu cho cac nhan CPU.

Hai nhan cho moi AC:
    y1 = log10(c_i)      -- xac suat va cham (Eq. 4)
    y2 = theta_i = -log10 Pr(D_i >= D_max,i)   (Eq. 17)

Vi sao hoc log10(c) chu khong hoc thang P_loss:
    Eq.(6)  P_loss_i = c_i^{R_i}
    Eq.(18) muc tieu  = sum_i -log10(P_loss_i) = sum_i -R_i * log10(c_i)
Ham muc tieu la ham TUYEN TINH cua log10(c). Du doan log10(c) roi ap dung Eq.(6)
CHINH XAC (R_i da biet) la mot "physics-informed head": khong bao gio de mang
phai hoc phep luy thua ^R, va sai so khong bi khuech dai luy thua.

Tuong tu, hoc theta = -log10 Pr(...) chu khong hoc Pr(...): rang buoc nam o vung
1e-7..1e-4, hoi quy truc tiep xac suat o thang do do la vo vong.

CHIEN LUOC LAY MAU -- day la phan quyet dinh chat luong surrogate.

Do dac thuc te (xem README muc 6.2 va do bang tay):
    lay mau ngau nhien deu       -> 0 / 300 cau hinh kha thi
    lay mau "co cau truc"        -> 0 / 300 cau hinh kha thi
    nhieu quanh NGHIEM TOI UU    -> chi 1.5 % kha thi
Vung kha thi MONG NHU DAO CAO: AC1 va AC2 (eps = 1e-7, 1e-6) la rang buoc siet,
rieng chung chi dat 4-5 % ngay trong vung co cau truc. Neu chi lay mau ngau nhien
thi surrogate KHONG BAO GIO nhin thay vung ma GA thuc su ra quyet dinh.

Nen tap du lieu duoc tron tu hai nguon:

  A. LAY MAU RONG (--ga-frac phan con lai)
     40% genome ngau nhien deu + 60% genome "co cau truc" + nhieu.
     Muc dich: phu rong khong gian, cho GNN hoc quy luat vat ly tong quat,
     va tong quat hoa sang kich ban khac (50% dung AC_SET cua Sec. V, 50%
     ngau nhien 3..6 AC -- dieu MLP khong lam duoc).

  B. QUY DAO GA (--ga-frac)
     Chay cac GA ngan va ghi lai MOI cau hinh ma GA danh gia.
     Diem mau chot: moi cau hinh GA danh gia DA CO NHAN CHINH XAC roi (chinh la
     ket qua evaluate_config ma GA can de chay). Ghi lai quy dao GA vi vay
     KHONG TON THEM MOT LAN DANH GIA NAO -- du lieu mien phi hoan toan, va phan
     bo cua no khop dung phan bo ma surrogate se gap khi lam viec.

Chay:
    python -m datagen.dataset --n 120000 --out results/gnn_data.npz
    python -m datagen.dataset --n 500 --workers 4 --ga-frac 0.4 \
        --out results/gnn_smoke.npz
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from common.utility import evaluate_config
from config import (AIFSN_RANGE, RETRY_RANGE, TXOP_RANGE_US, TXOP_STEP_US,
                    ACConfig, EDCAParams)
from datagen.graph import MAX_AC, build_graph
from datagen.scenario import main_scenario

# Kep nhan ve mien huu han truoc khi hoi quy.
LOG_C_MIN, LOG_C_MAX = -6.0, 0.0     # c in [1e-6, 1]
THETA_MAX = 16.0                     # Pr(D>=Dmax) >= 1e-16 la du sau


# ---------------------------------------------------------------------------
# Lay mau kich ban
# ---------------------------------------------------------------------------


def sample_scenario(rng: np.random.Generator) -> List[ACConfig]:
    """50% dung kich ban chinh (Sec. V), 50% ngau nhien 3..6 AC."""
    if rng.random() < 0.5:
        return main_scenario()

    n_ac = int(rng.integers(3, 7))
    out = []
    for i in range(n_ac):
        d_max = float(rng.choice([30, 50, 60, 100, 150, 200, 300]))
        out.append(ACConfig(
            name=f"AC{i+1}",
            n_sta=int(rng.integers(1, 7)),
            payload_bytes=int(rng.choice([50, 210, 256, 512, 800, 1500, 2000])),
            d_max_ms=d_max,
            epsilon=float(10.0 ** rng.uniform(-7.5, -0.3)),
        ))
    return out


# ---------------------------------------------------------------------------
# Lay mau cau hinh EDCA
# ---------------------------------------------------------------------------


def _sample_uniform(rng: np.random.Generator, n_ac: int,
                    n_links: int) -> List[EDCAParams]:
    """Genome ngau nhien deu tren toan mien Sec. IV.B."""
    out = []
    for _ in range(n_ac):
        cw_min = int(2 ** rng.uniform(2.0, 10.0))
        span = int(2 ** rng.integers(0, 11))
        out.append(EDCAParams(
            cw_min=cw_min,
            cw_max=int(min(max(cw_min * span, cw_min), 1023)),
            aifsn=int(rng.integers(AIFSN_RANGE[0], AIFSN_RANGE[1] + 1)),
            txop_us=float(rng.integers(0, int(TXOP_RANGE_US[1] / TXOP_STEP_US) + 1)
                          * TXOP_STEP_US),
            retry=int(rng.integers(RETRY_RANGE[0], RETRY_RANGE[1] + 1)),
            link=int(rng.integers(0, max(n_links, 1))),
        ))
    return out


def _sample_structured(rng: np.random.Generator, ac_set: Sequence[ACConfig],
                       n_links: int) -> List[EDCAParams]:
    """Cau hinh trong vung ma GA thuc su suc suc, kem nhieu.

    Cau truc cua nghiem tot (xem training/ga.structured_seeds va README muc 6.2):
    CW_min lon + CW_max ~ CW_min (m_k = 0) + R = 7, AC co eps chat duoc uu tien
    AIFSN nho, AC nen bi day sang AIFSN lon va TXOP dai.
    """
    order = np.argsort([a.epsilon for a in ac_set])     # chat -> long
    rank = np.argsort(order)
    cw_base = int(2 ** rng.uniform(4.0, 10.0))
    out = []
    for i in range(len(ac_set)):
        rk = int(rank[i])
        cw_min = int(np.clip(cw_base * 2.0 ** rng.normal(0, 0.7), 4, 1023))
        span = int(2 ** rng.integers(0, 3))             # thien ve m_k nho
        aifsn = int(np.clip(2 + rk + rng.integers(-1, 4),
                            AIFSN_RANGE[0], AIFSN_RANGE[1]))
        txop = float(rng.choice([0.0, 0.0, 32.0, 512.0, 1024.0, 2048.0, 4096.0]))
        out.append(EDCAParams(
            cw_min=cw_min,
            cw_max=int(min(max(cw_min * span, cw_min), 1023)),
            aifsn=aifsn,
            txop_us=txop,
            retry=int(rng.integers(RETRY_RANGE[0], RETRY_RANGE[1] + 1)),
            link=int(rng.integers(0, max(n_links, 1))),
        ))
    return out


# ---------------------------------------------------------------------------
# Sinh mot mau (chay trong worker)
# ---------------------------------------------------------------------------


def encode_sample(params, ac_set, res) -> Optional[Tuple[np.ndarray, ...]]:
    """(cau hinh, ket qua QoS) -> (do thi + nhan). None neu ket qua khong hop le."""
    if not np.all(np.isfinite(res.c)):
        return None
    g = build_graph(params, ac_set)
    n_ac = len(ac_set)

    y_logc = np.zeros(MAX_AC, dtype=np.float32)
    y_theta = np.zeros(MAX_AC, dtype=np.float32)
    y_logc[:n_ac] = np.clip(np.log10(np.maximum(res.c[:n_ac], 1e-12)),
                            LOG_C_MIN, LOG_C_MAX)
    y_theta[:n_ac] = np.clip(res.theta[:n_ac], 0.0, THETA_MAX)
    return (g.x, g.edge, g.adj, g.mask, g.retry, g.eps_log, y_logc, y_theta)


def make_sample(seed: int) -> Optional[Tuple[np.ndarray, ...]]:
    """Nguon A -- mot mau lay mau rong. None neu cau hinh loi."""
    rng = np.random.default_rng(seed)
    ac_set = sample_scenario(rng)
    n_links = 2 if rng.random() < 0.75 else 1

    if rng.random() < 0.4:
        params = _sample_uniform(rng, len(ac_set), n_links)
    else:
        params = _sample_structured(rng, ac_set, n_links)

    try:
        res = evaluate_config(params, ac_set)
    except Exception:
        return None
    return encode_sample(params, ac_set, res)


# --- Nguon B: quy dao GA ---------------------------------------------------

# so lan danh gia moi lan chay GA ngan (dung de uoc luong so lan chay can thiet)
GA_RUN_POP, GA_RUN_GEN = 40, 18


def make_ga_samples(seed: int) -> List[Tuple[np.ndarray, ...]]:
    """Nguon B -- chay MOT GA ngan va ghi lai moi cau hinh no danh gia.

    Moi cau hinh GA danh gia deu can `evaluate_config` de chay -- ta chi ghi lai
    ket qua san co, nen khong ton them lan danh gia nao. Phan bo mau vi the khop
    dung phan bo ma surrogate se gap khi ho tro GA.
    """
    import training.ga as ga_mod
    from config import GAParams

    rng = np.random.default_rng(seed)
    ac_set = sample_scenario(rng)
    n_links = 2 if rng.random() < 0.75 else 1

    rows: List[Tuple[np.ndarray, ...]] = []
    original = ga_mod.evaluate_config

    def recording(params, acs, **kw):
        res = original(params, acs, **kw)
        row = encode_sample(params, acs, res)
        if row is not None:
            rows.append(row)
        return res

    ga_mod.evaluate_config = recording          # chi trong tien trinh worker nay
    try:
        ga_mod.run_ga(ac_set, n_links=n_links,
                      ga=GAParams(n_pop=GA_RUN_POP, n_gen=GA_RUN_GEN,
                                  n_elite=4, n_stag=12,
                                  seed=int(rng.integers(1 << 30))),
                      polish=False, verbose=False)
    except Exception:
        pass
    finally:
        ga_mod.evaluate_config = original
    return rows


# ---------------------------------------------------------------------------
# Sinh song song
# ---------------------------------------------------------------------------


def _run_pool(fn, seeds, workers: int, chunk: int, label: str, target: int,
              t0: float, flatten: bool) -> List[Tuple[np.ndarray, ...]]:
    """Chay `fn` tren `seeds`, dung khi thu du `target` mau."""
    rows: List[Tuple[np.ndarray, ...]] = []
    step = max(target // 12, 1)
    next_report = step

    def absorb(r) -> None:
        if r is None:
            return
        rows.extend(r) if flatten else rows.append(r)

    if workers <= 1:
        for s in seeds:
            absorb(fn(s))
            if len(rows) >= next_report:
                print(f"  [{label}] {len(rows)}/{target} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                next_report += step
            if len(rows) >= target:
                break
    else:
        from multiprocessing import Pool
        with Pool(processes=workers) as pool:
            it = pool.imap_unordered(fn, seeds, chunksize=chunk)
            for r in it:
                absorb(r)
                if len(rows) >= next_report:
                    rate = len(rows) / max(time.time() - t0, 1e-9)
                    print(f"  [{label}] {len(rows)}/{target} "
                          f"({time.time()-t0:.0f}s, {rate:.0f} mau/s, "
                          f"con ~{max(target-len(rows),0)/max(rate,1e-9):.0f}s)",
                          flush=True)
                    next_report += step
                if len(rows) >= target:
                    pool.terminate()
                    break
    return rows[:target]


def generate(n: int, out_path: str, workers: int = 0, seed0: int = 10_000_000,
             chunk: int = 64, ga_frac: float = 0.4) -> None:
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    n_ga = int(n * ga_frac)
    rows: List[Tuple[np.ndarray, ...]] = []

    # --- Nguon B: quy dao GA (vung sat bien rang buoc) ---------------------
    if n_ga > 0:
        # moi lan chay GA cho ~GA_RUN_POP*GA_RUN_GEN mau; du seed de chac chan
        n_runs = int(np.ceil(n_ga / (GA_RUN_POP * GA_RUN_GEN * 0.5))) + workers
        rows += _run_pool(make_ga_samples,
                          list(range(seed0 + 5_000_000, seed0 + 5_000_000 + n_runs)),
                          workers, 1, "quy dao GA", n_ga, t0, flatten=True)

    # --- Nguon A: lay mau rong ---------------------------------------------
    n_broad = n - len(rows)
    if n_broad > 0:
        rows += _run_pool(make_sample,
                          list(range(seed0, seed0 + int(n_broad * 1.15) + 64)),
                          workers, chunk, "lay mau rong", n_broad, t0,
                          flatten=False)

    if not rows:
        raise RuntimeError("khong sinh duoc mau nao")

    keys = ["x", "edge", "adj", "mask", "retry", "eps_log", "y_logc", "y_theta"]
    data = {k: np.stack([r[i] for r in rows]) for i, k in enumerate(keys)}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **data)

    dt = time.time() - t0
    m = data["mask"] > 0
    excess = np.maximum(-data["y_theta"] - data["eps_log"], 0.0) * data["mask"]
    feas = (excess.sum(1) <= 0)
    near = (excess.sum(1) <= 2.0)

    print(f"\nDa luu {len(rows)} mau -> {out_path}  ({dt:.0f}s, "
          f"{len(rows)/dt:.0f} mau/s, {workers} worker)")
    print(f"  log10(c): min {data['y_logc'][m].min():.2f}  "
          f"max {data['y_logc'][m].max():.2f}  "
          f"trung binh {data['y_logc'][m].mean():.2f}")
    print(f"  theta   : min {data['y_theta'][m].min():.2f}  "
          f"max {data['y_theta'][m].max():.2f}  "
          f"trung binh {data['y_theta'][m].mean():.2f}")
    print(f"  so AC/mau: trung binh {data['mask'].sum(1).mean():.2f}")
    print(f"  KHA THI  : {feas.sum()} mau ({feas.mean()*100:.2f} %)")
    print(f"  gan kha thi (vuot < 2 bac): {near.sum()} mau ({near.mean()*100:.2f} %)")


def load(path: str):
    d = np.load(path)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sinh tap huan luyen cho GNN surrogate")
    ap.add_argument("--n", type=int, default=120_000, help="so mau")
    ap.add_argument("--out", type=str, default="results/gnn_data.npz")
    ap.add_argument("--workers", type=int, default=0, help="0 = so nhan CPU - 2")
    ap.add_argument("--seed0", type=int, default=10_000_000)
    ap.add_argument("--ga-frac", type=float, default=0.4,
                    help="ty le mau lay tu quy dao GA (vung sat bien rang buoc)")
    a = ap.parse_args()
    generate(a.n, a.out, workers=a.workers, seed0=a.seed0, ga_frac=a.ga_frac)

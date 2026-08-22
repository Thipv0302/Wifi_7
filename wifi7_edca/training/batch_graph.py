"""
training/batch_graph.py -- Ban VECTOR HOA cua datagen.graph.build_graph.

`build_graph` dung vong lap Python nen ~0.3 ms moi cau hinh. Huan luyen policy
(muc 9.8) can danh gia hang nghin cau hinh MOI BUOC, nen vong lap Python tro
thanh nut that. File nay dung dung cac dac trung do nhung tren ca LO va tren GPU.

Dat rieng khoi datagen/graph.py de module do KHONG phu thuoc torch: no duoc
import trong cac worker da tien trinh cua datagen/dataset.py, ma nap torch o day
lam cham khoi dong worker dang ke.

Tinh dung dan duoc rang buoc bang `python -m training.batch_graph`: so khop tung
phan tu voi `datagen.graph.build_graph` tren genome ngau nhien.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from common.timing import T_ACK_US, T_CTS_US, T_RTS_US
from config import (CW_MAX_RANGE, L_MAC_H, R_DATA, SIGMA_US, T_PHY_H_US,
                    T_SIFS_US)
from datagen.genome import (AIFSN_LEVELS, CW_LEVELS, CW_SPAN_LEVELS,
                            RETRY_LEVELS, TXOP_LEVELS)
from datagen.graph import MAX_AC

# so muc cua tung gen -- thu tu trung voi datagen.genome.GENES_PER_AC
GENE_SIZES = (len(CW_LEVELS), len(CW_SPAN_LEVELS), len(AIFSN_LEVELS),
              len(TXOP_LEVELS), len(RETRY_LEVELS))


class LevelTables:
    """Bang tra muc -> gia tri, giu san tren thiet bi dich."""

    def __init__(self, device: str):
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32,
                                      device=device)
        self.cw = f(CW_LEVELS)
        self.span = f(CW_SPAN_LEVELS)
        self.aifsn = f(AIFSN_LEVELS)
        self.txop = f(TXOP_LEVELS)
        self.retry = f(RETRY_LEVELS)


def batch_graph(levels: torch.Tensor, scen: Dict[str, torch.Tensor],
                tables: LevelTables, n_links: int
                ) -> Tuple[torch.Tensor, ...]:
    """(muc gen, kich ban) -> dung cac tensor ma training.gnn.EDCAGNN nhan.

    levels : (B, N, 6) long -- chi so muc cua (cw_min, span, aifsn, txop, R, link)
    scen   : dict cac tensor (B, N): n_sta, payload, d_max_us, eps, mask
    tra ve : x (B,N,17) · edge (B,N,N,5) · adj (B,N,N) · mask (B,N)
             · retry (B,N) · eps_log (B,N)
    """
    mask = scen["mask"]
    n_sta, payload = scen["n_sta"], scen["payload"]

    # --- giai ma muc -> gia tri tham so (khop datagen.genome.decode) --------
    cw_min = tables.cw[levels[..., 0]]
    cw_max = torch.clamp(cw_min * tables.span[levels[..., 1]],
                         max=float(CW_MAX_RANGE[1]))
    cw_max = torch.maximum(cw_max, cw_min)
    aifsn = tables.aifsn[levels[..., 2]]
    txop = tables.txop[levels[..., 3]]
    retry = tables.retry[levels[..., 4]]
    link = levels[..., 5] % max(n_links, 1)

    # --- dai luong closed-form (common/timing.py, Eq. 7-9) -----------------
    aifs_us = T_SIFS_US + aifsn * SIGMA_US
    t_data = T_PHY_H_US + (L_MAC_H + payload * 8.0) / R_DATA * 1e6
    delta_k = t_data + T_ACK_US + 2 * T_SIFS_US
    n_pkt = torch.clamp(torch.floor(txop / delta_k), min=1.0)
    t_c = T_RTS_US + aifs_us
    t_s = T_RTS_US + T_CTS_US + n_pkt * delta_k + T_SIFS_US + aifs_us

    # --- cau truc link: S_ij = 1 neu i, j cung link (KE CA duong cheo) -----
    same = (link.unsqueeze(2) == link.unsqueeze(1)).float()
    same = same * mask.unsqueeze(2) * mask.unsqueeze(1)
    sta_on_link = torch.einsum("bij,bj->bi", same, n_sta)
    ac_on_link = same.sum(dim=2)

    eye = torch.eye(mask.shape[1], device=mask.device).unsqueeze(0)
    adj = same * (1.0 - eye)

    lg = lambda v, lo=1.0: torch.log10(torch.clamp(v, min=lo))
    x = torch.stack([
        n_sta / 5.0,
        lg(payload) / 3.0,
        lg(scen["d_max_us"]) / 5.0,
        torch.log10(torch.clamp(scen["eps"], min=1e-12)) / 8.0,
        torch.log2(torch.clamp(cw_min, min=1.0)) / 10.0,
        torch.log2(cw_max / torch.clamp(cw_min, min=1.0)) / 10.0,
        aifsn / 15.0,
        txop / 8192.0,
        retry / 7.0,
        lg(t_data) / 4.0,
        lg(delta_k) / 4.0,
        lg(n_pkt) / 2.0,
        lg(t_c) / 4.0,
        lg(t_s) / 5.0,
        sta_on_link / 16.0,
        ac_on_link / 5.0,
        (sta_on_link / torch.clamp(ac_on_link, min=1.0)) / 5.0,
    ], dim=-1) * mask.unsqueeze(-1)

    # --- dac trung canh ----------------------------------------------------
    dh = (aifs_us.unsqueeze(1) - aifs_us.unsqueeze(2)) / SIGMA_US   # h_j - h_i
    nj = n_sta.unsqueeze(1).expand(-1, mask.shape[1], -1)
    aifs_i, aifs_j = aifs_us.unsqueeze(2), aifs_us.unsqueeze(1)
    edge = torch.stack([
        nj / 5.0,
        torch.tanh(dh / 5.0),
        (aifs_j <= aifs_i).float(),
        (aifs_j == aifs_i).float(),
        lg(nj) / 1.5,
    ], dim=-1) * adj.unsqueeze(-1)

    eps_log = torch.log10(torch.clamp(scen["eps"], min=1e-12)) * mask
    return x, edge, adj, mask, retry * mask, eps_log


# ---------------------------------------------------------------------------
# Kiem tra tinh dung dan so voi ban tham chieu
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from datagen.dataset import sample_scenario
    from datagen.genome import GenomeSpec, decode
    from datagen.graph import build_graph, stack

    rng = np.random.default_rng(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tables = LevelTables(dev)

    worst = {k: 0.0 for k in ("x", "edge", "adj", "retry", "eps_log")}
    for trial in range(60):
        ac_set = sample_scenario(rng)
        n_ac, n_links = len(ac_set), int(rng.integers(1, 3))
        spec = GenomeSpec(n_ac=n_ac, n_links=n_links, allow_link_choice=n_links > 1)

        g_flat = rng.integers(0, spec.upper())
        params = decode(g_flat, spec)
        ref = stack([build_graph(params, ac_set)])

        lv = np.zeros((1, MAX_AC, 6), dtype=np.int64)
        lv[0, :n_ac] = g_flat.reshape(n_ac, 6)
        scen = {
            "n_sta": np.zeros((1, MAX_AC), dtype=np.float32),
            "payload": np.zeros((1, MAX_AC), dtype=np.float32),
            "d_max_us": np.zeros((1, MAX_AC), dtype=np.float32),
            "eps": np.ones((1, MAX_AC), dtype=np.float32),
            "mask": np.zeros((1, MAX_AC), dtype=np.float32),
        }
        for i, a in enumerate(ac_set):
            scen["n_sta"][0, i] = a.n_sta
            scen["payload"][0, i] = a.payload_bytes
            scen["d_max_us"][0, i] = a.d_max_us
            scen["eps"][0, i] = a.epsilon
            scen["mask"][0, i] = 1.0

        t = lambda a: torch.as_tensor(a, device=dev)
        got = batch_graph(t(lv), {k: t(v) for k, v in scen.items()},
                          tables, n_links)

        for name, i in (("x", 0), ("edge", 1), ("adj", 2), ("retry", 4),
                        ("eps_log", 5)):
            d = float(np.abs(got[i].cpu().numpy() - ref[i]).max())
            worst[name] = max(worst[name], d)

    print(f"thiet bi: {dev} · 60 cau hinh ngau nhien")
    for k, v in worst.items():
        flag = "OK" if v < 1e-5 else "LECH"
        print(f"  {k:8} sai lech lon nhat = {v:.2e}   {flag}")
    assert max(worst.values()) < 1e-5, "ban vector hoa KHONG khop build_graph"
    print("\nBan vector hoa khop chinh xac voi datagen.graph.build_graph.")

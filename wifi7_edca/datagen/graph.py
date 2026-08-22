"""
datagen/graph.py -- MODULE SINH DU LIEU (3/3): bieu dien DO THI cua mot cau hinh.

Bien mot cau hinh EDCA da link (params + ac_set) thanh do thi de dua vao GNN.

Vi sao la do thi -- va vi sao GNN chu khong phai MLP:

  Eq.(4)  c_k = sum_j (pi_j / sum pi) * (1 - (prod_l r_l^{n_l}) / r_k)

  Lay log phan tich:   log prod_l r_l^{n_l} = sum_l n_l * log r_l

  Ve phai la mot phep SUM-AGGREGATION BAT BIEN HOAN VI tren cac node lang gieng,
  voi n_l la trong so canh va log r_l la "message". Lop ngoai (trung binh co
  trong so pi_j theo vung AIFS) la mot phep aggregation thu hai -- giong attention
  theo vung. Nghia la vong lap diem co dinh Eq.(4)-(5) trong common/collision.py
  CHINH LA T buoc message passing tren do thi day du. Do khong phai phep loai suy
  ma la trung khop cau truc: mot GNN chia se trong so qua T lop co the bieu dien
  Eq.(4) chinh xac, con MLP thi phai hoc thuoc long.

Cau truc do thi (mot "mau" = mot cau hinh):

  * Node  : moi AC la mot node (khong dung node link rieng -- phep gan link duoc
            ma hoa THANG vao ma tran ke, xem duoi).
  * Canh  : AC_i -- AC_j neu CUNG LINK. Day la diem mau chot: bien `link` trong
            genome lam thay doi TOPOLOGY cua do thi chu khong phai gia tri dac
            trung. MLO von di la bai toan do thi.
  * Edge feature: n_j (so tram cua lang gieng -- dung la trong so trong Eq. 4),
            do lech AIFS h_j - h_i (slot), va co j nam trong vung tranh chap
            dau tien cua i hay khong.

Do thi chi 3-8 node nen dung MA TRAN KE DAY (dense) thay vi thua: nhanh hon
scatter thua o quy mo nay va khong can cai torch_geometric.

Tat ca dai luong closed-form re tien (common/timing.py Eq. 7-9, common/zones.py
Eq. 2) duoc dua vao lam DAC TRUNG -- cho khong, chinh xac tuyet doi, va giup GNN
khong phai hoc lai nhung thu da co cong thuc dong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from common.timing import ac_timing
from config import SIGMA_US, ACConfig, EDCAParams

# So node toi da (dem cho AC); ho tro kich ban 2..MAX_AC access category.
MAX_AC = 8

# --- so chieu dac trung ----------------------------------------------------
N_NODE_FEAT = 17
N_EDGE_FEAT = 5


@dataclass
class GraphSample:
    """Mot mau do thi da dem (padding den MAX_AC node)."""

    x: np.ndarray          # (MAX_AC, N_NODE_FEAT)  dac trung node
    edge: np.ndarray       # (MAX_AC, MAX_AC, N_EDGE_FEAT) dac trung canh
    adj: np.ndarray        # (MAX_AC, MAX_AC) 1.0 neu i,j cung link va i != j
    mask: np.ndarray       # (MAX_AC,) 1.0 neu node ton tai
    retry: np.ndarray      # (MAX_AC,) R_i -- can de dung lai P_loss = c^R
    eps_log: np.ndarray    # (MAX_AC,) log10(eps_i) -- can de tinh fitness


def build_graph(params: Sequence[EDCAParams],
                ac_set: Sequence[ACConfig]) -> GraphSample:
    """Cau hinh EDCA da link -> do thi (dac trung node + canh + ma tran ke)."""
    n_ac = len(ac_set)
    if n_ac > MAX_AC:
        raise ValueError(f"so AC ({n_ac}) vuot MAX_AC = {MAX_AC}")

    x = np.zeros((MAX_AC, N_NODE_FEAT), dtype=np.float32)
    edge = np.zeros((MAX_AC, MAX_AC, N_EDGE_FEAT), dtype=np.float32)
    adj = np.zeros((MAX_AC, MAX_AC), dtype=np.float32)
    mask = np.zeros(MAX_AC, dtype=np.float32)
    retry = np.zeros(MAX_AC, dtype=np.float32)
    eps_log = np.zeros(MAX_AC, dtype=np.float32)

    link_of = np.array([p.link for p in params], dtype=int)
    n_sta = np.array([a.n_sta for a in ac_set], dtype=float)
    aifs = np.array([p.aifs_us for p in params], dtype=float)

    # so tram / so AC tren cung link -- dac trung "tai" cua link ma node thuoc ve
    sta_on_link = np.array([n_sta[link_of == link_of[i]].sum()
                            for i in range(n_ac)])
    ac_on_link = np.array([float((link_of == link_of[i]).sum())
                           for i in range(n_ac)])

    for i in range(n_ac):
        p, a = params[i], ac_set[i]
        t = ac_timing(p, a.payload_bytes)
        cw_min = max(float(p.cw_min), 1.0)
        cw_max = max(float(p.cw_max), cw_min)

        x[i] = [
            # --- kich ban / rang buoc QoS (khong phai bien quyet dinh) ------
            a.n_sta / 5.0,
            np.log10(max(a.payload_bytes, 1)) / 3.0,
            np.log10(max(a.d_max_us, 1.0)) / 5.0,
            np.log10(max(a.epsilon, 1e-12)) / 8.0,
            # --- bien quyet dinh EDCA --------------------------------------
            np.log2(cw_min) / 10.0,
            np.log2(cw_max / cw_min) / 10.0,          # m_k -- so bac nhan doi
            p.aifsn / 15.0,
            p.txop_us / 8192.0,
            p.retry / 7.0,
            # --- dai luong closed-form (Eq. 7-9) -- cho khong, chinh xac ----
            np.log10(max(t.t_data_us, 1.0)) / 4.0,
            np.log10(max(t.delta_us, 1.0)) / 4.0,
            np.log10(max(float(t.n_pkt), 1.0)) / 2.0,
            np.log10(max(t.t_c_us, 1.0)) / 4.0,
            np.log10(max(t.t_s_us, 1.0)) / 5.0,
            # --- dac trung cau truc link -----------------------------------
            sta_on_link[i] / 16.0,
            ac_on_link[i] / 5.0,
            # ty le "tai tren link" -- goi y truc tiep ve muc do tranh chap
            (sta_on_link[i] / max(ac_on_link[i], 1.0)) / 5.0,
        ]
        mask[i] = 1.0
        retry[i] = p.retry
        eps_log[i] = np.log10(max(a.epsilon, 1e-12))

    # --- canh: chi noi cac AC CUNG LINK ------------------------------------
    for i in range(n_ac):
        for j in range(n_ac):
            if i == j or link_of[i] != link_of[j]:
                continue
            adj[i, j] = 1.0
            dh = (aifs[j] - aifs[i]) / SIGMA_US        # h_j - h_i, Eq.(2)
            edge[i, j] = [
                n_sta[j] / 5.0,              # trong so n_j trong Eq.(4)
                np.tanh(dh / 5.0),           # do lech AIFS co dau
                1.0 if aifs[j] <= aifs[i] else 0.0,   # j tranh chap tu vung cua i
                1.0 if aifs[j] == aifs[i] else 0.0,   # cung vung AIFS
                np.log10(max(n_sta[j], 1.0)) / 1.5,
            ]

    return GraphSample(x=x, edge=edge, adj=adj, mask=mask,
                       retry=retry, eps_log=eps_log)


def stack(samples: Sequence[GraphSample]) -> Tuple[np.ndarray, ...]:
    """Gom nhieu GraphSample thanh cac mang batch (B, ...)."""
    return (np.stack([s.x for s in samples]),
            np.stack([s.edge for s in samples]),
            np.stack([s.adj for s in samples]),
            np.stack([s.mask for s in samples]),
            np.stack([s.retry for s in samples]),
            np.stack([s.eps_log for s in samples]))


if __name__ == "__main__":
    from config import AC_SET, DEFAULT_EDCA

    mlo = [p.copy() for p in DEFAULT_EDCA]
    for i, lk in enumerate([0, 1, 1, 0, 0]):
        mlo[i].link = lk

    g = build_graph(mlo, AC_SET)
    print("x     :", g.x.shape, "  (MAX_AC, N_NODE_FEAT)")
    print("edge  :", g.edge.shape)
    print("adj   :\n", g.adj[:5, :5].astype(int))
    print("mask  :", g.mask)
    print("kiem tra: AC1,AC4,AC5 tren link 0 -> ke nhau; AC2,AC3 tren link 1")
    print("bien do dac trung node: min = %.2f  max = %.2f"
          % (g.x[g.mask > 0].min(), g.x[g.mask > 0].max()))

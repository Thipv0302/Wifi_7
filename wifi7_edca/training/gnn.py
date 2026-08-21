"""
training/gnn.py -- GNN surrogate thay cho common/utility.evaluate_config.

KIEN TRUC -- va ly do chon tung thanh phan:

  h_i^0 = Enc(x_i)
  lap T lan (CHIA SE TRONG SO):
      m_ij  = Msg([h_i, h_j, e_ij])
      a_i   = sum_j adj_ij * m_ij      <- sum-aggregation, khop Eq.(4)
              || max_j adj_ij * m_ij   <- them max de bat "AC nao chi phoi kenh"
      h_i   = h_i + Upd([h_i, a_i])    <- residual: mot buoc lap diem co dinh
  (log10 c_i, theta_i) = Out(h_i)

1. SUM-AGGREGATION la bat buoc, khong phai mean. Eq.(4) chua
   log prod_l r_l^{n_l} = sum_l n_l log r_l -- mot TONG tren lang gieng. Dung
   mean-aggregation se chuan hoa mat thong tin "co bao nhieu tram dang tranh
   chap", dung la dai luong quyet dinh xac suat va cham.

2. CHIA SE TRONG SO qua T lop: vong lap trong common/collision.solve_link la mot
   phep lap diem co dinh voi CUNG mot toan tu moi vong. Chia se trong so phan
   anh dung dieu do (GNN hoi quy / kieu deep-equilibrium), va lam mang it tham so
   hon nen kho overfit hon. T = 6 la du: do thi day du nen thong tin lan het chi
   sau 1 hop; T lop mo phong T VONG LAP, khong phai T buoc lan truyen khoang cach.

3. MA TRAN KE DAY thay vi thua: do thi chi 3-8 node. Dense einsum nhanh hon
   scatter thua o quy mo nay VA khong can cai torch_geometric (von rat hay gay
   tren Windows voi torch 2.7+cu118).

4. Ham muc tieu duoc dung lai bang CONG THUC CHINH XAC tu dau ra cua mang:
   mang du doan log10(c) va theta, con P_loss = c^R (Eq. 6) va Eq.(18) thi TINH
   THANG. Mang khong bao gio phai hoc phep luy thua ^R.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from common.utility import P_LOSS_FLOOR, VIOLATION_PENALTY
from datagen.graph import (MAX_AC, N_EDGE_FEAT, N_NODE_FEAT, GraphSample,
                           build_graph, stack)

# tran diem cua moi AC khi -log10(P_loss) bi kep boi P_LOSS_FLOOR
OBJ_CAP = -np.log10(P_LOSS_FLOOR)      # = 10.0


def _mlp(sizes: Sequence[int], act_last: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or act_last:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class EDCAGNN(nn.Module):
    """GNN du doan (log10 c_i, theta_i) cho tung AC cua mot cau hinh EDCA."""

    def __init__(self, hidden: int = 128, n_layers: int = 6,
                 edge_hidden: int = 32, shared: bool = True):
        super().__init__()
        self.n_layers = n_layers
        self.shared = shared

        self.enc = _mlp([N_NODE_FEAT, hidden, hidden])
        self.edge_enc = _mlp([N_EDGE_FEAT, edge_hidden, edge_hidden], act_last=True)

        n_blocks = 1 if shared else n_layers
        self.msg = nn.ModuleList([
            _mlp([2 * hidden + edge_hidden, hidden, hidden]) for _ in range(n_blocks)])
        self.upd = nn.ModuleList([
            _mlp([3 * hidden, hidden, hidden]) for _ in range(n_blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_blocks)])

        self.out = _mlp([hidden, hidden, 2])

    def forward(self, x: torch.Tensor, edge: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """x (B,N,F) · edge (B,N,N,E) · adj (B,N,N) · mask (B,N) -> (B,N,2)."""
        h = self.enc(x) * mask.unsqueeze(-1)
        e = self.edge_enc(edge)
        a = adj.unsqueeze(-1)                                   # (B,N,N,1)

        for t in range(self.n_layers):
            b = 0 if self.shared else t
            hi = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)  # (B,N,N,H) node dich
            hj = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)  # (B,N,N,H) node nguon
            m = self.msg[b](torch.cat([hi, hj, e], dim=-1)) * a

            agg_sum = m.sum(dim=2)                              # Eq.(4): TONG
            agg_max = m.masked_fill(a == 0, -1e9).max(dim=2).values
            agg_max = torch.where(adj.sum(-1, keepdim=True) > 0,
                                  agg_max, torch.zeros_like(agg_max))

            h = h + self.upd[b](torch.cat([h, agg_sum, agg_max], dim=-1))
            h = self.norm[b](h) * mask.unsqueeze(-1)

        return self.out(h) * mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Tu dau ra cua mang -> muc tieu Eq.(18) va fitness (CONG THUC CHINH XAC)
# ---------------------------------------------------------------------------


def fitness_from_pred(log_c: torch.Tensor, theta: torch.Tensor,
                      retry: torch.Tensor, eps_log: torch.Tensor,
                      mask: torch.Tensor,
                      penalty: float = VIOLATION_PENALTY) -> torch.Tensor:
    """Tai lap common.utility.fitness tu (log10 c, theta) du doan.

        P_loss_i = c_i^{R_i}                     (Eq. 6, tinh chinh xac)
        muc tieu = sum_i min(-R_i log10 c_i, 10) (Eq. 18 voi san P_LOSS_FLOOR)
        excess_i = max(-theta_i - log10 eps_i, 0)
        fit = muc tieu neu moi excess = 0, nguoc lai -lambda * sum excess
    """
    obj_i = torch.clamp(-retry * log_c, max=OBJ_CAP) * mask
    excess_i = torch.clamp(-theta - eps_log, min=0.0) * mask

    obj = obj_i.sum(dim=-1)
    excess = excess_i.sum(dim=-1)
    return torch.where(excess <= 0.0, obj, -penalty * excess)


# ---------------------------------------------------------------------------
# Boc goi de goi tu GA
# ---------------------------------------------------------------------------


class Surrogate:
    """Boc GNN da huan luyen: danh gia HANG LOAT cau hinh trong mot forward pass.

    Day la ly do surrogate nhanh: GA danh gia ca quan the 200 ca the bang MOT
    lan goi GPU, thay vi 200 lan giai diem co dinh + nghich dao Fourier tuan tu.
    """

    def __init__(self, model: EDCAGNN, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @staticmethod
    def load(path: str, device: Optional[str] = None) -> "Surrogate":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = EDCAGNN(**ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        return Surrogate(model, device)

    @torch.no_grad()
    def predict(self, graphs: Sequence[GraphSample]):
        """-> (log10 c, theta) dang numpy, shape (B, MAX_AC)."""
        x, edge, adj, mask, retry, eps_log = stack(graphs)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        out = self.model(t(x), t(edge), t(adj), t(mask))
        return out[..., 0].cpu().numpy(), out[..., 1].cpu().numpy()

    @torch.no_grad()
    def fitness(self, graphs: Sequence[GraphSample]) -> np.ndarray:
        """-> fitness (B,) theo dung dinh nghia cua common.utility.fitness."""
        x, edge, adj, mask, retry, eps_log = stack(graphs)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        m, r, el = t(mask), t(retry), t(eps_log)
        out = self.model(t(x), t(edge), t(adj), m)
        f = fitness_from_pred(out[..., 0], out[..., 1], r, el, m)
        return f.cpu().numpy()

    def fitness_of(self, params_list, ac_set) -> np.ndarray:
        """Tien ich: danh sach cau hinh EDCA -> fitness."""
        return self.fitness([build_graph(p, ac_set) for p in params_list])


if __name__ == "__main__":
    from config import AC_SET, DEFAULT_EDCA

    model = EDCAGNN()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"so tham so: {n_par:,}")

    mlo = [p.copy() for p in DEFAULT_EDCA]
    for i, lk in enumerate([0, 1, 1, 0, 0]):
        mlo[i].link = lk
    g = build_graph(mlo, AC_SET)

    sur = Surrogate(model, "cpu")
    lc, th = sur.predict([g, g])
    print("log10 c :", np.round(lc[0][:5], 3))
    print("theta   :", np.round(th[0][:5], 3))
    print("fitness :", sur.fitness([g])[0], "(mang chua huan luyen)")

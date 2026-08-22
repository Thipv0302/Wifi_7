"""
training/policy.py -- GNN POLICY: sinh thang tham so EDCA tu kich ban (PA2).

Khac biet co ban voi surrogate (training/gnn.py):

  surrogate:  (kich ban + THAM SO EDCA) -> (c, theta)      -- thay ham danh gia
  policy   :   kich ban                 -> THAM SO EDCA    -- thay ca GA

Policy nhan dau vao chi gom kich ban (n_i, L_i, D_max,i, eps_i) va xuat ra phan
phoi tren cac muc roi rac cua tung gen, KE CA phep gan link. Suy luan la mot
forward pass -- vai mili-giay -- thay cho mot lan chay GA hang chuc giay.

Do thi dau vao la DO THI DAY DU tren cac AC: khac voi surrogate, o day phep gan
link chua duoc quyet dinh (no la dau ra), nen khong the dung no lam ma tran ke.
Moi cap AC deu co the tranh chap nen deu duoc noi.

Dac trung kich ban co chu y dua vao ca HANG TUONG DOI cua eps va D_max giua cac
AC: quyet dinh cua GA phu thuoc chu yeu vao viec AC nao chat hon AC nao (xem
`training/ga.structured_seeds`), khong phai gia tri tuyet doi.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import ACConfig
from datagen.graph import MAX_AC
from training.batch_graph import GENE_SIZES

N_SCEN_FEAT = 10
N_SCEN_EDGE = 4
MAX_LINKS = 4
N_GENES = 6                     # (cw_min, span, aifsn, txop, R, link)


def _mlp(sizes: Sequence[int], act_last: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or act_last:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Ma hoa kich ban -> tensor
# ---------------------------------------------------------------------------


def encode_scenarios(ac_sets: Sequence[Sequence[ACConfig]], n_links: int,
                     device: str) -> Tuple[Dict[str, torch.Tensor], ...]:
    """Danh sach kich ban -> (scen cho batch_graph, dac trung dau vao policy)."""
    b = len(ac_sets)
    z = lambda: np.zeros((b, MAX_AC), dtype=np.float32)
    n_sta, payload, d_max, eps, mask = z(), z(), z(), np.ones((b, MAX_AC),
                                                              np.float32), z()
    for k, acs in enumerate(ac_sets):
        for i, a in enumerate(acs):
            n_sta[k, i], payload[k, i] = a.n_sta, a.payload_bytes
            d_max[k, i], eps[k, i], mask[k, i] = a.d_max_us, a.epsilon, 1.0

    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    scen = {"n_sta": t(n_sta), "payload": t(payload), "d_max_us": t(d_max),
            "eps": t(eps), "mask": t(mask)}

    m = scen["mask"]
    n_ac = m.sum(1, keepdim=True)                            # (B,1)
    tot_sta = (scen["n_sta"] * m).sum(1, keepdim=True)
    le = torch.log10(torch.clamp(scen["eps"], min=1e-12))
    ld = torch.log10(torch.clamp(scen["d_max_us"], min=1.0))

    # hang tuong doi: bao nhieu AC khac chat hon AC nay
    strict = ((le.unsqueeze(1) < le.unsqueeze(2)).float()
              * m.unsqueeze(1)).sum(1) / torch.clamp(n_ac, min=1)
    tight = ((ld.unsqueeze(1) < ld.unsqueeze(2)).float()
             * m.unsqueeze(1)).sum(1) / torch.clamp(n_ac, min=1)

    x = torch.stack([
        scen["n_sta"] / 5.0,
        torch.log10(torch.clamp(scen["payload"], min=1.0)) / 3.0,
        ld / 5.0,
        le / 8.0,
        (tot_sta / 16.0).expand(-1, MAX_AC),
        (n_ac / 5.0).expand(-1, MAX_AC),
        torch.full_like(m, n_links / 2.0),
        strict,
        tight,
        scen["n_sta"] / torch.clamp(tot_sta, min=1.0),
    ], dim=-1) * m.unsqueeze(-1)

    # do thi DAY DU (phep gan link chua biet)
    eye = torch.eye(MAX_AC, device=device).unsqueeze(0)
    adj = m.unsqueeze(2) * m.unsqueeze(1) * (1.0 - eye)

    nj = scen["n_sta"].unsqueeze(1).expand(-1, MAX_AC, -1)
    edge = torch.stack([
        nj / 5.0,
        (ld.unsqueeze(1) - ld.unsqueeze(2)) / 2.0,
        (le.unsqueeze(1) - le.unsqueeze(2)) / 8.0,
        (le.unsqueeze(1) < le.unsqueeze(2)).float(),
    ], dim=-1) * adj.unsqueeze(-1)

    return scen, {"x": x, "edge": edge, "adj": adj, "mask": m}


# ---------------------------------------------------------------------------
# Mang policy
# ---------------------------------------------------------------------------


class PolicyGNN(nn.Module):
    """Kich ban -> logit tren cac muc roi rac cua tung gen, cho tung AC."""

    def __init__(self, hidden: int = 128, n_layers: int = 5,
                 edge_hidden: int = 32, n_links: int = 2):
        super().__init__()
        self.n_layers, self.n_links = n_layers, n_links
        self.enc = _mlp([N_SCEN_FEAT, hidden, hidden])
        self.edge_enc = _mlp([N_SCEN_EDGE, edge_hidden, edge_hidden], act_last=True)
        self.msg = _mlp([2 * hidden + edge_hidden, hidden, hidden])
        self.upd = _mlp([3 * hidden, hidden, hidden])
        self.norm = nn.LayerNorm(hidden)
        sizes = list(GENE_SIZES) + [MAX_LINKS]
        self.heads = nn.ModuleList([_mlp([hidden, hidden, s]) for s in sizes])

    def forward(self, inp: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        x, e_raw, adj, mask = inp["x"], inp["edge"], inp["adj"], inp["mask"]
        h = self.enc(x) * mask.unsqueeze(-1)
        e = self.edge_enc(e_raw)
        a = adj.unsqueeze(-1)

        for _ in range(self.n_layers):
            hi = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
            hj = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
            m = self.msg(torch.cat([hi, hj, e], dim=-1)) * a
            agg_sum = m.sum(dim=2)
            agg_max = m.masked_fill(a == 0, -1e9).max(dim=2).values
            agg_max = torch.where(adj.sum(-1, keepdim=True) > 0, agg_max,
                                  torch.zeros_like(agg_max))
            h = h + self.upd(torch.cat([h, agg_sum, agg_max], dim=-1))
            h = self.norm(h) * mask.unsqueeze(-1)

        out = [head(h) for head in self.heads]
        if self.n_links < MAX_LINKS:                 # chan cac link khong ton tai
            out[-1] = out[-1].clone()
            out[-1][..., self.n_links:] = -1e9
        return out


# ---------------------------------------------------------------------------
# Lay mau / xac suat
# ---------------------------------------------------------------------------


def sample_levels(logits: List[torch.Tensor], k: int, temperature: float = 1.0,
                  greedy: bool = False) -> torch.Tensor:
    """-> (B, k, N, 6) chi so muc lay mau doc lap tung gen."""
    outs = []
    for g in range(N_GENES):
        lg = logits[g] / max(temperature, 1e-6)                  # (B,N,S)
        if greedy:
            idx = lg.argmax(-1, keepdim=True).expand(-1, -1, k)  # (B,N,k)
        else:
            p = torch.softmax(lg, dim=-1)
            b, n, s = p.shape
            idx = torch.multinomial(p.reshape(-1, s), k, replacement=True)
            idx = idx.reshape(b, n, k)
        outs.append(idx.permute(0, 2, 1))                        # (B,k,N)
    return torch.stack(outs, dim=-1)


def log_prob(logits: List[torch.Tensor], levels: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """log p(levels | kich ban), cong tren moi gen va moi AC. levels (B,k,N,6)."""
    total = 0.0
    for g in range(N_GENES):
        lp = torch.log_softmax(logits[g], dim=-1).unsqueeze(1)   # (B,1,N,S)
        lp = lp.expand(-1, levels.shape[1], -1, -1)
        total = total + lp.gather(-1, levels[..., g:g + 1]).squeeze(-1)
    return (total * mask.unsqueeze(1)).sum(-1)                   # (B,k)


def entropy(logits: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    tot = 0.0
    for g in range(N_GENES):
        p = torch.softmax(logits[g], dim=-1)
        lp = torch.log_softmax(logits[g], dim=-1)
        tot = tot - (p * lp).sum(-1)
    return (tot * mask).sum(-1) / torch.clamp(mask.sum(-1), min=1)


def levels_to_genome(levels: np.ndarray, n_ac: int) -> np.ndarray:
    """(N,6) -> nhiem sac the phang ma datagen.genome.decode nhan."""
    return np.asarray(levels[:n_ac], dtype=int).ravel()


if __name__ == "__main__":
    from datagen.scenario import main_scenario

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    acs = main_scenario()
    scen, inp = encode_scenarios([acs, acs], n_links=2, device=dev)

    pol = PolicyGNN(n_links=2).to(dev)
    print(f"so tham so: {sum(p.numel() for p in pol.parameters()):,}")
    logits = pol(inp)
    print("so head:", len(logits), "· kich thuoc:", [tuple(l.shape) for l in logits])

    lv = sample_levels(logits, k=4)
    print("levels  :", tuple(lv.shape), "(B, k, N, 6)")
    print("log_prob:", tuple(log_prob(logits, lv, inp["mask"]).shape))
    print("entropy :", entropy(logits, inp["mask"]).mean().item())

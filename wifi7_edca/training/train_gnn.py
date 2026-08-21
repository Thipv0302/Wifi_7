"""
training/train_gnn.py -- Huan luyen GNN surrogate.

Chay:
    # kiem tra nhanh (CPU, vai chuc giay)
    python -m training.train_gnn --data results/gnn_smoke.npz --epochs 2 \
        --hidden 32 --layers 3 --device cpu --out results/gnn_smoke.pt

    # huan luyen that (GPU)
    python -m training.train_gnn --data results/gnn_data.npz --epochs 60 \
        --out results/gnn_model.pt

CHI SO DANH GIA -- doc theo thu tu quan trong:

  1. `feas.acc`  -- ty le cau hinh ma surrogate quyet dinh KHA THI / VI PHAM
     giong het mo hinh giai tich. Day la chi so QUAN TRONG NHAT: GA dung
     surrogate de sang loc, nen phan loai sai o bien rang buoc moi la thu lam
     hong ket qua, khong phai RMSE.
  2. `rho`       -- tuong quan hang (Spearman) cua fitness. GA chi can THU TU
     dung, khong can gia tri tuyet doi dung.
  3. `RMSE theta`-- sai so o thang log. RMSE 0.3 ~ sai so 2x ve xac suat, dung
     bang muc sai so cua CHINH mo hinh giai tich so voi mo phong (README muc 5).
  4. `RMSE logc` -- sai so nay bi khuech dai R lan trong muc tieu Eq.(18), nen
     0.01 tren log10(c) thanh 0.07 diem fitness moi AC.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from training.gnn import EDCAGNN, fitness_from_pred


def load_split(path: str, val_frac: float = 0.1, seed: int = 0):
    d = np.load(path)
    n = d["x"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(int(n * val_frac), 1)
    va, tr = perm[:n_val], perm[n_val:]
    keys = ["x", "edge", "adj", "mask", "retry", "eps_log", "y_logc", "y_theta"]
    return ({k: d[k][tr] for k in keys}, {k: d[k][va] for k in keys})


def to_torch(d: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in d.items()}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, va: Dict[str, torch.Tensor],
             batch: int = 2048) -> Dict[str, float]:
    model.eval()
    n = va["x"].shape[0]
    preds = []
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        preds.append(model(va["x"][sl], va["edge"][sl], va["adj"][sl],
                           va["mask"][sl]))
    out = torch.cat(preds, dim=0)
    m = va["mask"]
    nm = m.sum().clamp(min=1)

    err_c = (out[..., 0] - va["y_logc"]) * m
    err_t = (out[..., 1] - va["y_theta"]) * m
    rmse_c = float(torch.sqrt((err_c ** 2).sum() / nm))
    rmse_t = float(torch.sqrt((err_t ** 2).sum() / nm))
    mae_t = float(err_t.abs().sum() / nm)

    f_pred = fitness_from_pred(out[..., 0], out[..., 1], va["retry"],
                               va["eps_log"], m)
    f_true = fitness_from_pred(va["y_logc"], va["y_theta"], va["retry"],
                               va["eps_log"], m)

    # "excess" = so bac do lon vuot nguong eps -- dai luong GA thuc su leo doc
    ex_pred = torch.clamp(-out[..., 1] - va["eps_log"], min=0) * m
    ex_true = torch.clamp(-va["y_theta"] - va["eps_log"], min=0) * m
    feas_pred = ex_pred.sum(-1) <= 0
    feas_true = ex_true.sum(-1) <= 0

    # tap con SAT BIEN rang buoc -- noi quyet dinh cua GA thuc su duoc dua ra
    near = ex_true.sum(-1) <= 2.0
    n_near = int(near.sum())

    fp, ft = f_pred.cpu().numpy(), f_true.cpu().numpy()
    out_m = {
        "rmse_logc": rmse_c,
        "rmse_theta": rmse_t,
        "mae_theta": mae_t,
        "rmse_excess": float(torch.sqrt(((ex_pred - ex_true) ** 2).sum() / nm)),
        "feas_acc": float((feas_pred == feas_true).float().mean()),
        "feas_rate": float(feas_true.float().mean()),
        "n_feas": int(feas_true.sum()),
        "rho_fit": _spearman(fp, ft),
        "rmse_fit": float(np.sqrt(np.mean((fp - ft) ** 2))),
        "n_near": n_near,
    }
    if n_near > 0:
        nm_near = (m * near.unsqueeze(-1)).sum().clamp(min=1)
        out_m["rmse_theta_near"] = float(torch.sqrt(
            ((err_t * near.unsqueeze(-1)) ** 2).sum() / nm_near))
        out_m["feas_acc_near"] = float(
            (feas_pred[near] == feas_true[near]).float().mean())
        out_m["rho_fit_near"] = _spearman(fp[near.cpu().numpy()],
                                          ft[near.cpu().numpy()])
    return out_m


def train(args) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"thiet bi: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tr_np, va_np = load_split(args.data, args.val_frac, args.seed)
    print(f"tap huan luyen: {tr_np['x'].shape[0]:,} mau · "
          f"kiem tra: {va_np['x'].shape[0]:,} mau")

    tr = to_torch(tr_np, device)
    va = to_torch(va_np, device)

    # can bang hai dau ra: chia du theo do lech chuan cua tung nhan
    m_tr = tr["mask"] > 0
    std_c = float(tr["y_logc"][m_tr].std().clamp(min=1e-3))
    std_t = float(tr["y_theta"][m_tr].std().clamp(min=1e-3))
    print(f"do lech chuan nhan: log10(c) = {std_c:.3f} · theta = {std_t:.3f}")

    arch = dict(hidden=args.hidden, n_layers=args.layers, shared=not args.no_share)
    model = EDCAGNN(**arch).to(device)
    print(f"so tham so: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = tr["x"].shape[0]
    steps = max(n // args.batch, 1) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps, pct_start=0.15)
    huber = nn.HuberLoss(reduction="none", delta=1.0)
    use_amp = device == "cuda" and not args.no_amp

    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(tr["x"][idx], tr["edge"][idx], tr["adj"][idx],
                            tr["mask"][idx])
                m = tr["mask"][idx]
                nm = m.sum().clamp(min=1)
                l_c = (huber(out[..., 0] / std_c,
                             tr["y_logc"][idx] / std_c) * m).sum() / nm
                l_t = (huber(out[..., 1] / std_t,
                             tr["y_theta"][idx] / std_t) * m).sum() / nm
                loss = l_c + args.w_theta * l_t

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
            tot += float(loss); nb += 1

        if ep % max(args.epochs // 20, 1) == 0 or ep == args.epochs - 1:
            mt = evaluate(model, va)
            print(f"ep {ep:3d}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
                  f"RMSE logc {mt['rmse_logc']:.4f}  theta {mt['rmse_theta']:.3f}  "
                  f"feas.acc {mt['feas_acc']*100:.2f}%  rho {mt['rho_fit']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    mt = evaluate(model, va)
    print("\n=== Ket qua tren tap kiem tra ===")
    print(f"  RMSE log10(c)       : {mt['rmse_logc']:.4f}   "
          f"(-> {mt['rmse_logc']*7:.3f} diem fitness moi AC khi R = 7)")
    print(f"  RMSE theta          : {mt['rmse_theta']:.4f}   "
          f"(~ {10**mt['rmse_theta']:.2f}x ve xac suat)")
    print(f"  MAE  theta          : {mt['mae_theta']:.4f}")
    print(f"  RMSE excess         : {mt['rmse_excess']:.4f}")
    print(f"  Spearman rho fitness: {mt['rho_fit']:.4f}")
    print(f"  RMSE fitness        : {mt['rmse_fit']:.3f}")
    print(f"  Do chinh xac kha thi: {mt['feas_acc']*100:.2f}%  "
          f"({mt['n_feas']} mau kha thi that = {mt['feas_rate']*100:.2f}%)")
    if mt.get("n_near"):
        print(f"  --- rieng tap SAT BIEN rang buoc ({mt['n_near']} mau) ---")
        print(f"  RMSE theta          : {mt['rmse_theta_near']:.4f}")
        print(f"  Do chinh xac kha thi: {mt['feas_acc_near']*100:.2f}%")
        print(f"  Spearman rho fitness: {mt['rho_fit_near']:.4f}")
    print(f"  thoi gian huan luyen: {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": arch, "metrics": mt},
               args.out)
    print(f"\nDa luu mo hinh -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Huan luyen GNN surrogate cho EDCA")
    ap.add_argument("--data", type=str, default="results/gnn_data.npz")
    ap.add_argument("--out", type=str, default="results/gnn_model.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--w-theta", type=float, default=1.5,
                    help="trong so cua nhanh theta (rang buoc quan trong hon)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-share", action="store_true",
                    help="khong chia se trong so giua cac lop message passing")
    ap.add_argument("--no-amp", action="store_true")
    train(ap.parse_args())

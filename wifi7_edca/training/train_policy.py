"""
training/train_policy.py -- Huan luyen GNN policy (PA2) va do lai voi GA.

PHUONG PHAP: cross-entropy method / tu mo phong (self-imitation).

    moi buoc:
      1. policy xuat phan phoi tren cac muc gen cho tung AC
      2. lay K cau hinh tu phan phoi do
      3. SURROGATE cham diem ca B*K cau hinh trong MOT forward pass
      4. lay nhom tinh hoa (top 12.5 %) lam nhan, cap nhat bang cross-entropy
      5. cong them thuong entropy (giam dan) de tranh sup do som

Vi sao khong dung REINFORCE hay Gumbel-softmax:
  * REINFORCE co phuong sai lon voi khong gian rieng 17x11x14x257x4x2 moi AC;
  * Gumbel-softmax doi hoi tron MEM cac muc roi rac roi day qua surrogate, nhung
    dac trung Eq.(8) chua phep `floor(TXOP/Delta)` -- khong kha vi, va tron mem
    cac muc CW lam dac trung roi ra ngoai mien ma surrogate tung thay.
  * Cross-entropy method chi can THU TU cua fitness, dung dieu ma surrogate lam
    tot nhat (Spearman rho = 0.9996), va no chinh la phien ban khau hao cua GA
    ma no thay the -- lien he tu nhien de viet trong paper.

Diem mau chot ve chi phi: buoc 3 danh gia hang nghin cau hinh moi buoc. Voi mo
hinh giai tich dieu do la bat kha thi (16 ms x 2048 = 33 giay MOI BUOC); voi
surrogate no la ~15 ms. Policy chi ton tai duoc nho surrogate cua muc 9.

Chay:
    python -m training.train_policy --surrogate results/gnn_model.pt \
        --steps 3000 --out results/policy.pt
    python -m training.train_policy --eval-only --policy results/policy.pt \
        --surrogate results/gnn_model.pt
"""
from __future__ import annotations

import argparse
import time
from typing import List, Optional, Sequence

import numpy as np
import torch

from common.utility import evaluate_config, fitness
from datagen.dataset import sample_scenario
from datagen.genome import GenomeSpec, decode, describe
from datagen.scenario import main_scenario
from training.batch_graph import LevelTables, batch_graph
from training.gnn import OBJ_CAP, Surrogate, fitness_from_pred
from training.policy import (PolicyGNN, encode_scenarios, entropy, log_prob,
                             sample_levels)


# ---------------------------------------------------------------------------
# Cham diem mot lo cau hinh bang surrogate
# ---------------------------------------------------------------------------


def shaped_score(log_c: torch.Tensor, theta: torch.Tensor, retry: torch.Tensor,
                 eps_log: torch.Tensor, mask: torch.Tensor,
                 lam: float = 2.0, margin: float = 0.0) -> torch.Tensor:
    """Diem CO DINH HUONG dung KHI HUAN LUYEN: muc tieu tru phat tuyen tinh.

        shaped = sum_i min(-R_i log10 c_i, 10) - lam * sum_i excess_i
        excess_i = max(-theta_i - log10 eps_i + margin, 0)

    HAI THAM SO NAY DEU THIET YEU, hoc duoc tu mot lan chay hong:

    `lam` co dinh = 2 lam diem toi uu cua chinh ham nay roi vao vung VI PHAM NHE.
    Lan chay dau cho policy dat muc tieu 49.731 (cao hon ca GA goc 49.704) nhung
    AC3 vuot nguong 0.048 bac: phat chi 2*0.048 = 0.096 diem, trong khi day CW
    toi do loi hon the. Vi vay `lam` duoc TANG DAN theo lich (2 -> 20): giai doan
    dau can phat nhe de thoat goc suy bien, giai doan sau can phat nang de rang
    buoc tro thanh that su rang buoc.

    `margin` xu ly mot van de sau hon. Sai so cua surrogate tren theta la 0.39
    (he so 2.47, muc 9.5) trong khi bien do can phan giai o AC3 chi la 0.048 bac.
    Policy dang duoc yeu cau ha canh dung phia cua mot duong bien ma bo danh gia
    cua no KHONG NHIN THAY NOI. Ep policy nham vao TRONG bien mot khoang bang
    dung do bat dinh cua surrogate la cach duy nhat nhat quan: nghiem thu duoc se
    nam sau trong vung kha thi du surrogate co lech mot bac chuan.

    AC3 dung la rang buoc ma README muc 6.4 da ghi nhan la "nam sat bien" -- noi
    mo hinh giai tich cho 6.9e-5 con mo phong cho 1.1e-4. Do khong phai trung
    hop: do la rang buoc chat nhat cua bai toan.

    Vi sao khong dung thang common.utility.fitness: ham do chuyen BAC giua vung
    kha thi (>= 0) va vi pham (<= 0). Dieu do dung cho BAO CAO nhung tao mot bay
    khi huan luyen -- xem muc 9.8 cua README:

      mo hinh cua paper co mot goc SUY BIEN, khi CW rat nho thi c -> 1, hau nhu
      khong goi nao thanh cong, nen tre CO DIEU KIEN cua so it goi song sot lai
      rat ngan va rang buoc Pr(D >= D_max) < eps duoc thoa man mot cach hinh
      thuc. Cau hinh do "kha thi" voi fitness = 0 (P_loss ~ 1).

    Voi ham fitness goc, cau hinh vo dung do (0) xep TREN moi cau hinh dang tien
    gan kha thi (am), nen cross-entropy method sup do vao no ngay lap tuc. Voi
    diem co dinh huong, mot cau hinh co muc tieu 45 va vuot nguong 1 bac duoc
    43 diem -- van hon han 0 -- nen ton tai duong doc di ra.

    GA khong gap van de nay vi `training/ga.structured_seeds` gieo san quan the
    vao vung tot; policy khong co gi tuong duong nen can dinh huong tu ham diem.
    """
    obj = (torch.clamp(-retry * log_c, max=OBJ_CAP) * mask).sum(-1)
    excess = (torch.clamp(-theta - eps_log + margin, min=0.0) * mask).sum(-1)
    return obj - lam * excess


@torch.no_grad()
def score_levels(levels: torch.Tensor, scen, tables: LevelTables, n_links: int,
                 sur_model, shaped: bool = False, lam: float = 2.0,
                 margin: float = 0.0) -> torch.Tensor:
    """levels (B,k,N,6) -> diem (B,k) theo surrogate.

    shaped = False -> dung common.utility.fitness (dung de BAO CAO / chon nghiem)
    shaped = True  -> diem co dinh huong (dung de HUAN LUYEN)
    """
    b, k = levels.shape[:2]
    flat = levels.reshape(b * k, *levels.shape[2:])
    rep = {kk: v.repeat_interleave(k, dim=0) for kk, v in scen.items()}
    x, edge, adj, mask, retry, eps_log = batch_graph(flat, rep, tables, n_links)
    out = sur_model(x, edge, adj, mask)
    fn = shaped_score if shaped else fitness_from_pred
    kw = {"lam": lam, "margin": margin} if shaped else {}
    f = fn(out[..., 0], out[..., 1], retry, eps_log, mask, **kw)
    return f.reshape(b, k)


# ---------------------------------------------------------------------------
# Danh gia CHINH XAC (mo hinh giai tich) -- dung de bao cao
# ---------------------------------------------------------------------------


def exact_best(levels_np: np.ndarray, ac_set, n_links: int, n_check: int = 8,
               order: Optional[np.ndarray] = None):
    """Kiem chung `n_check` ung vien bang evaluate_config, tra ve cai tot nhat."""
    spec = GenomeSpec(n_ac=len(ac_set), n_links=n_links,
                      allow_link_choice=n_links > 1)
    idx = range(min(n_check, len(levels_np))) if order is None else order[:n_check]
    best_f, best_p, best_q = -np.inf, None, None
    for i in idx:
        params = decode(levels_np[int(i), :len(ac_set)].ravel(), spec)
        res = evaluate_config(params, ac_set)
        f = fitness(res, ac_set)
        if f > best_f:
            best_f, best_p, best_q = f, params, res
    return best_f, best_p, best_q


@torch.no_grad()
def run_policy(policy: PolicyGNN, sur: Surrogate, ac_set, n_links: int,
               n_sample: int = 256, greedy: bool = False):
    """Mot lan suy luan policy -> (levels da xep hang theo surrogate, thoi gian)."""
    dev = sur.device
    t0 = time.perf_counter()
    scen, inp = encode_scenarios([ac_set], n_links, dev)
    logits = policy(inp)
    levels = sample_levels(logits, k=1 if greedy else n_sample, greedy=greedy)
    tables = LevelTables(dev)

    # Xep hang theo HAI thang roi gop lai. Ly do: fitness goc cham moi ca the vi
    # pham bang -lambda*excess, tuc chi nhin do vuot nguong ma BO QUA hoan toan
    # ham muc tieu -- mot cau hinh (excess 0.1, muc tieu 5) se xep tren
    # (excess 0.2, muc tieu 49). Khi chua co ung vien nao kha thi, thang co dinh
    # huong moi chi ra dung cai dang o gan nghiem tot. Ca hai deu duoc kiem chung
    # chinh xac sau do nen viec gop khong lam mat tinh dung dan.
    f = score_levels(levels, scen, tables, n_links, sur.model)[0]
    fs = score_levels(levels, scen, tables, n_links, sur.model, shaped=True)[0]
    o1 = torch.argsort(f, descending=True).cpu().numpy()
    o2 = torch.argsort(fs, descending=True).cpu().numpy()
    order, seen = [], set()
    for a, b in zip(o1, o2):                       # xen ke hai thu tu
        for v in (int(a), int(b)):
            if v not in seen:
                seen.add(v)
                order.append(v)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return levels[0].cpu().numpy(), np.array(order), dt


# ---------------------------------------------------------------------------
# Huan luyen
# ---------------------------------------------------------------------------


def train(args):
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sur = Surrogate.load(args.surrogate, dev)
    for p in sur.model.parameters():
        p.requires_grad_(False)
    tables = LevelTables(dev)
    print(f"thiet bi: {dev} · surrogate: {args.surrogate}")

    rng = np.random.default_rng(args.seed)
    pool = [main_scenario() if rng.random() < 0.5 else sample_scenario(rng)
            for _ in range(args.pool)]
    print(f"kho kich ban: {len(pool)} (50 % la kich ban Sec. V)")

    policy = PolicyGNN(hidden=args.hidden, n_layers=args.layers,
                       n_links=args.links).to(dev)
    print(f"so tham so policy: {sum(p.numel() for p in policy.parameters()):,}")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps,
                                                pct_start=0.1)
    n_elite = max(int(args.k * args.elite_frac), 1)
    acs_main = main_scenario()
    t0 = time.time()

    for step in range(args.steps):
        picks = rng.integers(0, len(pool), size=args.batch)
        scen, inp = encode_scenarios([pool[i] for i in picks], args.links, dev)

        logits = policy(inp)
        with torch.no_grad():
            levels = sample_levels(logits, k=args.k, temperature=args.temp)
            lam_t = args.lam * (1.0 + 9.0 * step / max(args.steps - 1, 1))
            f = score_levels(levels, scen, tables, args.links, sur.model,
                             shaped=True, lam=lam_t, margin=args.margin)
            elite = torch.topk(f, n_elite, dim=1).indices              # (B,E)
        el_lv = torch.gather(
            levels, 1, elite[..., None, None].expand(-1, -1, levels.shape[2],
                                                     levels.shape[3]))

        lp = log_prob(logits, el_lv, inp["mask"])                      # (B,E)
        beta = args.beta * (1.0 - step / args.steps)
        loss = -lp.mean() - beta * entropy(logits, inp["mask"]).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % max(args.steps // 15, 1) == 0 or step == args.steps - 1:
            lv, order, _ = run_policy(policy, sur, acs_main, args.links,
                                      n_sample=128)
            fx, _, q = exact_best(lv, acs_main, args.links, n_check=4,
                                  order=order)
            print(f"buoc {step:5d}/{args.steps}  elite(surrogate) "
                  f"{f.gather(1, elite).mean():7.2f}  "
                  f"| kich ban Sec.V: fitness CHINH XAC {fx:7.3f}  "
                  f"kha thi {str(q.feasible(acs_main)):>5}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"state_dict": policy.state_dict(),
                "arch": dict(hidden=args.hidden, n_layers=args.layers,
                             n_links=args.links)}, args.out)
    print(f"\nDa luu policy -> {args.out}")
    return policy, sur


def load_policy(path: str, device: str) -> PolicyGNN:
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = PolicyGNN(**ck["arch"]).to(device)
    pol.load_state_dict(ck["state_dict"])
    return pol.eval()


# ---------------------------------------------------------------------------
# Bao cao cuoi
# ---------------------------------------------------------------------------


def final_report(policy: PolicyGNN, sur: Surrogate, n_links: int,
                 n_sample: int = 256):
    from common.utility import summarize

    acs = main_scenario()
    print("\n" + "=" * 74)
    print("KET QUA -- kich ban Sec. V, moi fitness deu do MO HINH GIAI TICH tinh")
    print("=" * 74)

    rows = []
    for tag, greedy, ns, ncheck in (
            ("Policy (greedy, 1 mau)", True, 1, 1),
            (f"Policy ({n_sample} mau, surrogate xep hang)", False, n_sample, 8)):
        lv, order, dt = run_policy(policy, sur, acs, n_links,
                                   n_sample=ns, greedy=greedy)
        fx, params, q = exact_best(lv, acs, n_links, n_check=ncheck, order=order)
        rows.append((tag, dt, fx, q.feasible(acs), params, q))

    print(f"{'phuong phap':44} {'thoi gian':>12} {'fitness':>9} {'kha thi':>8}")
    print("-" * 74)
    for tag, dt, fx, ok, _, _ in rows:
        print(f"{tag:44} {dt*1000:9.1f} ms {fx:9.3f} {str(ok):>8}")
    print(f"{'GA + GNN surrogate (muc 9.6)':44} {15.7:9.1f} s  {49.370:9.3f} "
          f"{'True':>8}")
    print(f"{'GA goc (muc 9.6)':44} {157.0:9.1f} s  {49.704:9.3f} {'True':>8}")
    print("-" * 74)

    best = max(rows, key=lambda r: r[2])
    print(f"\n--- Nghiem tot nhat cua policy ({best[0]}) ---")
    print(describe(best[4]))
    print(summarize(best[5], acs))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Huan luyen GNN policy (PA2)")
    ap.add_argument("--surrogate", type=str, default="results/gnn_model.pt")
    ap.add_argument("--out", type=str, default="results/policy.pt")
    ap.add_argument("--policy", type=str, default="results/policy.pt")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32, help="so kich ban moi buoc")
    ap.add_argument("--k", type=int, default=64, help="so mau moi kich ban")
    ap.add_argument("--elite-frac", type=float, default=0.125)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02, help="thuong entropy")
    ap.add_argument("--lam", type=float, default=2.0,
                    help="he so phat BAN DAU; tang dan toi 10x cuoi qua trinh")
    ap.add_argument("--margin", type=float, default=0.45,
                    help="bien an toan (bac) -- dat bang do bat dinh cua surrogate")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--links", type=int, default=2)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--n-sample", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()

    if a.eval_only:
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        final_report(load_policy(a.policy, dev), Surrogate.load(a.surrogate, dev),
                     a.links, a.n_sample)
    else:
        pol, sur = train(a)
        final_report(pol, sur, a.links, a.n_sample)

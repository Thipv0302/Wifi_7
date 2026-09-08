"""Hinh ket qua cho PHUONG PHAP 2 (GNN policy) -- song song voi fig07-10 cua PP1."""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotting.style import paper_style, BENCH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "results", "pp2_policy.json"), encoding="utf-8"))
B = json.load(open(os.path.join(ROOT, "results", "benchmark.json"), encoding="utf-8"))
C_GA, C_SUR, C_SEED, C_POL = (BENCH["ga"]["c"], BENCH["ga_gnn"]["c"],
                              BENCH["ga_gnn_policy"]["c"], BENCH["policy"]["c"])
C_RND = "#8c8c8c"
i200 = B["pops"].index(200)
T_GA = B["sweep"]["ga"][i200]["wall_med"]
T_SUR = B["sweep"]["ga_gnn"][i200]["wall_med"]
T_SEED = B["sweep"]["ga_gnn_policy"][i200]["wall_med"]
F_GA = B["sweep"]["ga"][i200]["fit_med"]
F_SEED = B["sweep"]["ga_gnn_policy"][i200]["fit_med"]
T_TRAIN = 222.0                       # tu policy_log.txt


def legend_below(fig, ax, ncol=4, y=-0.02, anchor=0.5):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=8.8,
               bbox_to_anchor=(anchor, y), frameon=True, edgecolor="black",
               framealpha=1.0, fancybox=False)


def save(fig, name):
    p = os.path.join(ROOT, "figures", name + ".png")
    fig.savefig(p, bbox_inches="tight", dpi=190)
    plt.close(fig)
    print("saved", p)
    return p


# ==========================================================  Fig. 12
paper_style()
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
e4 = D["e4"]
st, el = np.array(e4["step"]), np.array(e4["elite"])
ex, ok = np.array(e4["exact"]), np.array(e4["feas"])
a = ax[0]
a.plot(st, el, color=C_POL, lw=2.0, marker="o", ms=4,
       label="Điểm elite (surrogate chấm)")
a.scatter(st[ok], ex[ok], s=55, marker="^", c=C_SEED, edgecolors="black",
          lw=0.7, zorder=4, label="Kiểm chứng chính xác — khả thi")
a.scatter(st[~ok], ex[~ok], s=55, marker="x", c="#C0392B", lw=1.6,
          zorder=4, label="Kiểm chứng chính xác — vi phạm")
a.axhline(D["e1"]["greedy"]["fit"], color=C_POL, ls="--", lw=1.4,
          label="Checkpoint trên đĩa hiện đạt 47.76")
a.set_xlabel("Bước huấn luyện")
a.set_ylabel("Fitness")
a.set_title("(a) Huấn luyện policy (3000 bước, 222 s)")
a.set_ylim(-9, 58)
a.text(0.34, 0.28, "log huấn luyện thuộc một lần chạy KHÁC\nvới checkpoint đang có trên đĩa",
       transform=a.transAxes, fontsize=7.8, color="#C0392B", va="bottom")

b = ax[1]
n = np.arange(1, 51)
b.plot(n, n * T_GA, color=C_GA, ls="-", lw=2.0, label="GA gốc")
b.plot(n, n * T_SUR, color=C_SUR, ls="--", lw=2.0, label="GA + GNN surrogate")
b.plot(n, T_TRAIN + n * T_SEED, color=C_SEED, ls="-", lw=2.0,
       label="GA + GNN + policy seeding")
b.plot(n, T_TRAIN + n * D["e1"]["greedy"]["wall"], color=C_POL, ls=":", lw=2.4,
       label="Policy đơn thuần")
for T, c in ((T_SEED, C_SEED), (D["e1"]["greedy"]["wall"], C_POL)):
    x = T_TRAIN / (T_GA - T)
    b.plot([x], [T_TRAIN + x * T], marker="*", ms=13, c=c, mec="black",
           mew=0.7, zorder=5)
b.set_xscale("log")
b.set_yscale("log")
b.set_xlabel("Số kịch bản cần giải")
b.set_ylabel("Tổng thời gian tích luỹ (s)")
b.set_title("(b) Khấu hao chi phí huấn luyện offline")
b.text(0.97, 0.05,
       "Dấu sao = điểm hoà vốn ≈ 2 kịch bản\n(222 s huấn luyện policy;\nchưa tính sinh dữ liệu + surrogate)",
       transform=b.transAxes, ha="right", va="bottom", fontsize=7.6, color="#555555")
h0, l0 = ax[0].get_legend_handles_labels()
h1, l1 = ax[1].get_legend_handles_labels()
fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=4, fontsize=8.4,
           bbox_to_anchor=(0.5, -0.21), frameon=True, edgecolor="black",
           framealpha=1.0, fancybox=False)
fig.suptitle("PP2 — GNN policy: huấn luyện một lần, dùng lại mãi mãi", fontsize=12, y=1.0)
fig.tight_layout()
save(fig, "fig12_policy_training")

# ==========================================================  Fig. 13
paper_style()
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
ks = D["e1"]["ks"]


def med(src, key):
    return np.array([np.median(D["e1"][src][str(k)][key]) for k in ks])


pw, pf = med("policy", "wall"), med("policy", "fit")
rw, rf = med("random", "wall"), med("random", "fit")
g = D["e1"]["greedy"]

panels = ((ax[0], pw * 1e3, rw * 1e3, "Thời gian một lần suy luận (ms)",
           "(a) Chất lượng theo THỜI GIAN"),
          (ax[1], np.array(ks, float), np.array(ks, float),
           "Số mẫu $K$ (surrogate xếp hạng)", "(b) Chất lượng theo SỐ MẪU"))
for a, xs_p, xs_r, xl, tt in panels:
    a.plot(xs_p, pf, color=C_POL, ls="-", marker="*", ms=9,
           label="Policy: lấy $K$ mẫu, surrogate chọn tốt nhất")
    a.plot(xs_r, rf, color=C_RND, ls="-.", marker="v", ms=5,
           label="Ngẫu nhiên: lấy $K$ mẫu, surrogate chọn tốt nhất")
    a.axhline(F_SEED, color=C_SEED, ls="-", lw=1.6,
              label="GA + GNN + policy seeding (%.2f · %.1f s)" % (F_SEED, T_SEED))
    a.axhline(F_GA, color=C_GA, ls="--", lw=1.4,
              label="GA gốc, trung vị 10 seed (%.2f · %.0f s)" % (F_GA, T_GA))
    a.axhline(0.0, color="#bbbbbb", lw=0.9)
    a.set_xscale("log")
    a.set_xlabel(xl)
    a.set_ylabel("Fitness (mô hình giải tích)")
    a.set_title(tt)
    a.set_ylim(-13, 60)
ax[0].plot([g["wall"] * 1e3], [g["fit"]], marker="P", ms=11, c=C_POL,
           mec="black", mew=0.8, zorder=5)
ax[0].annotate("policy greedy\n1 mẫu · 5.8 ms · fitness 47.76",
               (g["wall"] * 1e3, g["fit"]), xytext=(0.26, 0.70),
               textcoords="axes fraction", fontsize=8.0, color="#333333",
               arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))
ax[1].text(0.5, 0.32, "lấy thêm mẫu KHÔNG giúp gì:\npolicy đã hội tụ về một cấu hình",
           transform=ax[1].transAxes, ha="center", fontsize=8.2, color=C_POL)
ax[1].text(0.5, 0.14, "ngẫu nhiên: 0/7 khả thi ở MỌI $K$ đến 4096",
           transform=ax[1].transAxes, ha="center", fontsize=8.2, color="#C0392B")
legend_below(fig, ax[0], ncol=2, y=-0.19)
fig.suptitle("PP2 — đường anytime ở thang mili-giây (kịch bản Sec. V, trung vị 7 lần)",
             fontsize=12, y=1.0)
fig.tight_layout()
save(fig, "fig13_policy_anytime")

# ==========================================================  Fig. 14
paper_style()
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
e2 = D["e2"]
bins = np.linspace(-10, 52, 63)
sets = [("random", "Ngẫu nhiên", C_RND),
        ("policy_t1.6", "Policy, $T$=1.6", C_SEED),
        ("policy_t1.0", "Policy, $T$=1.0", C_POL)]
for key, lab, c in sets:
    f = np.array(e2[key]["fit"])
    ax[0].hist(f, bins=bins, color=c, alpha=0.78, label=lab + " (n=2000)",
               edgecolor="black", linewidth=0.35)
ax[0].set_yscale("log")
ax[0].set_xlabel("Fitness của MẪU THÔ (chưa xếp hạng)")
ax[0].set_ylabel("Số mẫu")
ax[0].set_title("(a) Phân bố chất lượng mẫu thô")
ax[0].axvline(0, color="#C0392B", ls="--", lw=1.2)
ax[0].text(0.5, 0.93, "fitness < 0  ⇔  VI PHẠM ràng buộc", transform=ax[0].transAxes,
           ha="center", fontsize=8, color="#C0392B")

names = ["Ngẫu nhiên", "Quần thể GA\nkhởi tạo (1/3 gieo)", "Policy\n$T$=1.6", "Policy\n$T$=1.0"]
vals = [100 * np.mean(e2["random"]["feas"]),
        100 * np.mean([r["frac_feasible_init"] for e in B["sweep"]["ga_gnn_policy"]
                       for r in e["runs"]]),
        100 * np.mean(e2["policy_t1.6"]["feas"]),
        100 * np.mean(e2["policy_t1.0"]["feas"])]
cols = [C_RND, C_SEED, C_SEED, C_POL]
x = np.arange(4)
ax[1].bar(x, vals, color=cols, edgecolor="black", lw=0.9, width=0.6)
for xi, v in zip(x, vals):
    ax[1].text(xi, v + 2.5, ("%.2f %%" % v) if (v < 1 or v > 99.9) else ("%.1f %%" % v),
               ha="center", fontsize=9)
ax[1].set_xticks(x)
ax[1].set_xticklabels(names, fontsize=8.6)
ax[1].set_ylabel("Cấu hình khả thi (%)")
ax[1].set_ylim(0, 120)
ax[1].set_title("(b) Tỉ lệ khả thi — bài toán thật nằm ở đây")
ax[1].text(0.03, 0.70, "0 / 2000 mẫu ngẫu nhiên khả thi",
           transform=ax[1].transAxes, fontsize=8.2, color="#C0392B")
legend_below(fig, ax[0], ncol=3, y=-0.10)
fig.suptitle("PP2 — vùng khả thi mỏng tới mức tìm được đường vào nó chính là toàn bộ bài toán",
             fontsize=12, y=1.0)
fig.tight_layout()
save(fig, "fig14_policy_reliability")

# ==========================================================  Fig. 15
paper_style()
fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
acn, eps = D["ac"], np.array(D["eps"])
pol_v = np.array(D["e3"]["greedy"]["viol"]) / eps
ga_v = np.array(D["e3"]["ga"]["viol"]) / eps
pol_c = -np.log10(np.maximum(D["e3"]["greedy"]["p_loss"], 1e-10))
ga_c = -np.log10(np.maximum(D["e3"]["ga"]["p_loss"], 1e-10))
x = np.arange(len(acn))
w = 0.38

ax[0].bar(x - w / 2, pol_c, w, color=C_POL, edgecolor="black", lw=0.8,
          label="Policy đơn thuần (tổng %.2f)" % sum(pol_c))
ax[0].bar(x + w / 2, ga_c, w, color=C_SEED, edgecolor="black", lw=0.8,
          label="GA + GNN + policy (tổng %.2f)" % sum(ga_c))
ax[0].axhline(10.0, color="#C0392B", ls="--", lw=1.2)
ax[0].text(-0.42, 10.28, "trần $P_{loss}=10^{-10}$", ha="left", fontsize=8, color="#C0392B")
for xi, (a_, b_) in enumerate(zip(pol_c, ga_c)):
    if b_ - a_ > 0.3:
        ax[0].annotate("+%.2f" % (b_ - a_), (xi + w / 2, b_ + 0.32), ha="center",
                       fontsize=8.6, color="#C0392B", fontweight="bold")
ax[0].set_xticks(x)
ax[0].set_xticklabels(acn)
ax[0].set_ylabel(r"Đóng góp fitness  $-\log_{10} P_{loss,i}$")
ax[0].set_ylim(0, 11.5)
ax[0].set_title("(a) 1.94 điểm chênh lệch đến từ đâu")

ax[1].bar(x - w / 2, pol_v, w, color=C_POL, edgecolor="black", lw=0.8,
          label="Policy đơn thuần")
ax[1].bar(x + w / 2, np.maximum(ga_v, 1e-6), w, color=C_SEED, edgecolor="black",
          lw=0.8, label="GA + GNN + policy")
ax[1].axhline(1.0, color="#C0392B", ls="--", lw=1.4)
ax[1].text(4.45, 1.3, "ngưỡng vi phạm", ha="right", fontsize=8, color="#C0392B")
ax[1].set_yscale("log")
ax[1].set_xticks(x)
ax[1].set_xticklabels(acn)
ax[1].set_ylabel(r"$\Pr(D \geq D_{max})\, /\, \varepsilon_i$")
ax[1].set_ylim(1e-6, 8)
ax[1].set_title("(b) Policy để thừa biên an toàn, GA thì tiêu hết")
legend_below(fig, ax[0], ncol=2, y=-0.07)
fig.suptitle("PP2 — vì sao policy đơn thuần mất 1.94 điểm so với GA (47.76 vs 49.70)",
             fontsize=12, y=1.0)
fig.tight_layout()
save(fig, "fig15_policy_margin")

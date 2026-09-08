"""make_figures.py -- every figure of Paper 1.

    py -3 make_figures.py

The manuscript is six pages, so it carries THREE result figures and puts the
rest of the evidence in Table I. Those three are the ones main.tex includes:

  fig08_sweep_methods   Does it hold when the scenario changes, and which part
                        of the pipeline holds it up?  The tightest threshold is
                        swept over five decades and all three ablation levels
                        are re-solved at each point.  Panel (a) is bounded below
                        by monotonicity -- F does not contain epsilon, so
                        relaxing it cannot lower the optimum -- and panel (b),
                        which plots a quantity that is not the objective, is
                        not.  A dip means different things in the two panels.

  fig09_qos_methods     Does the returned configuration meet the constraints?
                        Per-category delay violation and packet loss for the
                        default table and all three levels.  Packet loss is
                        worth a panel only because the screened-but-unseeded
                        level is present; without it every optimised
                        configuration sits on the reporting floor.

  fig10_gnn_training    Do the two networks converge, and to what?  Both on one
                        objective axis, which means scoring the evaluation
                        network by the ranking it induces rather than by its
                        regression error -- the latter cannot be drawn beside a
                        curve in objective points.

The remaining functions produce figures kept for the record and for talks; they
are not included by main.tex.  See FIGURES.md.

Every value plotted is computed by the analytical model; the surrogate ranks
candidates during search and never reports a number that reaches a figure.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

import ieee_style as S

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "Image")

with open(os.path.join(HERE, "data", "pp1_paper_figs.json"),
          encoding="utf-8") as f:
    D = json.load(f)

C_DEF = "#7F7F7F"                                  # default EDCA table
C_GA = S.METHOD_STYLE["ga"]["c"]                   # genetic baseline
C_PIPE = S.METHOD_STYLE["ga_gnn_policy"]["c"]      # our pipeline
C_EVAL = S.METHOD_STYLE["ga_gnn"]["c"]

LAB_GA = "GA baseline (Yi et al.)"
LAB_PIPE = "GNN pipeline (proposed)"

# One name per method, shared by every figure.  "Evaluation net" and "proposal
# net" alone do not say what the boxes are, so each label carries the kind of
# network, the symbol used in Sections IV-B and IV-C, and the job it does.
LAB_EVAL = r"$+$ evaluation GNN $g_\phi$ (ranks candidates)"
LAB_PROP = r"$+$ proposal GNN $\pi_\psi$ (seeds the population)"
# Short forms, for panels where the full label will not fit.
LAB_EVAL_S = r"$+\,g_\phi$ (evaluation GNN)"
LAB_PROP_S = r"$+\,\pi_\psi$ (proposal GNN)"
# Two-line forms for the bar chart, whose four categories sit side by side and
# collide as soon as a label is more than about a dozen characters wide.
LAB_EVAL_2 = "\n".join([r"$+\,g_\phi$", "(evaluation GNN)"])
LAB_PROP_2 = "\n".join([r"$+\,\pi_\psi$", "(proposal GNN)"])
M_LAB = {"ga": LAB_GA, "ga_gnn": LAB_EVAL, "ga_gnn_policy": LAB_PROP}
M_LAB_S = {"ga": LAB_GA, "ga_gnn": LAB_EVAL_S, "ga_gnn_policy": LAB_PROP_S}

with open(os.path.join(HERE, "data", "benchmark.json"), encoding="utf-8") as f:
    B = json.load(f)
METHODS = ["ga", "ga_gnn", "ga_gnn_policy"]
BAR_C = {"ga": C_GA, "ga_gnn": C_EVAL, "ga_gnn_policy": C_PIPE}
with open(os.path.join(HERE, "data", "pipeline_ngen1.json"),
          encoding="utf-8") as f:
    NG1 = json.load(f)["160x1"]        # N_pop = 160, one refinement generation


# ===========================================================================
# Fig. 2 -- convergence of the proposal network against a heuristic baseline
# ===========================================================================
def fig_convergence():
    e1 = D["e1"]
    step = np.array(e1["step"], float)
    exact = np.array(e1["exact"], float)
    ok = np.array(e1["feasible"], bool)

    fig, ax = plt.subplots(figsize=(S.COL, 2.30))

    # The GA has no training axis: it is a single number per run.  Drawing it
    # as a horizontal band is the comparison the reader needs.
    ax.axhline(D["ga_fit_max"], color=C_GA, ls="-", lw=1.0,
               label="%s, best of 10 seeds" % LAB_GA)
    ax.axhline(D["ga_fit_med"], color=C_GA, ls="--", lw=1.0,
               label="%s, median" % LAB_GA)
    ax.axhspan(D["ga_fit_med"], D["ga_fit_max"], color=C_GA, alpha=0.07,
               zorder=0)

    ax.plot(step, exact, color=C_PIPE, lw=1.2, zorder=3,
            label="Proposal network (this work)")
    ax.plot(step[ok], exact[ok], "^", color=C_PIPE, ms=3.6, mec="black",
            mew=0.4, ls="none", zorder=4, label="Feasible")
    ax.plot(step[~ok], exact[~ok], "x", color=S.RED, ms=3.8, mew=1.0,
            ls="none", zorder=4, label="Infeasible")

    first = int(np.argmax(ok)) if ok.any() else None
    if first is not None:
        ax.annotate("first feasible\nat step %d" % step[first],
                    (step[first], exact[first]), xytext=(0.30, 0.62),
                    textcoords="axes fraction", fontsize=S.PT_NOTE,
                    color="#333333",
                    arrowprops=dict(arrowstyle="->", color=S.GREY, lw=0.6))

    ax.set_xlabel("Cross-entropy training step")
    ax.set_ylabel("Objective $F$ (analytical model)")
    ax.set_ylim(-9, 60)
    # The curve occupies the top of the frame after step 300, so the legend
    # goes inside at the lower right rather than below: a legend strip under a
    # single-column figure costs almost as much height as the plot itself.
    ax.legend(loc="lower right", fontsize=S.PT_NOTE, frameon=True,
              edgecolor="black", framealpha=1.0, fancybox=False,
              borderpad=0.3, labelspacing=0.22, handlelength=1.6,
              handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    return S.save(fig, IMG, "fig02_convergence")


# ===========================================================================
# Fig. 3 -- per-category QoS of the returned configuration
# ===========================================================================
def fig_qos():
    e2 = D["e2"]
    acs = D["ac"]
    eps = np.array(D["eps"], float)
    x = np.arange(len(acs))
    w = 0.26
    series = (("Default EDCA", C_DEF, "Default EDCA"),
              ("Opt. MLO EDCA", C_GA, LAB_GA),
              ("GNN pipeline", C_PIPE, LAB_PIPE))

    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.30))

    # A bar whose value equals the axis minimum has zero height on a log scale
    # and disappears.  Both optimised configurations sit at 1e-11..2e-10, so the
    # floor of each panel is set a decade below the smallest value it must show.
    LO_A, LO_B = 1e-9, 1e-12

    ax = axes[0]
    for i, (k, c, lab) in enumerate(series):
        v = np.maximum(np.array(e2[k]["violation"], float), LO_A)
        ax.bar(x + (i - 1) * w, v, w, color=c, edgecolor="black", lw=0.4,
               label=lab, zorder=3)
    ax.plot(x, eps, "_", color=S.RED, ms=16, mew=1.4, ls="none", zorder=5,
            label=r"Target $\varepsilon_i$")
    ax.set_yscale("log")
    ax.set_ylim(LO_A, 5)
    ax.set_xticks(x)
    ax.set_xticklabels(acs)
    ax.set_xlabel("Access category")
    ax.set_ylabel(r"$\Pr(D \geq D_{\max})$")
    ax.set_title(r"(a) Delay violation probability")
    ax.text(0.02, 0.96, r"bars below $10^{-9}$ are clipped",
            transform=ax.transAxes, va="top", fontsize=S.PT_NOTE, color=S.GREY)

    # Panel (b) does NOT repeat packet loss.  On that axis the two optimised
    # configurations differ by 0.01-0.51 decades and both sit at the reporting
    # floor, so the panel would show only that optimisation beats the default
    # table -- which is the reference work's result, not ours.  It carries the
    # reliability comparison instead, which is what separates the pipeline from
    # the search it replaces.
    ax = axes[1]
    pops = B["pops"]
    xs = np.arange(len(pops))
    wb = 0.26
    best = max(r["fitness"] for m in METHODS
               for e in B["sweep"][m] for r in e["runs"])
    thr = 0.99 * best
    for i, m in enumerate(METHODS):
        v = [100.0 * np.mean([r["fitness"] >= thr for r in e["runs"]])
             for e in B["sweep"][m]]
        bars = ax.bar(xs + (i - 1) * wb, v, wb, color=BAR_C[m],
                      edgecolor="black", lw=0.4, zorder=3)
        for b, val in zip(bars, v):
            if val > 0:
                ax.text(b.get_x() + b.get_width() / 2, val + 2.0, "%.0f" % val,
                        ha="center", va="bottom", fontsize=S.PT_NOTE)
    tot = {m: sum(int(round(len(e["runs"]) *
                            np.mean([r["fitness"] >= thr for r in e["runs"]])))
                  for e in B["sweep"][m]) for m in METHODS}
    ax.text(0.02, 0.97, "over all 70 runs:  baseline %d/70,\n"
            r"$+\,g_\phi$ %d/70,  $+\,g_\phi+\pi_\psi$ %d/70"
            % (tot["ga"], tot["ga_gnn"], tot["ga_gnn_policy"]),
            transform=ax.transAxes, va="top", fontsize=S.PT_NOTE,
            color="#333333")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(xs)
    ax.set_xticklabels([str(p) for p in pops])
    ax.set_xlabel(r"Population size $N_{\mathrm{pop}}$")
    ax.set_ylabel(r"Runs reaching $F \geq %.2f$ (%%)" % thr)
    ax.set_title("(b) Success rate over 10 seeds per budget")

    # Panel (b) introduces a colour panel (a) does not use; the shared legend
    # must carry it or the middle bar of every triple is unlabelled.
    from matplotlib.patches import Patch
    h, l = axes[0].get_legend_handles_labels()
    h.append(Patch(facecolor=C_EVAL, edgecolor="black", lw=0.4))
    l.append(r"Evaluation GNN $g_\phi$ only (b)")
    fig.tight_layout(pad=0.3)
    S.legend_below(fig, axes[0], ncol=5, y=-0.16, handles=h, labels=l)
    return S.save(fig, IMG, "fig03_qos")


# ===========================================================================
# Fig. 4 -- behaviour when the tightest threshold moves
# ===========================================================================
def fig_sweep():
    e3 = D["e3"]
    eps1 = np.array(e3["eps1"], float)
    tgt = np.array(e3["target_sum_theta"], float)

    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.30))

    ax = axes[0]
    ax.plot(eps1, e3["ga_mlo"]["fitness"], "--^", color=C_GA, ms=4.0,
            mec="black", mew=0.4, label=LAB_GA)
    ax.plot(eps1, e3["pipeline"]["fitness"], "-s", color=C_PIPE, ms=4.0,
            mec="black", mew=0.4, label=LAB_PIPE)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel("Objective $F$")
    ax.set_title("(a) Objective against the tightest threshold")

    ax = axes[1]
    ax.plot(eps1, e3["ga_mlo"]["sum_theta"], "--^", color=C_GA, ms=4.0,
            mec="black", mew=0.4, label=LAB_GA)
    ax.plot(eps1, e3["pipeline"]["sum_theta"], "-s", color=C_PIPE, ms=4.0,
            mec="black", mew=0.4, label=LAB_PIPE)
    ax.plot(eps1, tgt, ":", color="black", lw=1.0, label=r"Target $\varepsilon$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel(r"$\sum_i \theta_i$")
    ax.set_title("(b) Sum of reliability indices")

    fig.tight_layout(pad=0.3)
    S.legend_below(fig, axes[1], ncol=3, y=-0.16)
    return S.save(fig, IMG, "fig04_epsilon_sweep")


# ===========================================================================
# Fig. 5 -- what one solve costs, component by component
# ===========================================================================
def fig_cost():
    """Cost of a single solve, on the two axes that are actually paid.

    Both are logarithmic because the configurations span two orders of
    magnitude, and each bar carries the objective it achieves so the reader
    cannot read a cost reduction without seeing what it bought: the two cheapest
    configurations are also the two best.
    """
    i = B["pops"].index(200)
    cfg = []
    for k, lab in (("ga", "GA baseline"), ("ga_gnn", LAB_EVAL_2),
                   ("ga_gnn_policy", LAB_PROP_2)):
        e = B["sweep"][k][i]
        cfg.append((lab, e["wall_med"], e["eval_med"], e["fit_med"], BAR_C[k]))
    cfg.append(("\n".join(["Pipeline", r"$N_{\mathrm{gen}}\!=\!1$"]),
                float(np.median(NG1["wall"])), float(np.median(NG1["eval"])),
                float(np.median(NG1["fit"])), C_PIPE))

    labs = [c[0] for c in cfg]
    x = np.arange(len(cfg))
    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.45))

    panels = ((axes[0], 1, "Wall-clock time per solve (s)",
               "(a) Time", r"$58\times$", "%.4g s"),
              (axes[1], 2, "Analytical-model calls per solve",
               "(b) Calls to the analytical model", r"$206\times$", "%.0f"))
    for ax, idx, ylabel, title, factor, fmt in panels:
        v = np.array([c[idx] for c in cfg], float)
        ax.bar(x, v, 0.58, color=[c[4] for c in cfg], edgecolor="black",
               lw=0.5, zorder=3)
        for xi, (val, c) in enumerate(zip(v, cfg)):
            ax.annotate((fmt % val) + "\n$F=%.1f$" % c[3], (xi, val),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=S.PT_NOTE)
        ax.set_yscale("log")
        ax.set_ylim(v.min() / 4.0, v.max() * 30)
        # One bracket from the incumbent to the proposed configuration: the
        # comparison the paper is making, drawn rather than left to arithmetic.
        top = v.max() * 7
        ax.annotate("", xy=(len(cfg) - 1, top), xytext=(0, top),
                    arrowprops=dict(arrowstyle="<->", color=S.RED, lw=0.9))
        ax.text((len(cfg) - 1) / 2.0, top * 1.35, factor, ha="center",
                va="bottom", fontsize=S.PT_LABEL, color=S.RED)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=S.PT_NOTE)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", which="major", lw=0.35, color="#D5D5D5")

    fig.tight_layout(pad=0.3)
    return S.save(fig, IMG, "fig05_cost")


# ===========================================================================
# Fig. 6 -- what more search buys, per method
# ===========================================================================
def fig_budget():
    r"""Quality and cost against search budget, for the three ablation levels.

    Fig.~5 compares the three methods at ONE budget, which cannot say whether
    the baseline would catch up given more search.  Here the budget is the x
    axis, so the question is answered rather than assumed.

    Both panels are indexed by $N_{pop}$ rather than by analytical-model calls.
    Calls are what the methods spend differently -- that is panel (b) -- but
    they are not monotone in the budget for the screened search: a larger
    population lets the surrogate reject more candidates before the exact model
    is called, so plotting against calls makes that curve double back and cross
    its own path.
    """
    THR = 49.21                       # success threshold used throughout
    pops = np.array(B["pops"], float)
    labs = [("%d" % p) for p in pops]
    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.45))

    for k, lab in (("ga", LAB_GA), ("ga_gnn", LAB_EVAL),
                   ("ga_gnn_policy", LAB_PROP)):
        st = S.METHOD_STYLE[k]
        e = B["sweep"][k]
        med = np.array([x["fit_med"] for x in e], float)
        lo = np.array([x["fit_min"] for x in e], float)
        hi = np.array([x["fit_max"] for x in e], float)
        calls = np.array([x["eval_med"] for x in e], float)

        axes[0].fill_between(pops, lo, hi, color=st["c"], alpha=0.12, lw=0)
        axes[0].plot(pops, med, ls=st["ls"], marker=st["m"], color=st["c"],
                     ms=3.6, mec="black", mew=0.4, label=lab)
        axes[1].plot(pops, calls, ls=st["ls"], marker=st["m"], color=st["c"],
                     ms=3.6, mec="black", mew=0.4, label=lab)

    ax = axes[0]
    ax.axhline(THR, color=S.RED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.03, THR + 0.8, r"success threshold $F \geq 49.21$",
            transform=ax.get_yaxis_transform(), fontsize=S.PT_NOTE,
            color=S.RED, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(pops)
    ax.set_xticklabels(labs)
    ax.minorticks_off()
    ax.set_xlabel(r"Population size $N_{\mathrm{pop}}$")
    ax.set_ylabel("Objective $F$")
    ax.set_title("(a) What more search buys (median and range, $10$ seeds)")

    ax = axes[1]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(pops)
    ax.set_xticklabels(labs)
    ax.minorticks_off()
    ax.set_xlabel(r"Population size $N_{\mathrm{pop}}$")
    ax.set_ylabel("Analytical-model calls per solve")
    ax.set_title("(b) What it costs")

    fig.tight_layout(pad=0.3)
    S.legend_below(fig, axes[0], ncol=3, y=-0.16)
    return S.save(fig, IMG, "fig06_budget")


# ===========================================================================
# Fig. 7 -- cost against quality on the threshold sweep, per method
# ===========================================================================
with open(os.path.join(HERE, "data", "pp1_budget_methods.json"),
          encoding="utf-8") as f:
    BM = json.load(f)

def fig_methods_budget():
    r"""Where the gain comes from, measured across all five thresholds.

    Fig.~6 varies the budget on the single scenario of Table~\ref{tab:scenario}.
    This one repeats that sweep at every threshold of Fig.~4 and reports the
    mean over the five, so a method cannot look good by suiting one scenario.

    Ten independent restarts per (method, budget, threshold).  Panel (a) plots
    the best of the ten, which is what a practitioner willing to pay for
    restarts would obtain, and the median, which is what one run returns.  The
    distance between the two lines is the unreliability of the search: it is
    the whole plot for the two unseeded methods and invisible for the seeded
    pipeline, whose median and best coincide at every budget.
    """
    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.45))

    ax = axes[0]
    for k in ("ga", "ga_gnn", "ga_gnn_policy"):
        st, e = S.METHOD_STYLE[k], BM["methods"][k]
        ax.plot(e["calls"], e["best_mean"], ls=st["ls"], marker=st["m"],
                color=st["c"], ms=3.6, mec="black", mew=0.4, label=M_LAB[k])
        ax.plot(e["calls"], e["median_mean"], ls=(0, (1, 1.6)), marker=st["m"],
                color=st["c"], ms=2.6, mfc="white", mew=0.5, lw=0.9)
        ax.fill_between(e["calls"], e["median_mean"], e["best_mean"],
                        color=st["c"], alpha=0.10, lw=0)
    ax.axhline(50, color=S.GREY, lw=0.6, ls=(0, (1, 2)))
    ax.text(0.97, 50.4, "ceiling $F = 50$", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=S.PT_NOTE, color=S.GREY)
    ax.set_xscale("log")
    ax.set_xlabel("Analytical-model calls per run")
    ax.set_ylabel(r"Mean $F$ over the five $\varepsilon_1$")
    ax.set_title("(a) Solid: best of $10$ restarts.  Dotted: median")
    ax.set_ylim(15, 54)

    # Panel (b): the cheapest budget, threshold by threshold.  At this cost the
    # two unseeded searches have not entered the good region at any threshold.
    ax = axes[1]
    eps1 = np.array(BM["eps1"], float)
    for k in ("ga", "ga_gnn", "ga_gnn_policy"):
        st, e = S.METHOD_STYLE[k], BM["methods"][k]
        ax.plot(eps1, e["best"][0], ls=st["ls"], marker=st["m"], color=st["c"],
                ms=3.6, mec="black", mew=0.4,
                label="%s, %d calls" % (M_LAB_S[k], round(e["calls"][0])))
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon_1$")
    ax.set_ylabel("$F$, best of $10$ restarts")
    ax.set_title("(b) At the smallest budget, threshold by threshold")
    ax.legend(fontsize=S.PT_NOTE, loc="lower right")
    ax.set_ylim(15, 54)

    fig.tight_layout(pad=0.3)
    S.legend_below(fig, axes[0], ncol=3, y=-0.16)
    return S.save(fig, IMG, "fig07_methods_budget")


# ===========================================================================
# Fig. 8 -- the threshold sweep with every method on it
# ===========================================================================
# Budget per method for Fig. 8.  A common (Npop, Ngen) is not a common budget:
# it costs the three methods 12,426, 1,070 and 474 calls to the analytical
# model, which is the scarce quantity.  The three entries below are matched on
# THAT instead -- 526, 533 and 474 calls, within 12% of each other -- so the
# figure compares what each method does with the same compute.  The legend
# carries each curve's cost; without it an unequal-(Npop, Ngen) figure would be
# unreadable.  Each is 40 restarts per point, not 10: at 10 the standard error
# of the mean is 1-3 objective points, enough to invent dips that are not there.
# The baseline runs at the configuration that reproduces the objective its own
# authors report (about 35 at eps_1 = 1e-4; 60x60 gives 40.43, the closest on
# our grid).  Each network runs at the largest configuration we measured -- and
# even there spends FEWER exact calls than the baseline, 1,070 and 474 against
# 3,362, because screening makes each candidate cheaper.  That is the point of
# the surrogate, and the legend carries the three costs so the reader can check
# it rather than take it on trust.
SWEEP_BUDGET = {"ga": "60x60", "ga_gnn": "120x120",
                "ga_gnn_policy": "120x120"}


def fig_sweep_methods(budget=None):
    r"""The threshold sweep with each method at a cost-matched budget.

    Two choices decide what this figure says.

    First, the curve is the MEAN over the ten restarts, not the best of them.
    The best answers "what can this method reach", the mean answers "what does
    one run of it return", and the second is what an operator gets.  The band
    is $\pm$ one standard deviation and is half the result: the seeded
    pipeline's is $0.00$ at four of five thresholds, so its mean is its
    outcome, while the unseeded methods scatter by $4$ to $9$ points.

    Second, the budgets are matched on analytical-model calls rather than on
    $(\Npop, \Ngen)$ -- see `SWEEP_BUDGET`.

    Neither curve here is bounded by the monotonicity of $F^\star$.  That bound
    is on the optimum, which the best-of-ten estimates; a mean is a property of
    the solver's output distribution and may dip, and the baseline's does.  The
    caller prints both checks.
    """
    eps1 = np.array(BM["eps1"], float)
    tgt = np.array(BM["target_sum_theta"], float)
    idx = {k: BM["methods"][k]["budget"].index(v)
           for k, v in SWEEP_BUDGET.items()}
    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.30))

    for k in ("ga", "ga_gnn", "ga_gnn_policy"):
        st, e, b = S.METHOD_STYLE[k], BM["methods"][k], idx[k]
        lab = "%s, $%s$ calls" % (M_LAB_S[k], "{:,}".format(round(e["calls"][b])))
        # Band in (a) only: in (b) the quantity is not the objective, the three
        # curves run within two points of each other, and three overlapping
        # bands would hide both them and the target line.
        panels = ((axes[0], e["mean"][b], e["std"][b]),
                  (axes[1], e["sum_theta_mean"][b], None))
        for ax, mu, sd in panels:
            mu = np.array(mu, float)
            if sd is not None:
                sd = np.array(sd, float)
                ax.fill_between(eps1, mu - sd, mu + sd, color=st["c"],
                                alpha=0.11, lw=0, zorder=1)
            ax.plot(eps1, mu, ls=st["ls"], marker=st["m"], color=st["c"],
                    ms=4.0, mec="black", mew=0.4, zorder=3, label=lab)

    ax = axes[0]
    ax.axhline(50, color=S.GREY, lw=0.6, ls=(0, (1, 2)), zorder=2)
    ax.text(0.02, 49.4, "ceiling $F = 50$", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=S.PT_NOTE, color=S.GREY)
    ax.set_ylabel("Objective $F$")
    # Read the restart count from the data rather than writing it in: it has
    # already changed once, and a stale number in a panel title is a lie.
    n_r = BM["methods"]["ga"]["n_restart"][idx["ga"]]
    ax.set_title("(a) Fitness value, mean of $%d$ restarts" % n_r)

    ax = axes[1]
    ax.plot(eps1, tgt, ":", color="black", lw=1.0, label=r"Target $\varepsilon$")
    ax.set_ylabel(r"$\sum_i \theta_i$")
    ax.set_title("(b) Sum of reliability indices")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\varepsilon_1$")

    fig.tight_layout(pad=0.3)
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    h.append(h2[-1]); l.append(l2[-1])
    S.legend_below(fig, axes[0], ncol=4, y=-0.16, handles=h, labels=l)
    path = S.save(fig, IMG, "fig08_sweep_methods")

    mono = lambda f: all(f[i] <= f[i + 1] + 1e-9 for i in range(len(f) - 1))
    for k in ("ga", "ga_gnn", "ga_gnn_policy"):
        e, b = BM["methods"][k], idx[k]
        print("    %-14s %-11s %6.0f calls, n=%d  mean %s  (non-decr. %s / best %s)"
              % (k, SWEEP_BUDGET[k], e["calls"][b], e["n_restart"][b],
                 " ".join("%5.2f" % v for v in e["mean"][b]),
                 mono(e["mean"][b]), mono(e["best"][b])))
    return path


# ===========================================================================
# Fig. 9 -- QoS of the returned configuration, method by method
# ===========================================================================
with open(os.path.join(HERE, "data", "pp1_qos_methods.json"),
          encoding="utf-8") as f:
    QM = json.load(f)

QM_SERIES = (("default", C_DEF, "Default EDCA"),
             ("ga", C_GA, LAB_GA),
             ("ga_gnn", C_EVAL, LAB_EVAL_S),
             ("ga_gnn_policy", C_PIPE, LAB_PROP_S))


def fig_qos_methods():
    r"""Both QoS quantities of the reference work, for all four configurations.

    Fig.~3 gives panel (a) for three configurations and spends panel (b) on
    reliability instead, because at that point packet loss separates nothing:
    every optimised configuration sits on the reporting floor.  With the two
    ablation levels present the floor is worth showing, since it is where the
    unseeded screened search stops being equivalent to the others.

    The configuration plotted for each method is its BEST of ten independent
    restarts at a common budget.  A single run would mostly plot seed noise:
    the spread between restarts is wider here than the spread between methods,
    which is the finding of Fig.~7 and not something a QoS panel should be
    made to carry.
    """
    acs = QM["ac"]
    eps = np.array(QM["eps"], float)
    x = np.arange(len(acs))
    w = 0.20
    # Log bars need a floor below the data. The default table puts AC2 at
    # 1e-16, thirteen decades below anything else on the panel; giving the axis
    # room for it would compress every comparison that matters into the top
    # fifth of the frame, so the floor is set just below the optimised values
    # and the two bars that fall through it are declared.
    LO_A, LO_B = 1e-11, 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.35))

    for i, (k, c, lab) in enumerate(QM_SERIES):
        off = (i - 1.5) * w
        va = np.maximum(np.array(QM["configs"][k]["violation"], float), LO_A)
        pl = np.maximum(np.array(QM["configs"][k]["p_loss"], float), LO_B)
        axes[0].bar(x + off, va, w, color=c, edgecolor="black", lw=0.4,
                    label=lab, zorder=3)
        axes[1].bar(x + off, pl, w, color=c, edgecolor="black", lw=0.4,
                    label=lab, zorder=3)

    ax = axes[0]
    ax.plot(x, eps, "_", color=S.RED, ms=15, mew=1.4, ls="none", zorder=5,
            label=r"Target $\varepsilon_i$")
    ax.set_ylim(LO_A, 1e3)
    ax.set_ylabel(r"$\Pr(D \geq D_{\max})$")
    ax.set_title("(a) Delay violation probability")
    ax.text(0.02, 0.96, r"default-table bars below $10^{-11}$ are clipped",
            transform=ax.transAxes, va="top", fontsize=S.PT_NOTE, color=S.GREY)

    ax = axes[1]
    ax.axhline(1e-10, color=S.GREY, lw=0.7, ls=(0, (3, 2)), zorder=4)
    # In the frame corner, not on the line: the floor cuts through the bars of
    # every category, so any in-place label sits on top of one.
    ax.text(0.02, 0.96, r"dashed: reporting floor $10^{-10}$",
            transform=ax.transAxes, va="top", fontsize=S.PT_NOTE, color=S.GREY)
    ax.set_ylim(LO_B, 1e2)
    ax.set_ylabel(r"$P_{\mathrm{loss}}$")
    ax.set_title("(b) Packet loss probability")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(acs)
        ax.set_xlabel("Access category")
        ax.grid(True, axis="y", which="major", lw=0.35, color="#D5D5D5")

    fig.tight_layout(pad=0.3)
    S.legend_below(fig, axes[0], ncol=5, y=-0.17)
    return S.save(fig, IMG, "fig09_qos_methods")


# ===========================================================================
# Fig. 10 -- training convergence of both networks, on one objective axis
# ===========================================================================
with open(os.path.join(HERE, "data", "gnn_train_curve.json"),
          encoding="utf-8") as f:
    TC = json.load(f)


def fig_gnn_training():
    r"""Both networks converging, measured in objective points.

    The two are trained on different problems and cannot share an x axis: one
    counts cross-entropy iterations, the other passes over a labelled set.  The
    y axis is shared, and that is the point -- each network is scored by the
    objective $F$ its output actually achieves under the analytical model, so
    the two curves are the same quantity.

    Panel (a): $\pi_\psi$ emits a configuration, and the curve is the exact $F$
    of that configuration.  Panel (b): $g_\phi$ emits an ordering, so it is
    scored the way the pipeline uses it -- the mean exact $F$ of the $V = 8$
    candidates it forwards for exact evaluation, out of a fixed pool of
    $1{,}560$ scored once in advance.  A regression error would not go on this
    axis; the ranking it induces does.  The dotted ceiling is what a perfect
    ordering of the same pool would score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.45), sharey=True)

    # ---- (a) proposal network -------------------------------------------
    e1 = D["e1"]
    step = np.array(e1["step"], float)
    exact = np.array(e1["exact"], float)
    ok = np.array(e1["feasible"], bool)

    ax = axes[0]
    # One run against one run.  The earlier version drew the baseline's BEST as
    # a solid line, which put it above the network's curve and invited exactly
    # the wrong reading: a best-of-N has N chances where the network curve has
    # one.  The reference is therefore the median, and the ten runs themselves
    # are drawn as a rug at the right edge so the spread is shown rather than
    # asserted -- seven of the ten sit below what one network run returns.
    runs = np.array(D["ga_runs"], float)
    ax.axhspan(D["ga_fit_min"], D["ga_fit_max"], color=C_GA, alpha=0.06,
               zorder=0)
    ax.axhline(D["ga_fit_med"], color=C_GA, ls="--", lw=1.0, zorder=2)
    ax.plot(np.full(len(runs), 0.955), runs, "_", color=C_GA, ms=8, mew=1.1,
            ls="none", transform=ax.get_yaxis_transform(), zorder=4)
    # One label, on the left where the curve is still below it, so it collides
    # with neither the rug nor the other annotation.
    ax.text(0.02, D["ga_fit_med"] - 1.4,
            "GA baseline: median (dashed), range (shaded),\neach of $10$ runs (right)",
            transform=ax.get_yaxis_transform(), ha="left", va="top",
            fontsize=S.PT_NOTE, color=C_GA, linespacing=1.25)

    ax.plot(step, exact, color=C_PIPE, lw=1.2, zorder=3,
            label=r"$\pi_\psi$, one run")
    ax.plot(step[ok], exact[ok], "^", color=C_PIPE, ms=3.6, mec="black",
            mew=0.4, ls="none", zorder=4, label="Feasible")
    ax.plot(step[~ok], exact[~ok], "x", color=S.RED, ms=3.8, mew=1.0,
            ls="none", zorder=4, label="Infeasible")
    ax.set_xlabel("Cross-entropy training step")
    ax.set_ylabel("Objective $F$ (analytical model)")
    # The budget belongs in the title.  The baseline's objective differs from
    # figure to figure only because its budget does, by up to 99x, and a reader
    # who cannot see the budget cannot reconcile 25 here with 50 there.
    ax.set_title(r"(a) Proposal GNN $\pi_\psi$ vs GA at $29{,}309$ exact calls")
    ax.legend(loc="lower right", fontsize=S.PT_NOTE, borderpad=0.3,
              labelspacing=0.22, handlelength=1.6, handletextpad=0.4)

    # ---- (b) evaluation network ------------------------------------------
    c = TC["curve"]
    ep = np.array([r["epoch"] for r in c], float)
    mean_v = np.array([r["exact"] for r in c], float)
    best_v = np.array([r["exact_best"] for r in c], float)
    ceil = TC["pool"]["oracle_topv"]

    ax = axes[1]
    ax.axhline(ceil, color=S.GREY, lw=0.8, ls=(0, (1, 2)), zorder=2)
    ax.text(0.98, ceil - 1.2, "perfect ordering of the same pool",
            transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=S.PT_NOTE, color=S.GREY)
    ax.fill_between(ep, mean_v, best_v, color=C_EVAL, alpha=0.12, lw=0)
    ax.plot(ep, best_v, ls=(0, (1, 1.6)), color=C_EVAL, lw=0.9, zorder=3,
            label=r"best of the $V = 8$ it forwards")
    ax.plot(ep, mean_v, "-s", color=C_EVAL, ms=3.2, mec="black", mew=0.4,
            zorder=4, label=r"mean of the $V = 8$ it forwards")
    ax.set_xlabel("Training epoch")
    ax.set_title(r"(b) Evaluation GNN $g_\phi$, $60$ epochs in "
                 r"$%d$ s" % round(c[-1]["elapsed"]))
    ax.legend(loc="lower right", fontsize=S.PT_NOTE, borderpad=0.3,
              labelspacing=0.22, handlelength=1.6, handletextpad=0.4)

    axes[0].set_ylim(-9, 60)
    fig.tight_layout(pad=0.3)
    return S.save(fig, IMG, "fig10_gnn_training")


# ===========================================================================
# Fig. 11 -- the three methods on one budget axis
# ===========================================================================
with open(os.path.join(HERE, "data", "pp1_anytime.json"), encoding="utf-8") as f:
    AT = json.load(f)


def fig_anytime():
    r"""Objective against compute, which is the only axis the three share.

    Table~\ref{tab:headline} compares them at a common $(\Npop, \Ngen)$, and on
    that comparison the evaluation network is NOT expected to raise the
    objective: it replaces an exact evaluator with an approximate one, so at an
    equal number of candidates examined it should match the baseline at best,
    and carries prediction error besides.  What it buys is that each candidate
    costs far less.

    Whether that buys anything the optimiser can use is a separate question,
    and it is the one this figure answers.  A cheaper evaluation is only
    valuable if the extra candidates it affords translate into a better
    solution at the same total spend.  So the budget goes on the x axis and
    each curve is read as "what had this method reached by then".

    Both panels are read at the same budgets, over twenty seeds, with no early
    stopping so that every curve runs to the end of its budget rather than
    stopping where its own criterion fires.
    """
    t = np.array(AT["t_grid"], float)
    thr = AT["success"]
    fig, axes = plt.subplots(1, 2, figsize=(S.PAGE, 2.45))

    for k in ("ga", "ga_gnn", "ga_gnn_policy"):
        st, a = S.METHOD_STYLE[k], AT["methods"][k]["wall"]
        mu = np.array(a["mean"], float)
        sd = np.array(a["std"], float)
        su = np.array(a["success"], float)
        # Before a method's first generation there is nothing to report; the
        # run-out is NaN rather than extrapolated backwards, since extrapolating
        # there would hand free credit to whichever method starts slowest.
        ok = np.isfinite(mu)
        axes[0].fill_between(t[ok], (mu - sd)[ok], (mu + sd)[ok],
                             color=st["c"], alpha=0.10, lw=0, zorder=1)
        axes[0].plot(t[ok], mu[ok], ls=st["ls"], color=st["c"], lw=1.3,
                     zorder=3, label=M_LAB[k])
        axes[1].plot(t[ok], 100 * su[ok], ls=st["ls"], color=st["c"], lw=1.3,
                     zorder=3, label=M_LAB[k])

    ax = axes[0]
    ax.axhline(thr, color=S.RED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.text(0.02, thr - 1.5, "success threshold $F \\geq %.2f$" % thr,
            transform=ax.get_yaxis_transform(), ha="left", va="top",
            fontsize=S.PT_NOTE, color=S.RED)
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock budget per solve (s)")
    ax.set_ylabel("Objective $F$ reached by then")
    ax.set_title("(a) What each method has reached, mean of $20$ seeds")

    ax = axes[1]
    ax.set_xscale("log")
    ax.set_ylim(-4, 108)
    ax.set_xlabel("Wall-clock budget per solve (s)")
    ax.set_ylabel(r"Seeds reaching $F \geq %.2f$ (\%%)" % thr)
    ax.set_title("(b) How often, at the same budget")

    fig.tight_layout(pad=0.3)
    S.legend_below(fig, axes[0], ncol=3, y=-0.17)
    path = S.save(fig, IMG, "fig11_anytime")

    # The claim the figure exists to support, checked rather than asserted.
    for axis, gk in (("wall", "t_grid"), ("evals", "e_grid")):
        g = np.array(AT[gk], float)
        a = np.array(AT["methods"]["ga_gnn"][axis]["mean"], float)
        b = np.array(AT["methods"]["ga"][axis]["mean"], float)
        c = np.array(AT["methods"]["ga_gnn_policy"][axis]["mean"], float)
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        print("    %-6s pipeline > g_phi > GA at %d of %d shared budgets "
              "(%.3g to %.3g)"
              % (axis, int(((c > a) & (a > b) & m).sum()), int(m.sum()),
                 g[m][0], g[m][-1]))
    return path


if __name__ == "__main__":
    S.use_ieee()
    print("serif family resolved to:", S.check_font())
    fig_convergence()
    fig_qos()
    fig_sweep()
    fig_cost()
    fig_budget()
    fig_methods_budget()
    fig_sweep_methods()
    fig_qos_methods()
    fig_gnn_training()
    fig_anytime()
    print("done ->", IMG)

# Colab package — three result figures and the running-time table

```
colab/
├── wifi7_edca_figures.ipynb    the notebook, 8 code cells
└── data/                       everything it reads
    ├── pp1_budget_methods.json
    ├── pp1_qos_methods.json
    ├── gnn_train_curve.json
    ├── pp1_paper_figs.json
    └── pipeline_ngen1.json
```

## How to run

**Colab.** Upload `wifi7_edca_figures.ipynb`, run Cell 1, and when it asks,
upload the five files from `data/` (Ctrl-click to select all at once). It also
finds them by itself if you mount Drive and place them in `/content/data`.

**Locally.** Keep `data/` beside the notebook and run it; Cell 1 finds the
folder without asking. Needs only `numpy`, `matplotlib` and — for the tables —
`pandas`. No project code is imported, so nothing else has to be installed.

## What each cell does

| cell | does |
|---|---|
| 1 | finds `data/`, or asks Colab to upload it |
| 2 | plotting style, method colours and labels |
| 3 | loads the five JSON files |
| 4 | **Fig. 1** — threshold sweep: (a) fitness value, (b) sum of reliability indices |
| 5 | **Fig. 2** — QoS: (a) delay violation probability, (b) packet loss probability |
| 6 | **Fig. 3** — training convergence of both GNNs, one objective axis |
| 7 | running-time tables |
| 8 | the per-solve table again as LaTeX, ready to paste |

Each figure cell writes a 600 dpi PNG next to the notebook and prints a check
under the plot — monotonicity per method for Fig. 1, per-category feasibility
for Fig. 2, the step and epoch at which each network converges for Fig. 3.

Nothing is re-simulated. Every number was produced by the analytical model in
the main repository and written to these JSON files; the notebook reads and
plots them.

## Reading the figures

**Fig. 1 panel (a) is bounded; panel (b) is not.** The objective
$F = \sum_i -\log_{10} P_{\mathrm{loss},i}$ does not contain $\varepsilon$,
which enters only through the constraints, so relaxing $\varepsilon_1$ only
enlarges the feasible set and the optimum cannot fall — a drop in (a) is a
convergence failure of the solver, not a property of the problem. No such bound
applies to $\sum_i \theta_i$ in (b), which is not the objective, so a dip there
is a property of the solution. The cell prints the monotonicity check.

**Fig. 2 plots each method's best of ten restarts.** On this scenario the spread
between restarts (GA best $49.70$ against median $43.11$) is wider than the
spread between methods, so a single run per method would mostly plot seed noise.

**Fig. 3 scores each network by what its own output achieves.** $\pi_\psi$ emits
a configuration, so it is scored by that configuration's exact $F$. $g_\phi$
emits an ordering, so it is scored the way the pipeline uses it: the mean exact
$F$ of the $V = 8$ candidates it forwards for exact evaluation, out of a fixed
pool of 1,560 scored once in advance. A regression error would not go on this
axis; the ranking it induces does.

## Running time

Two costs, paid at different times.

*Per solve*, on the reference scenario at a common budget, median of ten seeds:

| configuration | time (s) | exact calls | $F$ best | $F$ median | speed-up |
|---|---|---|---|---|---|
| GA baseline | 69.60 | 13 105 | 49.70 | 43.11 | 1.0× |
| `+` evaluation GNN $g_\phi$ | 9.12 | 1 112 | 45.31 | 30.92 | 7.6× |
| `+` proposal GNN $\pi_\psi$ | 4.09 | 554 | 49.70 | 49.70 | 17.0× |
| pipeline as deployed ($N_{pop}=160$, $N_{gen}=1$) | 2.62 | 142 | 49.70 | 49.70 | 26.5× |

Wall-clock depends on the machine (one RTX 4060, 26-core CPU); analytical-model
calls do not, so the calls column is the one that transfers.

*One-off training*, paid once for all scenarios: 159 s to generate the training
set, 401 s for $g_\phi$, 81 s for $\pi_\psi$ — 641 s in total. Reported
separately rather than amortised into a per-solve figure, because how it
amortises depends on how many scenarios a deployment solves.

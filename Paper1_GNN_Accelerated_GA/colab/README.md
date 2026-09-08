# Colab package — the manuscript's figures and table

```
colab/
├── wifi7_edca_figures.ipynb    9 code cells
└── data/                       everything it reads
    ├── pp1_paper_figs.json       Fig. 2(a)
    ├── gnn_train_curve.json      Fig. 2(b)
    ├── pp1_qos_methods.json      Fig. 3
    ├── pp1_budget_methods.json   Fig. 4
    ├── benchmark.json            Table I, upper block
    ├── pipeline_ngen1.json       Table I, upper block
    └── pp1_anytime.json          Table I, lower block
```

## How to run

**Colab.** Upload `wifi7_edca_figures.ipynb`, run Cell 1, and when it asks,
upload the seven files from `data/` (Ctrl-click to select all). It also finds
them by itself if you mount Drive and place them in `/content/data`.

**Locally.** Keep `data/` beside the notebook and run it; Cell 1 finds the
folder without asking. Needs only `numpy`, `matplotlib` and — for the table —
`pandas`. No project code is imported, so nothing else has to be installed.

## Cells

| cell | does |
|---|---|
| 1 | finds `data/`, or asks Colab to upload it |
| 2 | plotting style, method colours and labels |
| 3 | loads the seven JSON files |
| 4 | **Fig. 2** — training convergence of both networks |
| 5 | **Fig. 3** — QoS: delay violation and packet loss |
| 6 | **Fig. 4** — threshold sweep: fitness and sum of reliability indices |
| 7 | **Table I** — cost and quality, both blocks |
| 8 | the upper block again as LaTeX, ready to paste |
| 9 | extra: objective against compute (not in the paper) |

Each figure cell writes a 600 dpi PNG beside the notebook and prints a check
under the plot — where the network sits in the baseline's distribution for
Fig. 2, per-category feasibility for Fig. 3, monotonicity and the strict
ordering for Fig. 4.

Nothing is re-simulated. Every number was produced by the analytical model in
the main repository and written to these files; the notebook reads and plots.

## Three conventions worth knowing before reading the figures

**One run is compared with one run.** Fig. 2 draws the baseline as the *median*
of its ten runs, with the range shaded and the runs marked individually, not as
their best. A best-of-$N$ has $N$ chances where a single network run has one. On
that footing the network's $47.76$ sits above seven of the ten baseline runs.

**Budgets are stated because they differ.** The baseline's objective is not one
number — it depends on what it is given, and across these figures it runs at
3,362, 13,105 and 29,309 analytical-model calls. Fig. 4's legend carries each
curve's cost, and there the quality ordering and the cost ordering run opposite
ways: the pipeline is best *and* cheapest, at 478 calls against the baseline's
3,362.

**The monotonicity bound is on the optimum, not on the mean.** $F$ does not
contain $\varepsilon$, which enters only through the constraints, so relaxing
$\varepsilon_1$ enlarges the feasible set and $F^\star$ cannot fall. That
constrains the best-of-$N$, not the mean Fig. 4 plots, which is a property of the
solver — it happens to be non-decreasing for all three here, at 40 restarts per
point. At ten restarts the standard error is 1–3 objective points, enough to
produce dips that are not there.

## What the two blocks of Table I are for

They answer different questions, and the evaluation GNN looks different in each.

*Upper — one search configuration.* $g_\phi$ is **not** expected to raise the
objective here: it swaps an exact evaluator for an approximate one, so at an
equal number of candidates examined it can at best match the baseline, and it
carries prediction error besides. It returns $34.63$ against $35.04$ — for
$19.8\times$ fewer exact calls. That is the correct expectation, not a
disappointment.

*Lower — one wall-clock budget.* This is where cheaper evaluation has to prove
it buys something the optimiser can use. Over twenty seeds the ordering
$\pi_\psi > g_\phi > \text{baseline}$ holds at every one of the 40 wall-clock
and 48 exact-call budgets measured, without exception.

The honest limit, which cell 9 shows: $g_\phi$ alone settles at $43.01$ because
the surrogate's error caps what screening by it can select, while the baseline
keeps climbing and would cross that ceiling given time beyond the range
measured. Only $\pi_\psi$ lifts the ceiling.

No early stopping is used in the lower block, so each run continues to the end
of its budget — which is why the baseline's totals there (255 s, 52,056 calls)
exceed the upper block's (152.6 s, 29,309 calls). The two are not
interchangeable.

# Figures — what each one shows and where its numbers come from

Build every figure with:

```bash
py -3 make_figures.py
```

All values plotted are computed by the analytical model in
`wifi7_edca/common/utility.py`. The surrogate ranks candidates during search and
never reports a number that reaches a figure.

Method names are fixed once at the top of `make_figures.py` (`LAB_*`, `M_LAB*`)
and shared by every figure, so the three ablation levels are named identically
everywhere:

| key | name in figures | what it is |
|---|---|---|
| `ga` | GA baseline (Yi et al.) | genetic search on the analytical model |
| `ga_gnn` | `+` evaluation GNN $g_\phi$ | message-passing GNN predicting $(\log_{10} c_i, \theta_i)$; **ranks** candidates so the analytical model is called less |
| `ga_gnn_policy` | `+` proposal GNN $\pi_\psi$ | policy GNN mapping a scenario to a distribution over gene levels; **seeds** the initial population |

---

## The three result figures

The manuscript is six pages and carries three figures plus one table. The
anytime experiment that used to be a fourth figure now lives in the lower half
of that table: the same numbers, a fraction of the space.

### `fig11_anytime.png` — objective against compute *(not in the paper; its numbers are Table I, lower half)*

(a) mean over twenty seeds of what each method had reached by a given wall-clock
budget, shaded by one standard deviation; (b) the fraction of seeds above the
success threshold at the same budget.

This is the figure that closes the ablation. At a common $(N_{pop}, N_{gen})$
the evaluation GNN is **not** expected to raise the objective — it swaps an
exact evaluator for an approximate one, so at an equal number of candidates
examined it can at best match the baseline, and it carries prediction error
besides. What it buys is that each candidate costs less: the same grid point
costs the three methods 12,426, 1,070 and 474 exact calls. Whether the optimiser
can *use* that cheapness is a separate question, and only a budget axis answers
it. The ordering holds at all 40 shared wall-clock budgets and all 48 shared
exact-call budgets, with no exception.

The advantage of the evaluation GNN alone is bounded, and the paper says so: its
curve settles at 43.01 because the surrogate's error caps what screening by it
can select, while the baseline keeps climbing and would cross that ceiling given
time beyond the measured range. Only the proposal GNN lifts the ceiling.

No early stopping is used here, so each curve runs to the end of its budget.
That is why the baseline's totals (255 s, 52,056 calls) exceed the ones in
Table I, where the stagnation criterion is active — the two are not
interchangeable.

- data: `data/pp1_anytime.json`
- produced by: `wifi7_edca/training/pp1_anytime.py --n-seed 20`, which needs the
  per-generation timestamps added to `GAHistory` (`wall`, `evals`) in
  `training/ga.py` and `training/ga_surrogate.py`

### `fig08_sweep_methods.png` — behaviour when the tightest threshold moves

(a) objective $F$, (b) $\sum_i \theta_i$, both against $\varepsilon_1$ over five
decades, all three methods, **mean of 10 restarts** per point at the largest
search configuration run, $N_{pop} = N_{gen} = 120$. That is not an equal cost:
it buys the baseline 12,426 exact calls against the pipeline's 474, so the
baseline has 26x the budget and still stays below at every threshold.
The mean, not the best: the best says what a method can reach, the mean says
what one run of it returns.

The monotonicity bound applies to $F^\star$: $F = \sum_i -\log_{10} P_{loss,i}$
does not contain $\varepsilon$, which enters only through the constraints, so
relaxing $\varepsilon_1$ only enlarges the feasible set and the optimum
**cannot fall**. That bound is on the optimum, which the best-of-N estimates,
not on the mean plotted here — a mean is a property of the solver and may dip.
It does not here, for any of the three.

Panel (b) carries **no such bound**. $\sum_i \theta_i$ is not the objective, and
the configuration maximising $F$ is free to spend reliability margin where the
constraints do not ask for it, so a dip there is a property of the solution.
Do not read the two panels under the same rule.

- data: `data/pp1_budget_methods.json`
- produced by: `wifi7_edca/training/pp1_eps_baseline.py` (GA),
  `wifi7_edca/training/pp1_eps_methods.py` (both GNN branches),
  merged by `wifi7_edca/training/pp1_budget_collect.py`

### `fig09_qos_methods.png` — QoS of the returned configuration

(a) delay-violation probability against the per-category target $\varepsilon_i$,
(b) packet-loss probability, for the default IEEE 802.11e table and all three
methods.

The configuration plotted for each method is its **best of ten** independent
restarts at a common budget. A single run would mostly plot seed noise: on this
scenario the spread between restarts (GA best $49.70$ against median $43.11$) is
wider than the spread between methods.

Panel (b) is worth showing only because $g_\phi$ alone is present: with just the
baseline and the full pipeline every optimised configuration sits on the
$10^{-10}$ reporting floor and the panel separates nothing.

- data: `data/pp1_qos_methods.json`
- produced by: `wifi7_edca/training/pp1_qos_methods.py`

### `fig10_gnn_training.png` — training convergence of both networks

(a) $\pi_\psi$ against cross-entropy step, (b) $g_\phi$ against training epoch.
The x axes cannot be shared — the two are trained on different problems — but
the **y axis is**, and that is the point: each network is scored by the
objective $F$ its output actually achieves under the analytical model.

$\pi_\psi$ emits a configuration, so it is scored by the exact $F$ of that
configuration. $g_\phi$ emits an **ordering**, so it is scored the way the
pipeline uses it: the mean exact $F$ of the $V = 8$ candidates it forwards for
exact evaluation, out of a fixed pool of $1{,}560$ configurations scored once in
advance. A regression error (RMSE) does not belong on this axis; the ranking it
induces does.

Two earlier versions of this measurement were discarded, both because they
measured the wrong thing:

1. **Running a small GA with the surrogate at each checkpoint.** At a budget
   small enough to run every epoch the number is bottlenecked by the GA's search,
   not by the network: it sat at $14$–$19$ and did not rise while RMSE fell
   steadily.
2. **Top-1 pick from a pool with 12% near-optimal candidates.** An *untrained*
   network lands on a good one by chance, so the curve started at $47.76$ and
   only flickered. Fixed by cutting the good fraction to $2.2\%$ and averaging
   over the top $V = 8$ rather than the top 1.

- data: `data/gnn_train_curve.json` (evaluation GNN),
  `data/pp1_paper_figs.json` key `e1` (proposal GNN)
- produced by: `wifi7_edca/training/train_gnn.py --curve-out ...`
  (writes `results/gnn_model_curve.pt`, a **separate** checkpoint — the deployed
  `gnn_model.pt` is never overwritten), and `wifi7_edca/training/pp1_paper_figs.py`

---

## The other figures in `Image/`

Still generated by `make_figures.py`. `main.tex` includes `fig08`, `fig09` and
`fig10` only; the rest are kept for the record and for talks.

| file | shows |
|---|---|
| `fig02_convergence.png` | proposal network alone, against the GA band — superseded by `fig10` panel (a) |
| `fig03_qos.png` | violation probability + success rate — panel (a) superseded by `fig09` |
| `fig04_epsilon_sweep.png` | threshold sweep, GA and pipeline only — superseded by `fig08` |
| `fig05_cost.png` | wall-clock and analytical-model calls per solve, four configurations — now carried by Table I |
| `fig06_budget.png` | quality and cost against $N_{pop}$ on the main scenario, three methods |
| `fig07_methods_budget.png` | cost against quality across all five thresholds, three methods, best vs median over ten restarts |

## Budget sweep, for reference

Six budgets, ten restarts per (method, budget, threshold), mean $F$ over the five
thresholds. The pipeline reaches at $181$ analytical-model calls what the
baseline needs $12{,}426$ to reach.

| budget | GA | | `+`$g_\phi$ | | `+`$\pi_\psi$ | |
|---|---|---|---|---|---|---|
| | calls | best | calls | best | calls | best |
| 16×9 | 526 | 29.96 | 308 | 33.89 | 181 | **49.43** |
| 24×24 | 795 | 31.76 | 403 | 33.09 | 222 | **49.43** |
| 40×40 | 1 622 | 39.23 | 533 | 39.45 | 270 | **49.51** |
| 60×60 | 3 362 | 44.82 | 684 | 44.06 | 320 | **49.49** |
| 80×80 | 6 060 | 46.68 | 805 | 46.31 | 400 | **49.49** |
| 120×120 | 12 426 | 49.49 | 1 070 | 47.67 | 474 | **49.49** |

Every one of the eighteen (method, budget) curves is non-decreasing in
$\varepsilon_1$, as panel (a) of `fig08` requires.

## The table

`main.tex` carries one results table, built from two blocks that
`wifi7_edca/training/pp1_tables.py` writes out as LaTeX rather than typed by
hand — these numbers have moved several times, and a hand-copied table is one
more chance for the paper and the data to disagree.

- **Top block**, from `data/benchmark.json` and `data/pipeline_ngen1.json`: one
  search configuration, varying only which networks are present. This is where
  the evaluation GNN looks like it does nothing to the objective ($34.63$
  against $35.04$) while cutting exact calls $19.8\times$ — which is the
  correct expectation, not a disappointment.
- **Bottom block**, from `data/pp1_anytime.json`: mean objective at a shared
  wall-clock budget. This is where the cheaper evaluation shows up as a better
  solution, and the ordering is strict at all 40 wall-clock and all 48
  exact-call budgets measured.

Regenerate with `py -3 -m training.pp1_tables`, which writes
`results/pp1_table_body.tex`.

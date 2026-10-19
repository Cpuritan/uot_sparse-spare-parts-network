# Experiment 1: Fixed-Instance Solution Quality and Multi-Instance MC Trajectory Diagnostics

This directory is a self-contained paper experiment. It does not import modules or solver results from the legacy project. It was reorganized on 2026-10-18; the implementation follows the joint location-inventory-allocation model and Algorithm 1 in `input/manuscript.tex`. The manuscript snapshot is provided for method checks only; the instance counts and numerical values in the manuscript's older experiment section are not used for this run.

## 1. Scope

| Study | Instances | Repetitions and outputs |
|---|---|---|
| Solution quality and computing time | One fixed instance for each of S/M/L, seed=42 | One analytical run, one continuous-model MILP, and 20 repetitions at each of three MC budgets per instance; `table-exp1-quality.csv/.tex` |
| Inventory, shared weights, warehouse activation, and region evolution | Fixed M, seed=42, analytical run | Eight-panel `figure-exp1-1`; maps at t=1, 3, 5, and recovery R |
| MC trajectory deviation | **20 instances for each of S/M/L**, seeds 42–61 | **5 MC repetitions** at each budget and instance; 20 points per group in `figure-exp1-2` |

The budgets are 1,000 / 10,000 / 100,000. The first five MC repetitions for seed=42 are shared by both studies; the remaining fifteen are used only for fixed-instance quality. The trajectory-diagnostic time window is not expanded. In total, the experiment contains 60 analytical runs, 1,035 MC runs, and 3 MILPs: **1,098 unique planning records**. The fixed-instance quality table contains 186 raw records; the trajectory statistics contain 180 instance-budget observations and 900 per-repetition diagnostics.

This directory does not run Experiment 2, SAA, or sensitivity experiments, and it does not overwrite legacy-project figures. It restores the historical “20 instances × 5 repetitions” trajectory design; it does not relabel 20 repetitions from one fixed instance as different instances, and it does not redraw figures from historical results.

## 2. Flat layout and execution

```text
exp1/
  environment.py   Frozen instances, demand, costs, and random streams
  geometry.py      Continuous regions, Green boundary integration, and complete cost-order subdivision
  algorithm.py     Manuscript reweighted algorithm and MC counterpart with only the integrator replaced
  evaluation.py    Continuous-model primal/dual cost intervals for fixed inventory
  benchmark.py     MILP reference for the same continuous model
  diagnostics.py   Path alignment, D/V/E metrics, and sampling audits
  experiment1.py   Single entry point, checkpoints, and statistics
  plot_exp1.py     Two paper figures and auditable plotting data
  test_exp1.py     Method and experiment-definition checks
  input/           20 frozen master instances, 60 derived instances, and manuscript snapshot
  output/          Generated solutions, diagnostics, tables, figures, logs, and audits
```

Run from this directory:

```powershell
python -m unittest -v test_exp1
python -u experiment1.py
python experiment1.py --report-only
```

The reference environment uses Python `D:/Anaconda3/python.exe`. Exact dependency versions are listed in `requirements.txt`; the MILP requires a valid Gurobi license. `--report-only` reads completed local results and regenerates statistics and figures without planning again.

Each planning record is atomically written to `output/case_<scale>_<seed>.json`. Re-running the same command resumes completed local records; an unfinished planning call is rerun without fabricating time or status. The program rejects results whose core-source or input hashes differ. After changing scientific settings, preserve the old output and perform a separate full run; do not bypass the hash checks.

## 3. Correspondence with the manuscript method

| Manuscript object | Implementation and verification |
|---|---|
| `eq:design_objective`: one shared opening cost | Both inventory parts share one active-warehouse set; the continuous economic cost includes local delivery, external supply, leftover inventory, and one opening cost. |
| `eq:design_weights` | `algorithm.run_design` synchronously computes `F_i/(sum_p m_i^p + epsilon)` at every iteration, with epsilon fixed at 1. |
| `eq:design_partition`, `eq:design_update` | Every candidate warehouse participates, including zero-inventory warehouses; both demand parts are integrated after adjusted-cost partitioning. Ties prioritize external supply 0 and then increasing warehouse ID. |
| Joint inventory-allocation optimization | The design iteration has no additional dual optimization for fixed inventory, integer rounding, or random-scenario testing. |
| `eq:design_stopping` | Frobenius change divided by `max(1, preceding_norm)`, tolerance 1e-6, at most 200 iterations; the actual stopping status is retained. |
| `eq:design_recovery` | Zero-penalty recovery is performed only over the strictly positive terminal inventory set; zero-demand warehouses are removed and the warehouse subset is not optimized again. |
| `eq:design_logsum` | MM non-increase is checked for the fixed log-sum surrogate; the original economic cost is not required to decrease at every iteration. |
| Exact regional integration | `geometry.ExactEvaluator` uses local polynomial kernels, common affine subdivision, and Green boundary integration. Eight-point boundary Gauss quadrature is exact for the corresponding edge polynomials; it is not area sampling. |
| Continuous evaluation for fixed inventory | `evaluation.continuous_cost` independently computes the original primal feasible upper bound and dual lower bound after design; its time is excluded from planning and it does not change inventory. |
| Continuous-model MILP | `benchmark.solve_continuous_milp` uses complete cost-order subdivision and exact demand/cost integration. It does not constrain the reference solution to the analytical algorithm's warehouses or regions. |

`geometry.py` and `algorithm.py` were migrated from the already verified implementation without changing the scientific calculations. The remaining modules contain only what this experiment needs and use local input/output paths. Fourteen method/definition tests cover frozen inputs, independent regional integration, cost formulas, tie rules, shared weights, zero-inventory candidates, stopping/recovery, fixed-inventory evaluation, complete-subset MILP comparison, nested samples, and D/V definitions. A fifteenth test, enabled after full results exist, independently reads every raw trajectory, recomputes all 180 D/V values without calling the diagnostic function, and checks the summary tables. The actual run also audits mass conservation, shared weights, sample hashes, and record counts for every trajectory.

## 4. Inputs and randomness

The common v2-r3 environment is used without selecting favorable instances or tuning parameters: S/M/L have side lengths 30/45/60 km, 16/24/32 warehouses, and 100/250/500 devices. Within each scale, instances use the same nested master pool for a given seed, so scales are paired; master instances from different seeds are independent.

Part 1 probabilities are generated from U(0.15, 0.35), and Part 2 probabilities from U(0.05, 0.25), using independent random streams. The two demand and inventory parts are not copies. The Diamond triweight kernel, 1 km bandwidth, shared fixed cost 120–180, leftover cost 16, and original cost parameters are unchanged. The initial inventory is `m_i^p=Q_p/M`, evenly distributing each part's total demand over all candidate warehouses.

`input/master_42.npz` through `master_61.npz` are frozen inputs copied from the validated environment, not solver results; every generation checks arrays element by element against these files. `input/instance_*.npz` contains the derived instances generated for this run.

MC random streams use `[20261017, instance_seed, 7, repeat, part]`. Repetitions and parts are independent; budgets use nested prefixes of the same stream, and one complete algorithm run does not resample internally. Each part receives its share of the total budget, with sample weights `Q_p/B_p`, using an exact density-mixture sampler. This restores the reproducible historical random-stream design rather than the later fixed-instance diagnostic batch based on operating-system entropy. Scales sharing a seed use paired random numbers and are not interpreted as independent cross-scale replications.

## 5. Statistical unit for trajectory deviation

Let `z_t` be the mass matrix containing external supply (row 0), all warehouses, and both spare-parts columns; let `a_t` be the analytical path and `z_rt` the MC path for repetition r. Each instance has five repetitions. The common window H is the largest stopping time among the analytical path and the five MC paths at all three budgets. It includes t=0; an early-stopping path keeps its final iterate, is not padded with zeros or interpolated, and does not include final recovery R.

```text
D = sqrt(mean_{r=1..5, t=0..H} ||z_rt - a_t||_F^2) / Q
V = sqrt(mean_{r=1..5, t=0..H} ||z_rt - mean_r(z_rt)||_F^2) / Q
E = sqrt(mean_{r=1..5, t=1..T_analytic} ||MC_r(w_t) - Exact(w_t)||_F^2) / Q
```

E replays integration at the analytical reference weights and separates one-step integration error from feedback-evolution deviation. D is the per-repetition **root mean square** deviation, not the arithmetic mean of five deviations; the identity `D² = V² + mean_path_deviation²` is checked. The denominator is always the instance's total demand; neither spare-parts columns nor the external-supply row may be omitted.

Each point in `figure-exp1-2` is one instance's D. The nine scale-budget groups contain 20 points each. Boxes show the across-instance quartiles and median, with whiskers and outliers retained. The paper figure displays D only; hollow-diamond V markers and their legend are omitted, while auxiliary diagnostics remain available for audit. Only the horizontal position receives jitter; the vertical values are neither clipped nor altered, and results are not resampled. D is not required to be monotone in budget and is not automatically interpreted as an economic-cost gap.

The MC standard deviation in the fixed-instance quality table describes algorithmic randomness for a given instance; it is not a standard error or confidence interval for a random-network population. The multi-instance trajectory statistics and fixed-instance quality statistics must not be pooled into one conclusion.

## 6. Solver status, outputs, and audits

MILPs use one thread, a 1,800-second time limit, and a 0.001 relative gap target, with an all-external feasible starting point. At the time limit, a feasible incumbent, valid bound, actual gap, and `time_limit` status are retained; the result is not called exactly optimal. A missing incumbent is reported as missing. Gurobi's `optimal` status means that its gap and numerical tolerances were satisfied, not that the mathematical error is exactly zero.

The quality table reports continuous-cost primal/dual intervals and optimality-gap intervals relative to the reference solution; it does not treat an MILP incumbent as a proven exact optimum. `quality_runs.csv` retains each actual solver gap and status. Floating-point safety margins are not strict interval-arithmetic proofs. If continuous certification is not reached, the record retains `interval_only` and the actual interval rather than claiming certification.

- `case_*.json`: complete planning trajectory, recovered inventory, status, and measured planning time; fixed instances also include continuous-cost evaluation.
- `diagnostic_*.json/.npz`: quality matrices, aligned paths, frozen-reference replay, E/D/V metrics, per-step deviations, and sample hashes.
- `quality_runs.csv` / `table-exp1-quality.csv/.tex`: raw and summarized fixed-instance quality and timing results.
- `trajectory_instances.csv` / `trajectory_repeats.csv`: 180 instance observations and 900 repetition diagnostics.
- `figure-exp1-1.png/.pdf/.svg`, `figure-exp1-2.png/.pdf/.svg`: 300 dpi raster figures and editable vector figures.
- `figure-exp1-1-data.json`, `figure-exp1-2-summary.csv`, `figure-captions.json`: plotting data and complete captions.
- `environment.json` / `progress.json` / `run.log` / `validation.json`: environment, progress, logs, counts/statuses/file hashes, and SVG checks for all 180 plotted points.

Numba compilation and solver initialization are warmed up separately. Timing includes method preprocessing, iterations, and recovery, but excludes continuous evaluation, file saving, and plotting. All formal planning runs are serial to avoid contention in method-time comparisons.

## 7. Completed run record (2026-10-18)

All 60 instances and 1,098 unique planning records were completed: 1,095 analytical/MC runs converged normally, and all 3 MILPs returned `optimal` with actual solver gap 0; there were no timeouts or missing incumbents. All 186 fixed-instance continuous evaluations met the `certified` standard. All 15 tests passed, including independent recomputation of all 180 D/V values from raw trajectories. PNG/PDF/SVG versions of both figures were generated and checked; SVG validation confirmed 20 points in each of the nine scatter groups.

The instance-level median D values in the new `figure-exp1-2` are (%):

| Scale | MC 1,000 | MC 10,000 | MC 100,000 |
|---|---:|---:|---:|
| S | 6.483 | 1.217 | 0.369 |
| M | 10.316 | 1.837 | 0.542 |
| L | 6.162 | 3.200 | 0.590 |

These are medians over 20 instances and do not replace the fixed-instance repetition statistics. All high-deviation observations are retained; outliers in the multi-instance version are not removed for the recovery study. Raw points can be traced from `trajectory_instances.csv` using its scale/seed/budget fields.

## 8. Historical COPT migration record (2026-10-18; superseded by Section 9)

For the revision, the three fixed-instance MILPs were solved with COPT 8.0.6 without reusing results or timings from the previous solver. The original results and source files were saved under `archive/pre_copt_20261018/`.

Only the solver interface in `benchmark.py` changed; input, geometry integration, analytical/MC algorithms, and the SciPy continuous evaluator were hash-checked as unchanged. The 1,095 analytical/MC records and their timings were retained, the three MILPs and continuous evaluations were rerun, and all 186 optimality-gap intervals were updated from the new COPT bounds. The 60-instance trajectories and both figure datasets were unchanged. Each run file retains its source-signature and migration pointers.

| Scale | COPT complete planning time (s) | Original objective cost (EUR) | Actual relative MIP gap |
|---|---:|---:|---:|
| S | 2.076765 | 2204.86390319 | 0 |
| M | 25.376504 | 5172.94845710 | 0 |
| L | 330.767287 | 10111.78059921 | 0 |

The COPT environment and results were archived under `archive/pre_gurobi_20261018/`; historical raw solutions and audits were also retained in `output/copt_milp_*.json` and `output/solver_migration_copt.json`. The current solver and formal results are defined by Section 9.

## 9. Current Gurobi rerun record (2026-10-18)

Under the latest revision, Experiment 1 uses Gurobi 13.0.1. The three fixed-instance MILPs were solved again without reusing historical Gurobi or COPT solutions or timings. The COPT results and pre-Gurobi source were saved under `archive/pre_gurobi_20261018/`.

Only the solver interface changed; the same one-thread setting, 1,800-second limit, and 0.001 relative-gap target were retained. Input, geometry integration, analytical/MC algorithms, continuous evaluator, and MILP coefficients were checked as unchanged. The 1,095 analytical/MC records were retained, the three MILPs and continuous evaluations were rerun, and all 186 optimality-gap intervals were recomputed from the new Gurobi bounds. The two figure datasets were unchanged.

| Scale | Gurobi complete planning time (s) | Original objective cost (EUR) | Actual relative MIP gap |
|---|---:|---:|---:|
| S | 1.617595 | 2204.86390319 | 0 |
| M | 9.695470 | 5172.94845710 | 0 |
| L | 39.426005 | 10111.78059921 | 0 |

The current environment, raw solutions, solver logs, and migration audit are referenced by `output/environment.json`, `output/gurobi_milp_*.json`, `output/gurobi_rerun.log`, and `output/solver_migration_gurobi.json`. The current `case_*.json`, quality tables, and validation files use the Gurobi results.

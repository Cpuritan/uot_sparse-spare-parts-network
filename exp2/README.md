# Experiment 2: Out-of-Sample Cost and Computing Time for Fixed Network Policies

## Current paper presentation: unified SAA

In the paper, SAA refers to SAA-C in the raw records: binary warehouse opening with continuous planning inventory. The two main figures currently show SD-TV-UOT, MVP-C, MVP-I, and SAA; the cost, time, and gap for SAA are read only from the original SAA-C record. The reported paper speedup is 11.5–50.7×. The original five-method records, complete statistics, and scenario files remain available for verification; the five-method execution description below documents the raw experiment.

To regenerate figures only, run `python plot_exp2.py`; this does not call planning or out-of-sample optimization. `figure-exp2-1-paper-data.csv` contains the 12,000 observations used by the main cost figure, and `figure-exp2-2-paper-data.csv` contains the 12-row complete statistics table; both use the paper label SAA. The original `table-exp2-summary.csv` and the two `figure-exp2-*-data.csv` files remain the complete raw sources. PNG, PDF, and SVG files are overwritten using the original figure names; the manuscript cites the corresponding files in its `Figures/` directory as Figures 6 and 7, respectively.

This directory is a self-contained, lightweight paper experiment. It does not import `exp1` or legacy-project modules, and it does not read old policies, solver times, or scenario costs. The settings from the 2026-10-18 revision take precedence over the older Experiment 2: each SAA method is trained once on 200 joint scenarios; the continuous-kernel bandwidth is 5 km; MVP/SAA use Gurobi; and costs are reported in euros. The implementation follows the method and Algorithm 1 in `input/manuscript.tex`; older experiment paragraphs in the manuscript are not used as the design specification for this run.

## 1. Flat layout and execution

```text
exp2/
  environment.py   Fixed instances, costs, and independent train/test streams
  geometry.py      Kernel boundary clipping, common affine subdivision, and exact Green integration
  algorithm.py     Shared reweighting and final recovery from Algorithm 1
  benchmark.py     Gurobi models for MVP-C/I and SAA-C/I
  evaluation.py    Unified integer execution and scenario-level cost/risk metrics
  experiment2.py   Single entry point, per-policy checkpoints, and summary statistics
  plot_exp2.py     Two paper figures that read only saved results
  test_exp2.py     Independent integration, manuscript, Gurobi, and transport-LP checks
  validate_exp2.py Independent post-run audit of complete results
  requirements.txt Exact dependency versions
  input/           Frozen seed-42 inputs, train/test scenarios, and manuscript snapshot
  output/          Generated solutions, scenario costs, tables, figures, logs, and audits
```

Run from this directory:

```powershell
python -m unittest -v test_exp2
python -u experiment2.py
python validate_exp2.py
python experiment2.py --report-only
```

The reference environment uses `D:/Anaconda3/python.exe`. Gurobi requires a valid license. The normal entry point runs serially; after interruption it reuses completed policies whose source and input signatures match, without resolving them. Each planning call immediately saves its incumbent, bound, actual gap, and timing before out-of-sample evaluation. If an unfinished solve cannot export an incumbent, no result is fabricated. If scientific settings or core source code change, preserve the current output and start a separate batch; the program rejects mixed signatures. `--report-only` does not solve or reevaluate scenarios.

## 2. Experimental settings

| Item | Setting |
|---|---|
| Instances | One fixed seed=42 instance for each of S/M/L |
| Region side length | 30 / 45 / 60 km |
| Devices | 100 / 250 / 500 |
| Candidate warehouses | 16 / 24 / 32 |
| Spatial input | Frozen v2-r3 master instances; six hotspots plus background; locations, probabilities, and costs unchanged |
| Part 1 / Part 2 | Independent probability columns U(0.15, 0.35) / U(0.05, 0.25); demand and inventory are not copied |
| SD-TV-UOT | Exact integration, manuscript diamond cubic-product kernel, bandwidth 5 km, epsilon=1 |
| Stopping rule | Frobenius relative inventory change at most 1e-6; at most 200 iterations |
| MVP-C/I | One solve per method and instance using expected device-point demand |
| SAA-C/I | One solve per method and instance using 200 joint training scenarios, repeat=0 |
| OOS | Common 1,000 joint Bernoulli test scenarios per instance |
| Solver | Gurobi 13.0.1, one thread, 1,800-second limit, target relative gap 0.001 for the full cost |
| Scope | Five policies per instance, 15 policies total; no other seeds or sensitivity experiments |
| Currency | Euro; figures use € and the original numerical cost scale without conversion |

The objects at a given scale and seed come from the nested master instance, so scales are not independent random networks. Training and test streams use streams 8 and 9, with entropy `[20261017,42,stream,0,part]`. SAA-C and SAA-I at the same scale share one training set; all five policies share one independent test set. A joint scenario contains all devices and both parts; the scenario count is not split between parts. Test locations are fixed and are not resampled from the KDE.

Costs are as follows: shared opening cost is an integer in €120–€180 per warehouse and period; local cost is `b_i + beta_i*d + gamma_i*max(d-10,0)`, where `[40, 0.9, 0.9]` is multiplied by independent U(0.9, 1.1) coefficients; external supply cost is `68 + 0.9*d0 + 0.9*max(d0-10,0)`, where `d0` is Manhattan distance to the region center; leftover cost is €16 per unit and period. The shared opening cost is counted once, the two inventory parts cannot substitute for each other, and there is no additional safety stock or shared-capacity constraint.

## 3. Boundary treatment for the 5 km bandwidth

The original device points remain fixed, so part of a 5 km kernel can cross the original square boundary. The continuous-demand section of the manuscript specifies retaining only kernel mass inside Omega when the full support is not contained in Omega. This experiment follows that rule exactly:

- Intersect each original normalized kernel support with `[0,L]²` exactly; do not expand the region, move points, resample, or renormalize at the boundary.
- Apply Green integration on the clipped polygons while retaining the original probability weights and kernel coefficients.
- Initialize Algorithm 1 with `mu_p(Omega)/M`, rather than the full point demand `sum_n p_n^p/M` after clipping.
- Check at every iteration that warehouse-region mass plus external mass equals `mu_p(Omega)`; do not force it to equal complete device-point demand.
- The discrete MVP/SAA models and OOS evaluation still cover all device points. `table-exp2-boundary-mass.csv` reports both demand totals and lost mass explicitly, so the boundary effect is not hidden as a method advantage or disadvantage.

Thus, changing the bandwidth can change both the smoothing range and the continuous demand mass inside the finite region. This follows from the original finite-domain density specification. Boundary mass must be considered when interpreting OOS differences. A model that strictly preserves total mass inside a finite region would be a separate boundary-correction specification and must not be introduced silently.

## 4. Manuscript correspondence and unified execution

| Manuscript object | Implementation |
|---|---|
| `eq:demand-kernel` | Normalized cubic-product diamond kernel; clipping follows the continuous-demand section. |
| `eq:design_weights` | Each iteration computes the shared weight `F_i/(sum_p m_i^p+epsilon)` from the previous complete inventory matrix. |
| `eq:design_partition`, `eq:design_update` | All candidate warehouses participate; adjusted costs are compared with external supply, with external supply first and warehouse ID as the tie-breaker; both parts are integrated simultaneously. |
| `eq:design_stopping` | Full inventory-matrix Frobenius change divided by `max(1, preceding_norm)`. |
| `eq:design_recovery` | Recovery is performed only over the terminal positive-inventory set, with the penalty removed; no second warehouse-subset search is performed. |
| `eq:design_objective` | The final original economic cost counts shared opening cost once; the log-sum surrogate is not used as the economic objective. |

`geometry.py` was migrated from the verified Experiment 1 mathematical kernel, retaining affine subdivision and Green integration while adding the manuscript-permitted region clipping and removing unused complete-ordering MILP/subset-enumeration helpers. The eight-point boundary Gauss rule is exact for the relevant edge polynomials, not two-dimensional area sampling, subject to floating-point error. `algorithm.py` retains the update formula, tie rules, and recovery logic without methodological changes. Tests independently check clipping mass, synchronous updates, MM non-increase, zero-inventory candidates, stopping, and recovery.

MVP-C/I and SAA-C/I share opening decisions and hold inventory by part; SAA uses shared inventory across training scenarios, with scenario-dependent allocation. MVP-I keeps continuous expected-demand allocation while inventory is integer. The matrix retains the external-cost constant, so the solver gap refers to full cost rather than an arbitrary savings shift. Only strictly economically dominated allocation arcs are removed.

For OOS execution, valid continuous inventory for SD-TV-UOT/MVP-C/SAA-C is rounded up componentwise; integer models are normalized using Gurobi's default integrality tolerance. Warehouses and inventory are fixed, while each scenario reoptimizes integer supply without nominal service-region restrictions; demand is met locally or externally and unused inventory is charged. The assignment algorithm is equivalent to the integer transportation problem and is independently checked with an explicit transport LP.

If a solve times out with an incumbent, the code saves and evaluates it while retaining actual gap, objective, valid bound, and `time_limit` status. If no incumbent exists, the policy is marked missing and figures show Unavailable; it is not filled with zero, an old solution, or a selectively rerun result. Solver `optimal` means termination within the configured tolerances, not an assertion of exact mathematical global optimality.

## 5. Statistics and figures

- `figure-exp2-1.{png,pdf,svg}`: three S/M/L panels with boxplots of 1,000 OOS total-cost observations for five policies. Diamonds are means, black lines are medians, and whiskers/outliers remain part of the data and statistics.
- `figure-exp2-2.{png,pdf,svg}`: complete planning time by scale, with y-axis `Solving time (s)` on a log scale, x-axis labels S/M/L, and a single-row legend. Each point is one measured run; no artificial repeat error bars are shown.
- Neither figure displays seed, training size, bandwidth, solver, time limit, or gap target; these details and actual statuses remain in the tables, raw records, and this README.
- Planning time includes SD preprocessing, iterations, recovery, and final region construction, or benchmark scenario generation, matrix construction, Gurobi model building, solving, and result assembly. It excludes shared input I/O, license initialization, Numba JIT, OOS evaluation, validation, and plotting. Policies run serially. Figure 2 therefore compares complete planning time, not Gurobi-only solver time against SD time.
- `table-exp2-summary.csv/.tex`: 15 policies with mean cost, CVaR95, external share, cost components, inventory, rounding increment, planning/solver/evaluation time, actual gap, bound, and status. SD relative inventory change is stored separately and is not filled as an MIP gap.
- CVaR95 is the mean of the policy's 50 highest-cost scenarios; external share is total external units divided by total demand. Policy tails are not pooled and cross-network confidence intervals are not estimated.
- `figure-exp2-1-data.csv` contains complete per-scenario cost components; `figure-exp2-2-data.csv` contains one timing/status row per policy; `table-exp2-inventory.csv` contains raw and executed inventory by warehouse.

Figures use editable SVG, embedded-font PDF, and 300 dpi PNG output. Legacy-project results, Experiment 1, and the manuscript are not overwritten; the local manuscript snapshot is used only for method checks.

## 6. Gurobi migration and result preservation

This run re-solved the 12 MVP/SAA policies that previously used COPT and recomputed their 12,000 OOS scenario costs. The three solver-free SD-TV-UOT policies reused their original inventory, trajectories, measured times, and 3,000 scenario costs. The 200 training scenarios, 1,000 common test scenarios, 5 km bandwidth, costs, and random streams were unchanged.

The pre-switch archive containing 72 source, input, raw-result, and figure files is stored at `input/before-gurobi-20261018.zip`. The program does not accept old COPT benchmark records as Gurobi records. The mathematical model construction was verified as textually unchanged apart from solver-interface parameter names. Shared opening, inventory types, valid bounds, external-cost constant, and feasibility checks are unchanged; Gurobi uses its default feasibility/integrality tolerances, solver seed=0, one thread, 1,800 seconds, and target gap 0.001.

`input/source-compatibility.json` permits exact source/record-hash reuse only for the three solver-free SD-TV-UOT records. It ignores only historical version and newly added solver fields, not scientific parameters. The old complete COPT results and interface proof remain archived and are not mixed into the current tables. Current tables include `solver` and `solver_version` fields.

Figure 2's `Solving time` is the end-to-end planning time read from `planning_seconds`, not only the optimizer call in `solver_seconds`. The latter remains a separate table field; SD-TV-UOT has no MIP optimizer call, so its complete planning time cannot be compared to benchmark-only optimizer time. This section changes only the axis label, legend layout, and scale labels, not the timing definition.

The current state and actual gaps are defined by `output/completion.json`, `output/table-exp2-summary.csv`, and `output/validation.json`. Historical COPT gaps, timings, and tests are available only as archived comparisons.

### Completed results for this run

All 12 benchmark policies were re-solved by Gurobi within 1,800 seconds and reached the 0.1% relative-gap target; there were no timeouts or missing incumbents. The maximum actual gap was 0.0009877656826097255 (approximately 0.098777%, S/MVP-I). Incumbent, objective, bound, and actual gap were retained without rewriting a nonzero gap as zero. The 12 benchmark complete-planning times totaled 528.808 seconds. Figure 2 uses these complete planning times (seconds):

| Scale | SD-TV-UOT (reused) | MVP-C | MVP-I | SAA-C | SAA-I |
|---|---:|---:|---:|---:|---:|
| S | 0.572 | 0.095 | 0.918 | 6.584 | 6.623 |
| M | 1.723 | 0.285 | 1.160 | 72.189 | 125.052 |
| L | 3.076 | 0.847 | 1.959 | 156.042 | 157.054 |

All 15 tests passed. The complete audit covered 15 policies, 15,000 scenario costs, 850 independent kernel-boundary integrals, and 105 independent transport LPs; the maximum absolute transport-cost error was 1.82e-12 euros. After running report-only mode, hashes of 38 policy, scenario, CSV, and PNG files remained unchanged.

`output/gurobi-switch-audit.json` records the migration audit: the three SD policies and their scenario files are byte-for-byte unchanged; frozen inputs and train/test hashes are unchanged; and all 12 benchmark policies have new Gurobi records. `output/preservation-audit.json` is the earlier preservation audit for the legacy project and Experiment 1 and is not modified by this repository. Under the fixed inputs, many policies have the same executed inventory and OOS costs after the new solve; those values are the new solve's results, not borrowed COPT records.

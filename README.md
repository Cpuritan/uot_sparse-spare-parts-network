# UOT Sparse Spare-Parts Network

Reproducible numerical experiments for the paper on spatially distributed spare-parts inventory design with unbalanced optimal transport.

The repository intentionally contains source code, frozen input data, tests, and method snapshots only. Generated policies, solver logs, diagnostics, figures, tables, caches, and other run-time files are excluded from version control.

## Contents

- `exp1/`: fixed-instance quality checks and Monte Carlo trajectory diagnostics.
- `exp2/`: fixed seed-42 out-of-sample comparison of SD-TV-UOT, MVP, and SAA.

Each experiment is self-contained and uses a flat numerical-experiment layout. Its local README documents the model, random streams, solver settings, and output definitions.

## Environment

- Python 3.11 recommended.
- Install dependencies separately for each experiment:

  ```powershell
  python -m pip install -r exp1/requirements.txt
  python -m pip install -r exp2/requirements.txt
  ```

- The MILP benchmarks require a valid Gurobi license. The analytical and Monte Carlo components can be inspected and tested without solving the benchmark MILPs.

## Reproduction

Run the unit checks first:

```powershell
cd exp1
python -m unittest -v test_exp1

cd ..\exp2
python -m unittest -v test_exp2
```

To run an experiment, execute its script from that experiment's directory so the local input and output paths resolve correctly:

```powershell
cd exp1
python experiment1.py

cd ..\exp2
python experiment2.py
```

Both experiments checkpoint completed policies under a generated `output/` directory and can be resumed after interruption. Use `--report-only` after a completed run to regenerate reports and figures without solving again. The generated `output/` directories are deliberately ignored by Git.

The committed `.npz` files are frozen inputs used to make the published numerical results byte-reproducible. The `environment.py` modules also document and validate how these inputs, demand probabilities, costs, and random streams are constructed.

## Solver and numerical scope

The current benchmark interface uses Gurobi 13.0.1, one thread, a 1,800-second time limit, and a 0.001 relative MIP-gap target. If a solve reaches the time limit with an incumbent, the code preserves the feasible solution, bound, and actual gap. It does not replace missing or incomplete results with generated placeholders.

## License and citation

This repository accompanies the associated manuscript. Please cite the paper when using the code or frozen inputs. A formal open-source license can be added once the manuscript's data and code-distribution policy is finalized.

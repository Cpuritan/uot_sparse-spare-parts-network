"""Run Experiment 1 locally: fixed-instance quality and 20-instance MC diagnostics."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import platform
from time import perf_counter
import numpy as np
import pandas as pd
from algorithm import run_design, warmup
from benchmark import initialize_solver, solve_continuous_milp
from diagnostics import diagnose
from environment import ROOT, DATA, OUT, SETTINGS, instance, digest, write_json
from evaluation import continuous_cost

CORE_FILES = ["environment.py", "geometry.py", "algorithm.py", "evaluation.py", "benchmark.py"]


def source_signature():
    return hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in CORE_FILES)).hexdigest()


def case_path(scale, seed):
    return OUT / f"case_{scale}_{seed}.json"


def run_case(scale, seed):
    inst = instance(scale, seed)
    signature = source_signature()
    path = case_path(scale, seed)
    bundle = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "scale": scale, "seed": seed, "core_signature": signature, "records": {},
        "probability_sha256": digest(inst.probabilities), "created_local": datetime.now().isoformat()}
    if bundle["core_signature"] != signature or bundle["probability_sha256"] != digest(inst.probabilities):
        raise RuntimeError("Existing local run has different code or inputs; preserve it and use a separate output directory")
    input_path = DATA / f"instance_{scale}_{seed}.npz"
    frozen = {name: getattr(inst, name) for name in ["xy", "sites", "probabilities", "true_probabilities",
              "fixed_costs", "base", "beta", "gamma", "equipment_ids", "warehouse_ids", "part_q"]}
    if input_path.exists():
        with np.load(input_path) as saved:
            for name, values in frozen.items():
                np.testing.assert_array_equal(saved[name], values)
    else:
        np.savez_compressed(input_path, **frozen)
    count = SETTINGS["quality_repeats"] if seed == SETTINGS["fixed_seed"] else SETTINGS["trajectory_repeats"]
    jobs = [("SD-TV-UOT", 0, 0)] + [(f"MC-{budget}", budget, repeat)
            for budget in SETTINGS["mc_sizes"] for repeat in range(count)]
    if seed == SETTINGS["fixed_seed"]:
        jobs.append(("MILP", 0, 0))
    new_runs = 0
    for method, budget, repeat in jobs:
        run_id = f"{scale}_{seed}_{method}_{repeat}"
        if run_id in bundle["records"]:
            continue
        created = datetime.now().isoformat()
        if method == "MILP":
            result, coefficients = solve_continuous_milp(inst)
            np.savez_compressed(OUT / f"milp_coefficients_{scale}_{seed}.npz", **coefficients)
            evaluator = None
        else:
            result, evaluator = run_design(inst, epsilon=SETTINGS["epsilon"], mc_size=budget, repeat=repeat)
        if result["inventory"] is not None and seed == SETTINGS["fixed_seed"]:
            result["continuous_evaluation"] = continuous_cost(inst, result["inventory"], evaluator if not budget else None)
        record = {"run_id": run_id, "scale": scale, "seed": seed, "method": method,
                  "budget": budget, "repeat": repeat, "created_local": created,
                  "completed_local": datetime.now().isoformat(), "record_state": "completed", **result}
        if signature != source_signature():
            raise RuntimeError("Core changed during execution")
        bundle["records"][run_id] = record
        write_json(path, bundle)
        new_runs += 1
        print(f"{run_id}: {record['status']}, planning={record['timing']['planning']:.3f}s, gap={record.get('gap')}", flush=True)
    assert len(bundle["records"]) == len(jobs)
    diagnostic, arrays = diagnose(inst, bundle["records"])
    write_json(OUT / f"diagnostic_{scale}_{seed}.json", diagnostic)
    np.savez_compressed(OUT / f"diagnostic_{scale}_{seed}.npz", **arrays)
    bundle.update(completed=True, completed_local=datetime.now().isoformat())
    write_json(path, bundle)
    return new_runs


def gap_bounds(lower, upper, reference):
    reference_lower, reference_upper = reference.get("lower_bound"), reference.get("planning_objective")
    if reference_lower is None or reference_upper is None or not np.isfinite([lower, upper]).all():
        return np.nan, np.nan
    allowance = 1e-7 * max(1., abs(reference_lower), abs(reference_upper))
    if reference_lower > min(reference_upper, upper) + allowance:
        raise ArithmeticError("MILP lower bound exceeds feasible continuous cost")
    reference_lower -= allowance
    if reference_lower <= 0:
        return np.nan, np.nan
    return max(0., 100 * (lower / min(reference_upper, upper) - 1)), max(0., 100 * (upper / reference_lower - 1))


def report():
    instance_rows, repeat_rows, quality_rows, all_records = [], [], [], []
    fixed_records = {}
    for scale in "SML":
        for seed in SETTINGS["seeds"]:
            bundle = json.loads(case_path(scale, seed).read_text(encoding="utf-8"))
            if not bundle["completed"] or bundle["core_signature"] != source_signature():
                raise RuntimeError("Incomplete or stale local experiment")
            diagnostic = json.loads((OUT / f"diagnostic_{scale}_{seed}.json").read_text(encoding="utf-8"))
            instance_rows.extend(diagnostic["instance_rows"])
            repeat_rows.extend(diagnostic["repeat_rows"])
            records = list(bundle["records"].values())
            all_records.extend(records)
            if seed != SETTINGS["fixed_seed"]:
                continue
            fixed_records[scale] = records
            reference = next(record for record in records if record["method"] == "MILP")
            for record in records:
                evaluation = record.get("continuous_evaluation", {})
                lower, upper = evaluation.get("lower", np.nan), evaluation.get("upper", np.nan)
                gap_lower, gap_upper = gap_bounds(lower, upper, reference)
                quality_rows.append({"scale": scale, "seed": seed, "method": record["method"], "repeat": record["repeat"],
                    "run_id": record["run_id"], "continuous_cost": upper, "continuous_lower": lower,
                    "evaluation_status": evaluation.get("status", "no_incumbent"), "evaluation_gap": evaluation.get("relative_gap"),
                    "gap_lower_percent": gap_lower, "gap_upper_percent": gap_upper,
                    "planning_seconds": record["timing"]["planning"], "status": record["status"],
                    "solver_gap": record.get("gap"), "solver_lower_bound": record.get("lower_bound"),
                    "solver_incumbent": record.get("objective"), "iterations": record.get("iterations"),
                    "active_count": len(record["active"]) if record["active"] is not None else np.nan})
    quality = pd.DataFrame(quality_rows)
    trajectories = pd.DataFrame(instance_rows)
    repeats = pd.DataFrame(repeat_rows)
    assert len(quality) == 186 and len(trajectories) == 180 and len(repeats) == 900
    summary = quality.groupby(["scale", "method"], sort=False).agg(
        repeats=("run_id", "size"), mean_cost=("continuous_cost", "mean"), cost_repeat_sd=("continuous_cost", "std"),
        gap_lower_percent=("gap_lower_percent", "mean"), gap_upper_percent=("gap_upper_percent", "mean"),
        planning_median_seconds=("planning_seconds", "median"),
        planning_q1_seconds=("planning_seconds", lambda values: values.quantile(.25)),
        planning_q3_seconds=("planning_seconds", lambda values: values.quantile(.75)),
        actual_solver_gap=("solver_gap", "max")).reset_index()
    for name, frame in [("quality_runs", quality), ("table-exp1-quality", summary),
                        ("trajectory_instances", trajectories), ("trajectory_repeats", repeats)]:
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    (OUT / "table-exp1-quality.tex").write_text(summary.to_latex(index=False, float_format="%.6f", na_rep="--"), encoding="utf-8")
    from plot_exp1 import make_figures
    figure_audit = make_figures(fixed_records, trajectories)
    planning_seconds = sum(record["timing"]["planning"] for record in all_records)
    status_counts = pd.Series([record["status"] for record in all_records]).value_counts().to_dict()
    solver_records = [record for record in all_records if record["method"] == "MILP"]
    audit = {"completed_local": datetime.now().isoformat(), "passed": True,
        "core_signature": source_signature(), "total_planning_runs": len(all_records),
        "quality_rows": len(quality), "trajectory_instances_per_scale": 20, "trajectory_repeats_per_budget": 5,
        "trajectory_instance_budget_rows": len(trajectories), "trajectory_repeat_rows": len(repeats),
        "fixed_quality_seed": SETTINGS["fixed_seed"], "fixed_quality_repeats_per_budget": 20,
        "trajectory_seeds": SETTINGS["seeds"], "representative_trajectory": {"scale": "M", "seed": 42},
        "planning_seconds": planning_seconds, "status_counts": status_counts,
        "MILP": [{key: record.get(key) for key in ["run_id", "solver", "solver_version", "status", "has_incumbent", "objective", "lower_bound", "gap", "target_gap", "time_limit"]}
                 for record in solver_records], "continuous_evaluation_status": quality.evaluation_status.value_counts().to_dict(),
        "figure_audit": figure_audit,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.glob("*.py")},
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in DATA.iterdir() if path.is_file()},
        "export_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in OUT.glob("figure-exp1-*.*")}}
    assert len(all_records) == 1098
    write_json(OUT / "validation.json", audit)
    print(json.dumps({key: audit[key] for key in ["passed", "total_planning_runs", "quality_rows", "trajectory_instance_budget_rows", "status_counts", "MILP"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Read local completed results; no planning")
    arguments = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if arguments.report_only:
        report()
        return
    started = perf_counter()
    write_json(OUT / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "matplotlib", "numba", "gurobipy"]},
        "settings": SETTINGS, "core_signature": source_signature(), "cache_policy": "Only this directory; no historical plan import"})
    warmup_started = perf_counter()
    warmup()
    initialize_solver()
    warmup_seconds = perf_counter() - warmup_started
    cases = [(scale, SETTINGS["fixed_seed"]) for scale in "SML"]
    cases += [(scale, seed) for scale in "SML" for seed in SETTINGS["seeds"] if seed != SETTINGS["fixed_seed"]]
    new_runs = 0
    for number, (scale, seed) in enumerate(cases, 1):
        new_runs += run_case(scale, seed)
        write_json(OUT / "progress.json", {"completed_cases": number, "expected_cases": len(cases),
            "latest_case": [scale, seed], "new_planning_runs_this_invocation": new_runs,
            "warmup_seconds": warmup_seconds, "elapsed_seconds": perf_counter() - started})
        print(f"Completed {number}/{len(cases)} cases: {scale}/{seed}", flush=True)
    report()


if __name__ == "__main__":
    main()

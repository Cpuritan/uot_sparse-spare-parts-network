"""One local experiment: three fixed networks, five policies, common OOS demands."""
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
from benchmark import initialize_solver, solve_discrete
from environment import ROOT, DATA, OUT, SETTINGS, POLICIES, instance, scenarios, digest, write_json
from evaluation import execution_inventory, evaluate_scenarios, summarize_samples

CORE_FILES = ["environment.py", "geometry.py", "algorithm.py", "benchmark.py", "evaluation.py", "experiment2.py"]


def source_signature():
    return hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in CORE_FILES)).hexdigest()


def plan_path(scale, method):
    return OUT / f"plan_{scale}_{method}.json"


def sample_path(scale, method):
    return OUT / f"samples_{scale}_{method}.npz"


def frozen_inputs(inst):
    arrays = {name: getattr(inst, name) for name in ["xy", "sites", "probabilities", "true_probabilities",
              "fixed_costs", "base", "beta", "gamma", "equipment_ids", "warehouse_ids"]}
    arrays.update(train=scenarios(inst, SETTINGS["n_train"], "train", SETTINGS["training_repeat"]),
                  test=scenarios(inst, SETTINGS["n_test"]))
    path = DATA / f"instance_{inst.scale}_{inst.seed}.npz"
    if path.exists():
        with np.load(path) as saved:
            if set(saved.files) != set(arrays):
                raise RuntimeError("Frozen input fields differ")
            for name, values in arrays.items():
                np.testing.assert_array_equal(saved[name], values)
    else:
        np.savez_compressed(path, **arrays)
    hashes = {name: digest(values) for name, values in arrays.items()}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return arrays, signature


def load_record(scale, method, input_signature):
    record = json.loads(plan_path(scale, method).read_text(encoding="utf-8"))
    signature_matches = record["core_signature"] == source_signature()
    settings_match = record["settings"] == SETTINGS
    proof_path = DATA / "source-compatibility.json"
    if not signature_matches and method == "SD-TV-UOT" and proof_path.exists():
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        signature_matches = (proof["current_core_signature"] == source_signature()
            and proof["executed_core_signature"] == record["core_signature"]
            and proof["record_sha256"].get(plan_path(scale, method).name)
            == hashlib.sha256(plan_path(scale, method).read_bytes()).hexdigest())
        settings_match = signature_matches and (
            {key: value for key, value in record["settings"].items() if key not in ["version", "solver"]}
            == {key: value for key, value in SETTINGS.items() if key not in ["version", "solver"]})
    if (not signature_matches or record["input_signature"] != input_signature
            or not settings_match or record["method"] != method or record["scale"] != scale
            or (method != "SD-TV-UOT" and record.get("solver") != "Gurobi")):
        raise RuntimeError("Existing results differ from current code or inputs; preserve them before a new experiment")
    return record


def run_scale(scale):
    inst = instance(scale, SETTINGS["fixed_seed"], bandwidth_sigma=SETTINGS["bandwidth_sigma"])
    arrays, input_signature = frozen_inputs(inst)
    signature = source_signature()
    for method in POLICIES:
        if plan_path(scale, method).exists():
            load_record(scale, method, input_signature)
            continue
        print(f"START {scale}/{method}", flush=True)
        created = datetime.now().isoformat()
        if method == "SD-TV-UOT":
            result, _ = run_design(inst)
        else:
            result = solve_discrete(inst, method, SETTINGS["n_train"], SETTINGS["training_repeat"], SETTINGS["solver"])
        record = {"scale": scale, "seed": inst.seed, "method": method, "repeat": 0,
                  "settings": SETTINGS, "core_signature": signature, "input_signature": input_signature,
                  "created_local": created, "completed_local": datetime.now().isoformat(),
                  "training_sha256": digest(arrays["train"]) if method.startswith("SAA") else None,
                  "test_sha256": digest(arrays["test"]), "record_state": "planned", **result}
        if signature != source_signature():
            raise RuntimeError("Core changed during execution")
        write_json(plan_path(scale, method), record)
        print(f"DONE {scale}/{method}: {record['status']}, planning={record['timing']['planning']:.3f}s, gap={record['gap']}", flush=True)
    for method in POLICIES:
        record = load_record(scale, method, input_signature)
        if record["record_state"] == "evaluated":
            if record["inventory"] is not None and not sample_path(scale, method).exists():
                raise FileNotFoundError(sample_path(scale, method))
            continue
        if record["inventory"] is None:
            record.update(record_state="evaluated", evaluation_status="no_incumbent", evaluation_seconds=0.)
        else:
            stock, corrections = execution_inventory(record["inventory"], method.endswith("-I"),
                record.get("feasibility_tolerance", 1e-6), record.get("integer_tolerance", 1e-6))
            samples, seconds = evaluate_scenarios(inst, stock, arrays["test"])
            np.savez_compressed(sample_path(scale, method), **samples)
            record.update(record_state="evaluated", evaluation_status="completed", execution_inventory=stock,
                          inventory_corrections=corrections, metrics=summarize_samples(samples), evaluation_seconds=seconds)
        write_json(plan_path(scale, method), record)
        print(f"OOS {scale}/{method}: {record['evaluation_status']}", flush=True)


def report():
    summaries, observations, inventory_rows, boundaries = [], [], [], []
    for scale in SETTINGS["scales"]:
        inst = instance(scale, SETTINGS["fixed_seed"], bandwidth_sigma=SETTINGS["bandwidth_sigma"])
        inputs, input_signature = frozen_inputs(inst)
        for method in POLICIES:
            record = load_record(scale, method, input_signature)
            if record["record_state"] != "evaluated":
                raise RuntimeError(f"Unfinished evaluation: {scale}/{method}")
            row = {"scale": scale, "seed": inst.seed, "method": method, "repeat": 0,
                   "n_train": SETTINGS["n_train"] if method.startswith("SAA") else None,
                   "n_test": SETTINGS["n_test"], "bandwidth_km": SETTINGS["bandwidth_sigma"], "currency": "EUR",
                   "status": record["status"], "evaluation_status": record["evaluation_status"],
                   "solver": record.get("solver"), "solver_version": record.get("solver_version"),
                   "planning_seconds": record["timing"]["planning"], "solver_seconds": record.get("solve_seconds"),
                   "evaluation_seconds": record["evaluation_seconds"], "solver_gap": record["gap"],
                   "solver_lower_bound": record["lower_bound"], "solver_objective": record.get("objective"),
                   "planning_objective": record["planning_objective"],
                   "target_gap_reached": record.get("target_reached"),
                   "relative_inventory_change": record.get("relative_inventory_change")}
            if record["inventory"] is not None:
                with np.load(sample_path(scale, method)) as saved:
                    samples = {name: saved[name] for name in saved.files}
                if record["test_sha256"] != digest(inputs["test"]):
                    raise RuntimeError("Policy does not use the common test set")
                for metric, value in summarize_samples(samples).items():
                    np.testing.assert_allclose(record["metrics"][metric], value, rtol=1e-12, atol=1e-12)
                stock = np.asarray(record["execution_inventory"])
                row.update(record["metrics"], active_count=int((stock.sum(axis=1) > 0).sum()),
                           part1_stock=int(stock[:, 0].sum()), part2_stock=int(stock[:, 1].sum()),
                           total_stock=int(stock.sum()), rounding_increment=record["inventory_corrections"]["rounding_increment"])
                observations.append(pd.DataFrame({"scale": scale, "method": method,
                    "scenario_id": np.arange(1, SETTINGS["n_test"] + 1), **samples}))
                for warehouse in range(len(stock)):
                    inventory_rows.append({"scale": scale, "method": method, "warehouse_id": warehouse,
                        "part1_planned": record["inventory"][warehouse][0], "part2_planned": record["inventory"][warehouse][1],
                        "part1_executed": stock[warehouse, 0], "part2_executed": stock[warehouse, 1]})
            if method == "SD-TV-UOT":
                for part in range(2):
                    boundaries.append({"scale": scale, "part": part + 1,
                        "point_expected_demand": record["point_demand_mass"][part],
                        "continuous_domain_mass": record["continuous_domain_mass"][part],
                        "lost_mass": record["boundary_lost_mass"][part],
                        "lost_fraction": record["boundary_lost_mass"][part] / record["point_demand_mass"][part]})
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "table-exp2-summary.csv", index=False, encoding="utf-8-sig")
    summary.to_latex(OUT / "table-exp2-summary.tex", index=False, float_format="%.6f", na_rep="--")
    pd.DataFrame(inventory_rows).to_csv(OUT / "table-exp2-inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boundaries).to_csv(OUT / "table-exp2-boundary-mass.csv", index=False, encoding="utf-8-sig")
    data = pd.concat(observations, ignore_index=True)
    data.to_csv(OUT / "figure-exp2-1-data.csv", index=False, encoding="utf-8-sig")
    summary[["scale", "method", "solver", "planning_seconds", "solver_seconds", "evaluation_seconds", "status", "solver_gap"]].to_csv(
        OUT / "figure-exp2-2-data.csv", index=False, encoding="utf-8-sig")
    from plot_exp2 import make_figures
    make_figures(summary, data)
    write_json(OUT / "completion.json", {"completed": True, "policies": len(summary),
        "evaluated_policies": int(summary.evaluation_status.eq("completed").sum()), "scenario_costs": len(data),
        "settings": SETTINGS, "core_signature": source_signature(),
        "status_counts": summary.status.value_counts().to_dict(), "completed_local": datetime.now().isoformat()})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if not args.report_only:
        started = perf_counter()
        warmup()
        initialize_solver(SETTINGS["solver"])
        metadata = {"settings": SETTINGS, "platform": platform.platform(), "python": platform.python_version(),
            "versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "matplotlib", "numba", "gurobipy"]},
            "warmup_and_license_seconds": perf_counter() - started, "core_signature": source_signature(),
            "timing_policy": "Serial complete planning including preprocessing/model building/recovery; exclude input I/O, JIT/license setup, OOS and plotting"}
        write_json(OUT / "environment.json", metadata)
        for scale in SETTINGS["scales"]:
            run_scale(scale)
    report()


if __name__ == "__main__":
    main()

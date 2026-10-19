"""Read all local results and independently audit methods, recourse and figures."""
import os
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from environment import OUT, SETTINGS, POLICIES, instance, scenarios, digest, write_json
from evaluation import execution_inventory, summarize_samples
from experiment2 import source_signature, frozen_inputs, load_record, sample_path
from geometry import ExactEvaluator
from test_exp2 import independent_transport_cost, independent_retained_mass


def main():
    summary = pd.read_csv(OUT / "table-exp2-summary.csv")
    plotted = pd.read_csv(OUT / "figure-exp2-1-data.csv")
    runtimes = pd.read_csv(OUT / "figure-exp2-2-data.csv")
    assert len(summary) == len(runtimes) == 15
    assert summary.seed.eq(42).all()
    assert not summary.duplicated(["scale", "method"]).any()
    paper_costs = pd.read_csv(OUT / "figure-exp2-1-paper-data.csv")
    paper_times = pd.read_csv(OUT / "figure-exp2-2-paper-data.csv")
    selected = summary[summary.method != "SAA-I"].copy()
    selected.loc[selected.method == "SAA-C", "method"] = "SAA"
    pd.testing.assert_frame_equal(paper_times, selected.reset_index(drop=True))
    selected_costs = plotted[plotted.method != "SAA-I"].copy()
    selected_costs.loc[selected_costs.method == "SAA-C", "method"] = "SAA"
    pd.testing.assert_frame_equal(paper_costs, selected_costs.reset_index(drop=True))
    checks, boundary_checks, methods = [], [], []
    maximum_component_error = 0.
    for scale in "SML":
        inst = instance(scale, 42, bandwidth_sigma=5.)
        inputs, input_signature = frozen_inputs(inst)
        demands = inputs["test"]
        np.testing.assert_array_equal(inputs["train"], scenarios(inst, 200, "train", 0))
        for method in POLICIES:
            record = load_record(scale, method, input_signature)
            row = summary[(summary.scale == scale) & (summary.method == method)].iloc[0]
            assert record["test_sha256"] == digest(demands)
            assert record["repeat"] == 0
            if method != "SD-TV-UOT":
                assert record["solver"] == row.solver == "Gurobi" and record["threads"] == 1
                assert record["time_limit"] == 1800. and record["target_gap"] == .001
                assert record["n_train"] == (200 if method.startswith("SAA") else 1)
                if method.startswith("SAA"):
                    assert record["training_sha256"] == digest(inputs["train"])
                if record["gap"] is not None:
                    np.testing.assert_allclose(row.solver_gap, record["gap"], atol=1e-15)
                    assert record["target_reached"] == (record["gap"] <= .001)
                assert row.status == record["status"]
            else:
                evaluator = ExactEvaluator(inst)
                history = np.asarray(record["inventory_history"])
                np.testing.assert_allclose(history[0], np.tile(evaluator.total_mass / len(inst.sites), (len(inst.sites), 1)))
                for position, threshold in enumerate(record["threshold_history"]):
                    penalties = inst.fixed_costs / (history[position].sum(axis=1) + SETTINGS["epsilon"])
                    np.testing.assert_allclose(threshold, -penalties)
                    np.testing.assert_allclose(evaluator.evaluate(penalties).inventory, history[position + 1], atol=1e-9)
                    change = np.linalg.norm(history[position + 1] - history[position]) / max(1., np.linalg.norm(history[position]))
                    np.testing.assert_allclose(change, record["trace"][position]["relative_change"], atol=1e-12)
                assert np.all(np.diff([step["surrogate"] for step in record["trace"]]) < 1e-6)
                allowed = np.asarray(record["pre_recovery_inventory"]).sum(axis=1) > 0
                np.testing.assert_allclose(evaluator.evaluate(np.zeros(len(inst.sites)), allowed).inventory, record["inventory"], atol=1e-9)
                retained = np.array([independent_retained_mass(center, 5., inst.domain_size) for center in inst.xy])
                np.testing.assert_allclose(retained, evaluator.kernel_mass, rtol=1e-9, atol=1e-10)
                np.testing.assert_allclose(retained @ inst.probabilities, record["continuous_domain_mass"], atol=1e-8)
                boundary_checks.append({"scale": scale, "kernels": len(retained),
                    "maximum_quadrature_error": float(np.abs(retained - evaluator.kernel_mass).max()),
                    "lost_mass_by_part": record["boundary_lost_mass"]})
            np.testing.assert_allclose(row.planning_seconds, record["timing"]["planning"], rtol=1e-12)
            if record["inventory"] is None:
                assert row.evaluation_status == "no_incumbent"
                assert plotted[(plotted.scale == scale) & (plotted.method == method)].empty
                continue
            stock, corrections = execution_inventory(record["inventory"], method.endswith("-I"),
                record.get("feasibility_tolerance", 1e-6), record.get("integer_tolerance", 1e-6))
            np.testing.assert_array_equal(stock, record["execution_inventory"])
            np.testing.assert_allclose(row.rounding_increment, corrections["rounding_increment"])
            with np.load(sample_path(scale, method)) as saved:
                arrays = {name: saved[name] for name in saved.files}
            assert len(arrays["total_cost"]) == 1000
            np.testing.assert_array_equal(arrays["demand_amount"], demands.sum(axis=(1, 2)))
            np.testing.assert_array_equal(arrays["opening_cost"], np.full(1000, inst.fixed_costs[stock.sum(axis=1) > 0].sum()))
            components = sum(arrays[name] for name in ["opening_cost", "local_cost", "external_cost", "unused_cost"])
            error = float(np.abs(components - arrays["total_cost"]).max())
            maximum_component_error = max(maximum_component_error, error)
            np.testing.assert_allclose(components, arrays["total_cost"], atol=1e-9)
            for metric, value in summarize_samples(arrays).items():
                np.testing.assert_allclose(value, row[metric], rtol=1e-12, atol=1e-10)
            selected = plotted[(plotted.scale == scale) & (plotted.method == method)].sort_values("scenario_id")
            assert len(selected) == 1000
            np.testing.assert_allclose(selected.total_cost, arrays["total_cost"], rtol=1e-12)
            for index in np.unique(np.r_[0, 249, 499, 749, 999, np.argmin(arrays["total_cost"]), np.argmax(arrays["total_cost"]) ]):
                expected = independent_transport_cost(inst, stock, demands[index])["total_cost"]
                error = abs(expected - arrays["total_cost"][index])
                np.testing.assert_allclose(expected, arrays["total_cost"][index], rtol=1e-10, atol=1e-7)
                checks.append({"scale": scale, "method": method, "scenario_id": int(index + 1), "absolute_error": error})
            methods.append({"scale": scale, "method": method, "status": record["status"], "gap": record["gap"]})
    for name in ["figure-exp2-1", "figure-exp2-2"]:
        root = ET.parse(OUT / f"{name}.svg").getroot()
        labels = " ".join("".join(element.itertext()) for element in root.iter() if element.tag.endswith("}text"))
        for forbidden in ["seed", "COPT", "Gurobi", "c.u.", "scenarios", "gap", "SAA-C", "SAA-I"]:
            assert forbidden not in labels
        assert "SAA" in labels
        if name.endswith("-1"):
            assert "€" in labels
        else:
            assert "Solving time (s)" in labels
    pd.DataFrame(checks).to_csv(OUT / "independent-transport-checks.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "validation.json", {"passed": True, "core_signature": source_signature(),
        "policies": len(summary), "evaluated_policies": len(methods), "scenario_costs": len(plotted),
        "paper_policies": len(paper_times), "paper_scenario_costs": len(paper_costs), "paper_saa_source": "SAA-C",
        "independent_transport_checks": len(checks),
        "maximum_transport_error": max(check["absolute_error"] for check in checks),
        "maximum_component_error": maximum_component_error, "boundary_checks": boundary_checks,
        "solver_outcomes": methods})
    print(f"Validated {len(summary)} policies, {len(plotted)} scenario costs and {len(checks)} independent transport LPs")


if __name__ == "__main__":
    main()

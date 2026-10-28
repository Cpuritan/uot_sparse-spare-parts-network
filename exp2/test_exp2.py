"""Independent checks of Algorithm 1, clipped Green integration, Gurobi and OOS."""
from dataclasses import replace
import unittest
import numpy as np
from scipy import sparse
from scipy.integrate import quad
from scipy.optimize import linprog
from algorithm import run_design
from benchmark import initialize_solver, solve_discrete
from environment import OUT, SETTINGS, Instance, instance, scenarios, cost_values
from evaluation import execution_inventory, evaluate_scenarios, summarize_samples
from geometry import ExactEvaluator, polygon_moments, clipped_kernel_support, _cell_evaluate


def independent_transport_cost(inst, stock, demand):
    stock = np.asarray(stock, float)
    demand = np.asarray(demand, float)
    costs = cost_values(inst, inst.xy)
    parts = {name: 0. for name in ["local_cost", "external_cost", "unused_cost"]}
    for part in range(2):
        warehouses = np.flatnonzero(stock[:, part] > 0)
        nodes = np.flatnonzero(demand[:, part] > 0)
        warehouse_count, node_count = len(warehouses), len(nodes)
        if not warehouse_count or not node_count:
            parts["external_cost"] += float(costs[:, 0] @ demand[:, part])
            parts["unused_cost"] += float(inst.holding @ stock[:, part])
            continue
        flow_cost = costs[nodes[:, None], 1 + warehouses].T.ravel()
        objective = np.r_[flow_cost, costs[nodes, 0], inst.holding[warehouses]]
        demand_rows = sparse.kron(np.ones((1, warehouse_count)), sparse.eye(node_count), format="csr")
        stock_rows = sparse.kron(sparse.eye(warehouse_count), np.ones((1, node_count)), format="csr")
        constraint = sparse.vstack([
            sparse.hstack([demand_rows, sparse.eye(node_count), sparse.csr_matrix((node_count, warehouse_count))]),
            sparse.hstack([stock_rows, sparse.csr_matrix((warehouse_count, node_count)), sparse.eye(warehouse_count)])], format="csr")
        solution = linprog(objective, A_eq=constraint, b_eq=np.r_[demand[nodes, part], stock[warehouses, part]],
                           bounds=(0., None), method="highs")
        if not solution.success:
            raise RuntimeError(solution.message)
        flow_count = len(flow_cost)
        parts["local_cost"] += float(flow_cost @ solution.x[:flow_count])
        parts["external_cost"] += float(costs[nodes, 0] @ solution.x[flow_count:flow_count + node_count])
        parts["unused_cost"] += float(inst.holding[warehouses] @ solution.x[flow_count + node_count:])
    parts["opening_cost"] = float(inst.fixed_costs[stock.sum(axis=1) > 0].sum())
    parts["total_cost"] = sum(parts.values())
    return parts


def toy():
    probabilities = np.array([[.3, .1], [.2, .4]])
    return Instance("toy", 42, 10., np.array([[1., 2.], [7., 6.]]),
        np.array([[3., 3.], [7., 7.]]), probabilities, probabilities.copy(),
        np.array([2., 3.]), np.array([40., 41.]), np.array([.9, 1.]),
        np.array([.9, 1.]), np.array([0, 1]), np.array([0, 1]), bandwidth=5.)


def independent_retained_mass(center, bandwidth, length):
    def integral(horizontal):
        lower = max(-1., -2 * center[0] / bandwidth - horizontal,
                    horizontal - 2 * (length - center[1]) / bandwidth)
        upper = min(1., 2 * (length - center[0]) / bandwidth - horizontal,
                    horizontal + 2 * center[1] / bandwidth)
        if lower >= upper:
            return 0.
        vertical_mass = quad(lambda vertical: (35 / 32) * (1 - vertical ** 2) ** 3,
                             lower, upper, epsabs=1e-13)[0]
        return (35 / 32) * (1 - horizontal ** 2) ** 3 * vertical_mass
    return quad(integral, -1., 1., epsabs=1e-11, epsrel=1e-11, limit=250)[0]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = toy()
        cls.record, cls.evaluator = run_design(cls.inst)

    def test_boundary_mass_against_independent_quadrature(self):
        for center in [[0., 0.], [0., 5.], [1., 2.], [5., 5.], [9., 8.]]:
            polygon = clipped_kernel_support(np.array(center), 5., 10.)
            actual = polygon_moments(polygon)[0]
            self.assertAlmostEqual(actual, independent_retained_mass(center, 5., 10.), places=9)
        self.assertAlmostEqual(polygon_moments(clipped_kernel_support(np.array([0., 0.]), 5., 10.))[0], .25)
        self.assertAlmostEqual(polygon_moments(clipped_kernel_support(np.array([0., 5.]), 5., 10.))[0], .5)

    def test_mass_not_renormalized(self):
        self.assertTrue(np.all(self.evaluator.total_mass < self.inst.part_q))
        update = self.evaluator.evaluate(np.zeros(2))
        np.testing.assert_allclose(update.inventory.sum(axis=0) + update.outside_mass,
                                   self.evaluator.total_mass, atol=1e-12)
        np.testing.assert_allclose(self.record["initial_inventory"],
            np.tile(self.evaluator.total_mass / 2, (2, 1)), atol=1e-12)

    def test_shared_weights_synchronous_update_and_stopping(self):
        history = self.record["inventory_history"]
        for index, threshold in enumerate(self.record["threshold_history"]):
            penalties = self.inst.fixed_costs / (history[index].sum(axis=1) + SETTINGS["epsilon"])
            np.testing.assert_allclose(threshold, -penalties)
            np.testing.assert_allclose(history[index + 1], self.evaluator.evaluate(penalties).inventory)
            change = np.linalg.norm(history[index + 1] - history[index]) / max(1., np.linalg.norm(history[index]))
            self.assertAlmostEqual(change, self.record["trace"][index]["relative_change"])
        self.assertLessEqual(self.record["relative_inventory_change"], SETTINGS["tolerance"])
        self.assertTrue(np.all(np.diff([step["surrogate"] for step in self.record["trace"]]) < 1e-9))

    def test_recovery_only_on_terminal_positive_set(self):
        allowed = self.record["pre_recovery_inventory"].sum(axis=1) > 0
        np.testing.assert_allclose(self.record["inventory"], self.evaluator.evaluate(np.zeros(2), allowed).inventory)
        self.assertTrue(set(self.record["active"]) <= set(np.flatnonzero(allowed)))

    def test_zero_inventory_does_not_exclude_candidates(self):
        inst = replace(self.inst, fixed_costs=np.array([.01, .01]))
        update = ExactEvaluator(inst).evaluate(inst.fixed_costs / SETTINGS["epsilon"])
        self.assertTrue(np.all(update.inventory.sum(axis=1) > 0))

    def test_ties_external_then_warehouse_id(self):
        polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        coefficients = np.zeros((3, 3))
        for allowed, expected in [(np.ones(3, bool), [1., 0., 0.]),
                                   (np.array([False, True, True]), [0., 1., 0.])]:
            np.testing.assert_allclose(_cell_evaluate(polygon, coefficients, np.zeros(3), allowed)[0], expected, atol=1e-12)

    def test_fixed_instance_and_scenario_settings(self):
        self.assertEqual(SETTINGS["n_train"], 200)
        self.assertEqual(SETTINGS["training_repeat"], 0)
        self.assertEqual(SETTINGS["n_test"], 1000)
        self.assertEqual(SETTINGS["bandwidth_sigma"], 5.)
        for scale, expected_count in zip("SML", [100, 250, 500]):
            inst = instance(scale, 42)
            original = instance(scale, 42, bandwidth_sigma=1.)
            for name in ["xy", "sites", "probabilities", "fixed_costs", "base", "beta", "gamma"]:
                np.testing.assert_array_equal(getattr(inst, name), getattr(original, name))
            self.assertEqual(scenarios(inst, 1000).shape, (1000, expected_count, 2))
            np.testing.assert_array_equal(scenarios(inst, 200, "train"), scenarios(inst, 300, "train")[:200])
            self.assertFalse(np.array_equal(scenarios(inst, 200, "train"), scenarios(inst, 200)))

    def test_cost_branches(self):
        points = np.random.default_rng(11).uniform(0, 10, (40, 2))
        branches = np.column_stack([np.max(points @ family[:, :2].T + family[:, 2], axis=1)
                                    for family in self.inst.cost_branches])
        np.testing.assert_allclose(branches, cost_values(self.inst, points), atol=1e-13)

    def test_integer_execution_rule(self):
        raw = np.array([[1.00000001, .00000001], [0., 2.]])
        executed, _ = execution_inventory(raw)
        np.testing.assert_array_equal(executed, [[2, 1], [0, 2]])
        executed, _ = execution_inventory(raw, integer_planning=True, integer_tolerance=1e-6)
        np.testing.assert_array_equal(executed, [[1, 0], [0, 2]])

    def test_assignment_against_explicit_transport_lp(self):
        stock = np.array([[1, 0], [0, 1]])
        demands = np.concatenate([np.zeros((1, 2, 2), bool), np.ones((1, 2, 2), bool),
                                  scenarios(self.inst, 8)])
        arrays, _ = evaluate_scenarios(self.inst, stock, demands)
        for index, demand in enumerate(demands):
            expected = independent_transport_cost(self.inst, stock, demand)
            self.assertAlmostEqual(arrays["total_cost"][index], expected["total_cost"], places=9)

    def test_tail_and_external_ratio(self):
        arrays = {"total_cost": np.arange(1., 1001.), "demand_amount": np.full(1000, 4.),
                  "external_amount": np.ones(1000)}
        for part in [1, 2]:
            arrays[f"part{part}_demand_amount"] = np.full(1000, 2.)
            arrays[f"part{part}_external_amount"] = np.full(1000, .5)
        result = summarize_samples(arrays)
        self.assertEqual(result["mean_cost"], 500.5)
        self.assertEqual(result["cvar95"], 975.5)
        self.assertEqual(result["external_share"], .25)

    def test_gurobi_four_models_on_micro_instance(self):
        for method in ["MVP-C", "MVP-I", "SAA-C", "SAA-I"]:
            result = solve_discrete(self.inst, method, 200, 0)
            self.assertEqual(result["solver"], "Gurobi")
            self.assertEqual(result["threads"], 1)
            self.assertEqual(result["time_limit"], 1800.)
            self.assertTrue(result["has_incumbent"])
            self.assertLessEqual(result["gap"], .001)
            demands = scenarios(self.inst, 200, "train") if method.startswith("SAA") else self.inst.probabilities[None]
            expected = np.mean([independent_transport_cost(self.inst, result["inventory"], demand)["total_cost"] for demand in demands])
            self.assertAlmostEqual(expected, result["planning_objective"], places=5)

    def test_copt_backend_rejected(self):
        with self.assertRaises(ValueError):
            initialize_solver("copt")

    @unittest.skipUnless((OUT / "completion.json").exists(), "Run the experiment before artifact checks")
    def test_completed_artifacts(self):
        import pandas as pd
        import xml.etree.ElementTree as ET
        summary = pd.read_csv(OUT / "table-exp2-summary.csv")
        costs = pd.read_csv(OUT / "figure-exp2-1-data.csv")
        self.assertEqual(len(summary), 15)
        completed = int(summary.evaluation_status.eq("completed").sum())
        self.assertEqual(len(costs), 1000 * completed)
        self.assertTrue(summary.seed.eq(42).all())
        self.assertTrue(summary.loc[summary.method.str.startswith("SAA"), "n_train"].eq(200).all())
        self.assertTrue(summary.loc[summary.method != "SD-TV-UOT", "solver"].eq("Gurobi").all())
        paper_costs = pd.read_csv(OUT / "figure-exp2-1-paper-data.csv")
        paper_summary = pd.read_csv(OUT / "figure-exp2-2-paper-data.csv")
        self.assertEqual(len(paper_costs), 12000)
        self.assertEqual(len(paper_summary), 12)
        self.assertEqual(set(paper_summary.method), {"SD-TV-UOT", "MVP-C", "MVP-I", "SAA"})
        for scale in "SML":
            original = summary[(summary.scale == scale) & (summary.method == "SAA-C")].iloc[0]
            displayed = paper_summary[(paper_summary.scale == scale) & (paper_summary.method == "SAA")].iloc[0]
            self.assertEqual(original.planning_seconds, displayed.planning_seconds)
            np.testing.assert_allclose(
                costs[(costs.scale == scale) & (costs.method == "SAA-C")].total_cost,
                paper_costs[(paper_costs.scale == scale) & (paper_costs.method == "SAA")].total_cost,
                rtol=0, atol=1e-10)
        for name in ["figure-exp2-1", "figure-exp2-2"]:
            for extension in ["png", "pdf", "svg"]:
                self.assertTrue((OUT / f"{name}.{extension}").exists())
            root = ET.parse(OUT / f"{name}.svg").getroot()
            drawing = " ".join("".join(element.itertext()) for element in root.iter()
                               if element.tag.endswith("}text"))
            for forbidden in ["seed", "COPT", "Gurobi", "c.u.", "1800", "200 scenarios", "gap", "SAA-C", "SAA-I"]:
                self.assertNotIn(forbidden, drawing)
            if name.endswith("-2"):
                labels = [element for element in root.iter() if element.tag.endswith("}text")]
                self.assertIn("Solving time (s)", drawing)
                self.assertNotIn("Planning time", drawing)
                for scale in "SML":
                    self.assertIn(scale, ["".join(element.itertext()) for element in labels])
                legend = [element for element in labels if "".join(element.itertext())
                          in ["SD-TV-UOT", "MVP-C", "MVP-I", "SAA"]]
                self.assertEqual(len(legend), 4)
                positions = [float(element.attrib["y"]) for element in legend]
                self.assertLess(max(positions) - min(positions), 1e-6)
        self.assertIn("€", (OUT / "figure-exp2-1.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

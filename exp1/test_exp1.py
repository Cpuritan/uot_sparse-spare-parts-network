"""Independent checks of the manuscript updates and the two experiment scopes."""
from dataclasses import replace
import csv
from itertools import product
import json
import unittest
import numpy as np
from scipy.integrate import dblquad
from algorithm import MonteCarloEvaluator, run_design
from benchmark import solve_continuous_milp
from diagnostics import align, path_metrics, trajectory_records
from environment import DATA, OUT, SETTINGS, Instance, initial_inventory, instance, master, cost_values, continuous_samples
from evaluation import continuous_cost
from geometry import ExactEvaluator, polygon_moments, _cell_evaluate


def toy():
    return Instance("toy", 42, 10., np.array([[3., 4.], [7., 6.]]),
        np.array([[3., 3.], [7., 7.]]), np.array([[.3, .1], [.2, .4]]),
        np.array([[.3, .1], [.2, .4]]), np.array([2., 3.]), np.array([40., 41.]),
        np.array([.9, 1.]), np.array([.9, 1.]), np.array([0, 1]), np.array([0, 1]))


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = toy()
        cls.record, cls.evaluator = run_design(cls.inst)

    def test_manuscript_snapshot(self):
        source = (DATA / "manuscript.tex").read_text(encoding="utf-8")
        for label in ["eq:design_weights", "eq:design_partition", "eq:design_update"]:
            self.assertIn(label, source)

    def test_frozen_twenty_inputs(self):
        for seed in SETTINGS["seeds"]:
            pool = master(seed)
            self.assertEqual(pool["p"].shape, (500, 2))
            self.assertFalse(np.array_equal(pool["p"][:, 0], pool["p"][:, 1]))
            self.assertTrue(np.all((pool["p"][:, 0] >= .15) & (pool["p"][:, 0] <= .35)))
            self.assertTrue(np.all((pool["p"][:, 1] >= .05) & (pool["p"][:, 1] <= .25)))
        for scale in "SML":
            inst = instance(scale, 42)
            np.testing.assert_allclose(initial_inventory(inst), np.tile(inst.part_q / len(inst.sites), (len(inst.sites), 1)))
            self.assertNotEqual(inst.part_q[0], inst.part_q[1])

    def test_green_moments_against_area_quadrature(self):
        polygon = np.array([[-.7, -.8], [.8, -.8], [.8, .2], [-.7, .2]])
        density = lambda vertical, horizontal: (35 / 32) ** 2 * (1 - horizontal ** 2) ** 3 * (1 - vertical ** 2) ** 3
        expected = [dblquad(lambda vertical, horizontal: density(vertical, horizontal) * factor(horizontal, vertical),
                    -.7, .8, lambda horizontal: -.8, lambda horizontal: .2, epsabs=1e-13)[0]
                    for factor in [lambda horizontal, vertical: 1., lambda horizontal, vertical: horizontal,
                                   lambda horizontal, vertical: vertical]]
        np.testing.assert_allclose(polygon_moments(polygon), expected, atol=2e-14)
        np.testing.assert_allclose(polygon_moments(np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])), [1., 0., 0.], atol=2e-14)

    def test_max_affine_cost_is_original_cost(self):
        points = np.random.default_rng(14).uniform(0, 10, (50, 2))
        values = np.column_stack([np.max(points @ branches[:, :2].T + branches[:, 2], axis=1)
                                  for branches in self.inst.cost_branches])
        np.testing.assert_allclose(values, cost_values(self.inst, points), atol=1e-13)

    def test_tie_order_external_then_warehouse(self):
        polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        coefficients = np.zeros((3, 3))
        for allowed, expected in [(np.ones(3, bool), [1., 0., 0.]),
                                   (np.array([False, True, True]), [0., 1., 0.])]:
            result = _cell_evaluate(polygon, coefficients, np.zeros(3), allowed)
            np.testing.assert_allclose(result[0], expected, atol=1e-12)

    def test_shared_fixed_epsilon_synchronous_update(self):
        record = self.record
        history = record["inventory_history"]
        np.testing.assert_allclose(history[0], initial_inventory(self.inst))
        for position, threshold in enumerate(record["threshold_history"]):
            expected = self.inst.fixed_costs / (history[position].sum(axis=1) + 1.)
            np.testing.assert_allclose(threshold, -expected)
            update = self.evaluator.evaluate(expected)
            np.testing.assert_allclose(history[position + 1], update.inventory, atol=1e-12)
            change = np.linalg.norm(history[position + 1] - history[position]) / max(1., np.linalg.norm(history[position]))
            self.assertAlmostEqual(change, record["trace"][position]["relative_change"])
        self.assertLessEqual(record["trace"][-1]["relative_change"], SETTINGS["tolerance"])
        self.assertTrue(np.all(np.diff([step["surrogate"] for step in record["trace"]]) <= 1e-10))

    def test_zero_stock_candidates_remain_eligible(self):
        small_fixed = replace(self.inst, fixed_costs=np.array([.01, .01]))
        evaluator = ExactEvaluator(small_fixed)
        update = evaluator.evaluate(small_fixed.fixed_costs / (np.zeros(2) + 1.))
        self.assertTrue(np.all(update.inventory.sum(axis=1) > 0))

    def test_recovery_uses_terminal_set_only(self):
        allowed = self.record["pre_recovery_inventory"].sum(axis=1) > 0
        expected = self.evaluator.evaluate(np.zeros(2), allowed)
        np.testing.assert_allclose(self.record["inventory"], expected.inventory, atol=1e-12)
        self.assertTrue(set(self.record["active"]).issubset(set(np.flatnonzero(allowed))))
        np.testing.assert_allclose(expected.inventory.sum(axis=0) + expected.outside_mass, self.inst.part_q, atol=1e-12)

    def test_mc_prefix_streams_and_mass(self):
        for part in range(2):
            small = continuous_samples(self.inst, 100, 0, part)
            large = continuous_samples(self.inst, 1000, 0, part)
            np.testing.assert_array_equal(small, large[:100])
            self.assertFalse(np.array_equal(small, continuous_samples(self.inst, 100, 1, part)))
        self.assertFalse(np.array_equal(continuous_samples(self.inst, 100, 0, 0), continuous_samples(self.inst, 100, 0, 1)))
        sampler = MonteCarloEvaluator(self.inst, 1000, 0)
        update = sampler.evaluate(np.zeros(2))
        self.assertEqual(sum(sampler.counts), 1000)
        np.testing.assert_allclose(update.inventory.sum(axis=0) + update.outside_mass, self.inst.part_q, atol=1e-12)
        np.testing.assert_array_equal(update.inventory, sampler.evaluate(np.zeros(2)).inventory)

    def test_continuous_certification_does_not_change_stock(self):
        stock = self.record["inventory"].copy()
        original = stock.copy()
        result = continuous_cost(self.inst, stock, self.evaluator)
        np.testing.assert_array_equal(stock, original)
        self.assertEqual(result["status"], "certified")
        self.assertLessEqual(result["lower"], self.record["planning_objective"])
        self.assertGreaterEqual(result["upper"], self.record["planning_objective"])

    def test_complete_ordering_milp_against_all_subsets(self):
        candidate_costs = []
        for bits in product([False, True], repeat=2):
            update = self.evaluator.evaluate(np.zeros(2), np.array(bits))
            candidate_costs.append(update.local_cost + update.external_cost + self.inst.fixed_costs[update.inventory.sum(axis=1) > 0].sum())
        result, coefficients = solve_continuous_milp(self.inst, time_limit=30.)
        self.assertEqual(result["solver"], "Gurobi")
        self.assertTrue(result["has_incumbent"])
        self.assertLessEqual(result["gap"], SETTINGS["mip_gap"])
        self.assertAlmostEqual(result["planning_objective"], min(candidate_costs), places=7)
        np.testing.assert_allclose(coefficients["mass"].sum(axis=0), self.inst.part_q, atol=1e-12)


class DiagnosticTests(unittest.TestCase):
    def test_quality_extra_repeats_do_not_enter_trajectory_scope(self):
        records = {"reference": {"method": "SD-TV-UOT", "budget": 0, "repeat": 0, "iterations": 3},
                   "milp": {"method": "MILP", "budget": 0, "repeat": 0, "iterations": 99}}
        records.update({str(repeat): {"method": "MC-1000", "budget": 1000, "repeat": repeat,
                                      "iterations": 5 if repeat < 5 else 200} for repeat in range(20)})
        selected = trajectory_records(records)
        self.assertEqual(len(selected), 6)
        self.assertEqual(max(record["iterations"] for record in selected), 5)

    def test_alignment_holds_terminal_not_recovery(self):
        history = np.array([[[1., 2.]], [[3., 4.]]])
        aligned = align(history, 4)
        np.testing.assert_array_equal(aligned[1:], np.repeat(history[-1:], 4, axis=0))
        with self.assertRaises(ValueError):
            align(history, 0)

    def test_rms_aggregation_and_dispersion_identity(self):
        reference = np.zeros((3, 2, 2))
        repetitions = np.stack([reference + offset for offset in [0., 1., 2., 3., 4.]])
        deviation, dispersion, mean_deviation, each = path_metrics(reference, repetitions, 10.)
        self.assertAlmostEqual(deviation, np.sqrt(np.mean(each ** 2)))
        self.assertNotAlmostEqual(deviation, each.mean())
        self.assertAlmostEqual(deviation ** 2, dispersion ** 2 + mean_deviation ** 2)


class SavedResultsTests(unittest.TestCase):
    @unittest.skipUnless((OUT / "validation.json").exists(), "Full experimental results not generated yet")
    def test_saved_paths_and_observations_independently(self):
        with (OUT / "trajectory_instances.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 180)
        indexed = {(row["scale"], int(row["seed"]), int(row["budget"])): row for row in rows}
        self.assertEqual(len(indexed), 180)
        total_records = 0
        for scale, seed in product("SML", SETTINGS["seeds"]):
            bundle = json.loads((OUT / f"case_{scale}_{seed}.json").read_text(encoding="utf-8"))
            self.assertTrue(bundle["completed"])
            records = list(bundle["records"].values())
            total_records += len(records)
            selected = [record for record in records if record["method"] == "SD-TV-UOT" or
                        (record["budget"] > 0 and record["repeat"] in range(5))]
            self.assertEqual(len(selected), 16)
            horizon = max(record["iterations"] for record in selected)
            inst = instance(scale, seed)
            paths = {}
            for record in selected:
                history = np.asarray(record["inventory_history"])
                outside = inst.part_q - history.sum(axis=1)
                balanced = np.concatenate([outside[:, None, :], history], axis=1)
                paths[record["budget"], record["repeat"]] = np.concatenate([balanced,
                    np.repeat(balanced[-1:], horizon + 1 - len(balanced), axis=0)])
            with np.load(OUT / f"diagnostic_{scale}_{seed}.npz") as saved:
                np.testing.assert_allclose(paths[0, 0], saved["aligned_reference"], atol=1e-12)
                for budget in SETTINGS["mc_sizes"]:
                    samples = np.stack([paths[budget, repeat] for repeat in range(5)])
                    np.testing.assert_allclose(samples, saved[f"aligned_{budget}"], atol=1e-12)
                    deviation = np.linalg.norm(samples - paths[0, 0]) / (inst.Q * np.sqrt(5 * (horizon + 1)))
                    dispersion = np.linalg.norm(samples - samples.mean(axis=0)) / (inst.Q * np.sqrt(5 * (horizon + 1)))
                    row = indexed[scale, seed, budget]
                    self.assertEqual(int(row["horizon"]), horizon)
                    self.assertEqual(int(row["n_repeats"]), 5)
                    self.assertAlmostEqual(deviation, float(row["D"]), places=13)
                    self.assertAlmostEqual(dispersion, float(row["V"]), places=13)
        self.assertEqual(total_records, 1098)


if __name__ == "__main__":
    unittest.main()

"""Manuscript Algorithm 1: exact shared reweighting and final reoptimization."""
from time import perf_counter
import numpy as np
from environment import SETTINGS, Instance
from geometry import ExactEvaluator, nominal_regions


def run_design(inst):
    started = perf_counter()
    evaluator = ExactEvaluator(inst)
    inventory = np.tile(evaluator.total_mass / len(inst.sites), (len(inst.sites), 1))
    preprocess_seconds = perf_counter() - started
    history, thresholds, trace = [inventory.copy()], [], []
    iteration_started = perf_counter()
    for iteration in range(1, SETTINGS["max_iterations"] + 1):
        penalties = inst.fixed_costs / (inventory.sum(axis=1) + SETTINGS["epsilon"])
        update = evaluator.evaluate(penalties)
        updated_inventory = update.inventory
        change = np.linalg.norm(updated_inventory - inventory) / max(1., np.linalg.norm(inventory))
        operational = update.local_cost + update.external_cost
        trace.append({"iteration": iteration, "relative_change": float(change),
                      "active_count": int(np.count_nonzero(updated_inventory.sum(axis=1) > 0)),
                      "surrogate": float(operational + inst.fixed_costs @ np.log1p(
                          updated_inventory.sum(axis=1) / SETTINGS["epsilon"])),
                      "economic_cost": float(operational + inst.fixed_costs[updated_inventory.sum(axis=1) > 0].sum()),
                      "external_mass": update.outside_mass, "mass_corrections": update.mass_corrections})
        inventory = updated_inventory.copy()
        history.append(inventory)
        thresholds.append(-penalties)
        if change <= SETTINGS["tolerance"]:
            break
    iteration_seconds = perf_counter() - iteration_started
    recovery_started = perf_counter()
    recovered = evaluator.evaluate(np.zeros(len(inst.sites)), inventory.sum(axis=1) > 0)
    final = recovered.inventory
    recovery_seconds = perf_counter() - recovery_started
    regions_started = perf_counter()
    regions = nominal_regions(inst, final.sum(axis=1) > 0)
    regions_seconds = perf_counter() - regions_started
    cost = recovered.local_cost + recovered.external_cost + inst.fixed_costs[final.sum(axis=1) > 0].sum()
    record = {"inventory": final, "active": np.flatnonzero(final.sum(axis=1) > 0),
              "initial_inventory": history[0], "pre_recovery_inventory": inventory,
              "inventory_history": np.asarray(history), "threshold_history": np.asarray(thresholds),
              "trace": trace, "iterations": iteration, "relative_inventory_change": float(change),
              "status": "converged" if change <= SETTINGS["tolerance"] else "iteration_limit",
              "epsilon": SETTINGS["epsilon"], "planning_objective": float(cost),
              "pre_recovery_objective": trace[-1]["economic_cost"],
              "recovery_cost_change": float(cost - trace[-1]["economic_cost"]), "regions": regions,
              "point_demand_mass": inst.part_q, "continuous_domain_mass": evaluator.total_mass,
              "boundary_lost_mass": inst.part_q - evaluator.total_mass,
              "kernel_retained_mass": evaluator.kernel_mass,
              "timing": {"preprocess": preprocess_seconds, "iteration": iteration_seconds,
                         "recovery": recovery_seconds, "final_regions": regions_seconds,
                         "planning": perf_counter() - started},
              "solver": None, "gap": None, "lower_bound": None}
    return record, evaluator


def warmup():
    probabilities = np.array([[.2, .3]])
    inst = Instance("toy", 0, 10., np.array([[1., 2.]]), np.array([[4., 4.], [6., 6.]]),
                    probabilities, probabilities.copy(), np.ones(2), np.full(2, 40.),
                    np.full(2, .9), np.full(2, .9), np.array([0]), np.array([0, 1]), bandwidth=5.)
    evaluator = ExactEvaluator(inst)
    evaluator.evaluate(np.ones(2))
    nominal_regions(inst, np.ones(2, bool))

"""Manuscript Algorithm 1; the MC versions change only regional integration."""
from time import perf_counter
import numpy as np
from environment import SETTINGS, continuous_samples, cost_values, initial_inventory
from geometry import ExactEvaluator, Evaluation, nominal_regions


class MonteCarloEvaluator:
    def __init__(self, inst, count, repeat):
        started = perf_counter()
        self.inst, self.count = inst, count
        first = int(np.floor(count * inst.part_q[0] / inst.Q))
        self.counts = [first, count - first]
        self.points = [continuous_samples(inst, n, repeat, p) for p, n in enumerate(self.counts)]
        self.costs = [cost_values(inst, points) for points in self.points]
        self.preprocess_seconds = perf_counter() - started

    def evaluate(self, penalties, allowed=None):
        m = len(self.inst.sites)
        allowed = np.ones(m, bool) if allowed is None else np.asarray(allowed, bool)
        masses, costs, external = [np.zeros((m + 1, 2)) for _ in range(3)]
        for p, original in enumerate(self.costs):
            values = original + np.r_[0., penalties]
            values[:, np.r_[False, ~allowed]] = np.inf
            choices = values.argmin(axis=1)  # external, then ascending warehouse ID
            weight = self.inst.part_q[p] / self.counts[p]
            masses[:, p] = np.bincount(choices, minlength=m + 1) * weight
            costs[:, p] = np.bincount(choices, weights=original[np.arange(self.counts[p]), choices], minlength=m + 1) * weight
            external[:, p] = np.bincount(choices, weights=original[:, 0], minlength=m + 1) * weight
        return Evaluation(masses[1:], float(costs[1:].sum()), float(costs[0].sum()), masses[0],
                          float(costs.sum() + penalties @ masses[1:].sum(axis=1)), costs, external)


def run_design(inst, epsilon=1., initialization="default", initial_repeat=0,
               mc_size=0, repeat=0, fixed_weights=False):
    started = perf_counter()
    evaluator = MonteCarloEvaluator(inst, mc_size, repeat) if mc_size else ExactEvaluator(inst)
    preprocess = perf_counter() - started
    if epsilon <= 0:
        raise ValueError("epsilon must be positive and fixed throughout a run")
    inventory = initial_inventory(inst, initialization, initial_repeat)
    history, thresholds, trace = [inventory.copy()], [], []
    iteration_start = perf_counter()
    for iteration in range(1, SETTINGS["max_iterations"] + 1):
        penalties = inst.fixed_costs / (inventory.sum(axis=1) + epsilon)
        # Every candidate participates, including zero-current-stock candidates.
        update = evaluator.evaluate(penalties)
        new = update.inventory
        change = np.linalg.norm(new - inventory) / max(1., np.linalg.norm(inventory))
        operational = update.local_cost + update.external_cost
        trace.append({"iteration": iteration, "relative_change": float(change),
                      "active_count": int(np.count_nonzero(new.sum(axis=1) > 0)),
                      "surrogate": float(operational + inst.fixed_costs @ np.log1p(new.sum(axis=1) / epsilon)),
                      "economic_cost": float(operational + inst.fixed_costs[new.sum(axis=1) > 0].sum()),
                      "external_mass": update.outside_mass,
                      "mass_corrections": update.mass_corrections})
        inventory = new.copy()
        history.append(inventory)
        thresholds.append(-penalties)
        if change <= SETTINGS["tolerance"] or fixed_weights:
            break
    iteration_seconds = perf_counter() - iteration_start
    recovery_start = perf_counter()
    recovery = evaluator.evaluate(np.zeros(len(inventory)), inventory.sum(axis=1) > 0)
    final = recovery.inventory
    recovery_seconds = perf_counter() - recovery_start
    output_start = perf_counter()
    regions = nominal_regions(inst, final.sum(axis=1) > 0)
    output_seconds = perf_counter() - output_start
    cost = recovery.local_cost + recovery.external_cost + inst.fixed_costs[final.sum(axis=1) > 0].sum()
    result = {"inventory": final, "active": np.flatnonzero(final.sum(axis=1) > 0),
              "pre_recovery_inventory": inventory, "initial_inventory": history[0],
              "inventory_history": np.asarray(history), "threshold_history": np.asarray(thresholds),
              "trace": trace, "iterations": iteration,
              "status": "fixed_weights" if fixed_weights else ("converged" if change <= SETTINGS["tolerance"] else "iteration_limit"),
              "epsilon": epsilon, "planning_objective": float(cost),
              "pre_recovery_objective": trace[-1]["economic_cost"],
              "recovery_cost_change": float(cost - trace[-1]["economic_cost"]),
              "regions": regions,
              "timing": {"preprocess": preprocess, "iteration": iteration_seconds,
                         "recovery": recovery_seconds, "final_regions": output_seconds,
                         "planning": perf_counter() - started},
              "sample_size": mc_size, "sample_repeat": repeat,
              "sample_sizes_by_part": evaluator.counts if mc_size else [0, 0],
              "recovery_mass_corrections": recovery.mass_corrections,
              "solver": None, "gap": None, "lower_bound": None}
    # The common fixed-stock evaluator is deliberately OUTSIDE planning timing.
    return result, evaluator


def warmup():
    """Compile numerical kernels on a deterministic toy problem, separately timed."""
    from environment import Instance
    inst = Instance("toy", 0, 10., np.array([[5., 5.]]), np.array([[4., 4.], [6., 6.]]),
                    np.array([[.2, .3]]), np.array([[.2, .3]]), np.ones(2), np.full(2, 40.),
                    np.full(2, .9), np.full(2, .9), np.array([0]), np.array([0, 1]))
    ev = ExactEvaluator(inst)
    ev.evaluate(np.ones(2))
    from geometry import ordering_coefficients
    ordering_coefficients(ev)
    nominal_regions(inst, np.ones(2, bool))

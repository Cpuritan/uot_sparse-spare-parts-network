"""Fixed-stock continuous primal/dual certification, outside planning timing."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from environment import SETTINGS
from geometry import ExactEvaluator


def _part_cost(inst, stock, part, ev):
    """Feasible primal upper bound and dual lower bound from manuscript Section 3.1.

    For every dual region, multiply its demand measure by min(1,m_i/q_i).
    This gives a feasible primal allocation without changing inventory; the
    remaining measure is served externally. This remains a valid certificate
    if the numerical dual maximizer has not converged.
    """
    allowed = stock > 0
    ids = np.flatnonzero(allowed)
    best_lower, best_upper = -np.inf, np.inf
    best_weights, calls = np.zeros(len(stock)), 0
    best_parts = None

    def objective(x):
        nonlocal best_lower, best_upper, calls, best_weights, best_parts
        weights = np.zeros(len(stock))
        weights[ids] = x
        r = ev.evaluate(-weights, allowed, part=part)
        demand = r.inventory[:, 0]
        lower = r.weighted_objective + stock @ weights
        fraction = np.ones(len(stock))
        positive = demand > 0
        fraction[positive] = np.minimum(1., stock[positive] / demand[positive])
        local = float(fraction @ r.option_costs[1:, 0])
        external = float(r.external_cost + (1 - fraction) @ r.external_by_option[1:, 0])
        unused = float(inst.holding @ np.maximum(stock - fraction * demand, 0.))
        upper = local + external + unused
        calls += 1
        if lower > best_lower:
            best_lower, best_weights = float(lower), weights.copy()
        if upper < best_upper:
            best_upper = float(upper)
            best_parts = {"local": local, "external": external, "unused": unused}
        return -float(lower), (demand - stock)[ids]

    objective(np.zeros(len(ids)))
    message = "zero-weight certificate"
    if len(ids) and (best_upper - best_lower) / max(1., abs(best_upper)) > SETTINGS["evaluation_gap"]:
        opt = minimize(objective, np.zeros(len(ids)), method="L-BFGS-B", jac=True,
                       bounds=[(None, 16.)] * len(ids),
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000, "maxls": 50, "maxcor": 20})
        objective(opt.x)
        message = str(opt.message)
    return {"lower": best_lower, "upper": best_upper, "weights": best_weights,
            "cost_components": best_parts, "optimizer_message": message, "calls": calls}


def continuous_cost(inst, inventory, evaluator=None):
    """Certify each fixed stock column separately, then charge opening once."""
    started = perf_counter()
    stock = np.asarray(inventory, float)
    if stock.shape != (len(inst.sites), 2) or np.any(stock < 0) or not np.isfinite(stock).all():
        raise ValueError("Prescribed inventory must be a nonnegative M-by-2 matrix")
    ev = ExactEvaluator(inst) if evaluator is None else evaluator
    parts = [_part_cost(inst, stock[:, p], p, ev) for p in range(2)]
    fixed = float(inst.fixed_costs[stock.sum(axis=1) > 0].sum())
    best_lower = fixed + sum(p["lower"] for p in parts)
    best_upper = fixed + sum(p["upper"] for p in parts)
    # A numerical floating-point allowance, not an interval-arithmetic proof.
    allowance = 5e-10 * max(1., abs(best_upper))
    lower, upper = best_lower - allowance, best_upper + allowance
    gap = max(0., upper - lower) / max(1., abs(upper))
    if lower > upper + allowance:
        raise ArithmeticError("Continuous primal/dual bounds are inconsistent")
    return {"lower": lower, "upper": upper, "relative_gap": gap,
            "status": "certified" if gap <= SETTINGS["evaluation_gap"] else "interval_only",
            "parts": parts, "calls": sum(p["calls"] for p in parts),
            "weights": np.column_stack([p["weights"] for p in parts]),
            "cost_components": {"opening": fixed, **{k: sum(p["cost_components"][k] for p in parts)
                                 for k in ["local", "external", "unused"]}},
            "integration_allowance": allowance,
            "seconds": perf_counter() - started}

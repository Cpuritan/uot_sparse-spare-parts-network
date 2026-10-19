"""Common integer execution for every frozen policy, with four economic costs."""
from time import perf_counter
import numpy as np
from scipy.optimize import linear_sum_assignment

from environment import cost_values


def execution_inventory(inventory, integer_planning=False, feasibility_tolerance=1e-6, integer_tolerance=1e-6):
    raw = np.asarray(inventory, float)
    if raw.ndim != 2 or raw.shape[1] != 2 or np.any(raw < -feasibility_tolerance) or not np.isfinite(raw).all():
        raise ValueError("Infeasible planned inventory")
    stock = np.maximum(raw, 0.)
    if integer_planning:
        if np.any(np.abs(stock - np.rint(stock)) > integer_tolerance):
            raise ValueError("Integer solver stock fails integrality tolerance")
        stock = np.rint(stock)
    else:
        # No snapping of positive continuous inventories to zero or integers.
        stock = np.ceil(stock)
    totals = np.maximum(raw, 0.).sum(axis=1)
    active = totals > 0
    if not integer_planning and not np.array_equal(active, stock.sum(axis=1) > 0):
        raise ArithmeticError("Ceiling changed the strictly positive opening set")
    return stock.astype(int), {"raw": raw, "executed": stock, "integer_planning": integer_planning,
                             "feasibility_tolerance": feasibility_tolerance,
                             "integer_tolerance": integer_tolerance,
                             "negative_entries_corrected": int(np.count_nonzero(raw < 0)),
                             "exact_zero_warehouses": int(np.count_nonzero(totals == 0)),
                             "minimum_positive_warehouse_stock": float(totals[active].min()) if active.any() else None,
                             "positive_subunit_entries": int(np.count_nonzero((raw > 0) & (raw < 1))),
                             "positive_subunit_warehouses": int(np.count_nonzero((totals > 0) & (totals < 1))),
                             "open_before_rounding": int(active.sum()),
                             "open_after_rounding": int(np.count_nonzero(stock.sum(axis=1) > 0)),
                             "rounding_increment_by_part": (stock - raw).sum(axis=0),
                             "rounding_increment": float(stock.sum() - raw.sum())}


def evaluate_scenarios(inst, integer_inventory, demands):
    """Integer transportation as a rectangular linear assignment problem.

    Each inventory unit is a slot. Dummy external slots are unlimited up to
    total realized demand. The reduced local cost c-a1-a2 incorporates saved
    holding cost. This exact integer execution needs no MIP per scenario.
    """
    started = perf_counter()
    stock = np.asarray(integer_inventory, int)
    if stock.shape != (len(inst.sites), 2) or np.any(stock < 0) or not np.array_equal(stock, integer_inventory):
        raise ValueError("Execution inventory must be nonnegative integers")
    demands = np.asarray(demands, bool)
    if demands.shape[1:] != (len(inst.xy), 2):
        raise ValueError("Each joint scenario must include both spare-part types")
    costs = cost_values(inst, inst.xy)
    a1, local = costs[:, 0], costs[:, 1:].T
    slots = [np.repeat(np.arange(len(stock)), stock[:, p]) for p in range(2)]
    count = len(demands)
    arrays = {k: np.zeros(count) for k in ["opening_cost", "local_cost", "external_cost", "unused_cost",
                                          "external_amount", "demand_amount", "unused_amount", "total_cost"]}
    part_metrics = ["local_cost", "external_cost", "unused_cost", "external_amount", "demand_amount", "unused_amount"]
    arrays.update({f"part{p + 1}_{key}": np.zeros(count) for p in range(2) for key in part_metrics})
    arrays["opening_cost"][:] = float(inst.fixed_costs[stock.sum(axis=1) > 0].sum())
    for s, demand in enumerate(demands):
        for p in range(2):
            nodes = np.flatnonzero(demand[:, p])
            shipped = np.zeros(len(stock), int)
            served = np.zeros(len(nodes), bool)
            local_cost = 0.
            if len(nodes) and len(slots[p]):
                savings = (local[slots[p]][:, nodes] - a1[nodes] - inst.holding[slots[p], None]).T
                matrix = np.column_stack((np.zeros((len(nodes), len(nodes))), savings))
                row, col = linear_sum_assignment(matrix)
                use = col >= len(nodes)
                used_nodes, warehouses = row[use], slots[p][col[use] - len(nodes)]
                served[used_nodes] = True
                shipped = np.bincount(warehouses, minlength=len(stock))
                local_cost = float(local[warehouses, nodes[used_nodes]].sum())
            values = [local_cost, float(a1[nodes[~served]].sum()),
                      float(inst.holding @ (stock[:, p] - shipped)), int((~served).sum()),
                      len(nodes), int((stock[:, p] - shipped).sum())]
            for key, value in zip(part_metrics, values):
                arrays[f"part{p + 1}_{key}"][s] = value
                arrays[key][s] += value
        arrays["total_cost"][s] = sum(arrays[k][s] for k in ["opening_cost", "local_cost", "external_cost", "unused_cost"])
    return arrays, perf_counter() - started


def summarize_samples(arrays, count=None):
    n = len(arrays["total_cost"]) if count is None else count
    v = arrays["total_cost"][:n]
    tail = int(round(.05 * n))
    if tail < 1 or not np.isclose(tail, .05 * n):
        raise ValueError("Use scenario counts with an integral 5% tail")
    total = float(arrays["demand_amount"][:n].sum())
    part_shares = {}
    for p in range(2):
        demand = arrays[f"part{p + 1}_demand_amount"][:n].sum()
        part_shares[f"part{p + 1}_external_share"] = float(arrays[f"part{p + 1}_external_amount"][:n].sum() / demand) if demand else None
    return {"mean_cost": float(v.mean()), "cvar95": float(np.partition(v, n - tail)[-tail:].mean()),
            "external_share": float(arrays["external_amount"][:n].sum() / total) if total else None,
            "n_test": n, **part_shares, **{f"mean_{k}": float(a[:n].mean()) for k, a in arrays.items() if k != "total_cost"}}

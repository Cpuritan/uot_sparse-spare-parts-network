"""MVP/SAA deterministic equivalents, solved once per policy using Gurobi.

All baselines retain the same economic objective, including its constant, so
relative MIP gaps refer to total cost rather than an arbitrary savings shift.
"""
from time import perf_counter
from importlib.metadata import version
import numpy as np
from scipy import sparse
from environment import SETTINGS, cost_values, scenarios

_ENVIRONMENTS = {}


def initialize_solver(backend="gurobi"):
    """Create the licensed environment before any timed planning call."""
    if backend not in _ENVIRONMENTS:
        if backend == "gurobi":
            import gurobipy as gp
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
        else:
            raise ValueError(backend)
        _ENVIRONMENTS[backend] = env
    return _ENVIRONMENTS[backend]


def solve_matrix(c, matrix, rhs, upper, types, constant, backend="gurobi", time_limit=None):
    """One mathematical matrix and the installed solver's DEFAULT feas/int tolerances."""
    time_limit = SETTINGS["time_limit"] if time_limit is None else time_limit
    setup = perf_counter()
    if backend == "gurobi":
        import gurobipy as gp
        env = initialize_solver(backend)
        model = gp.Model("paper_experiment", env=env)
        model.Params.Threads = SETTINGS["threads"]
        model.Params.MIPGap = SETTINGS["mip_gap"]
        model.Params.TimeLimit = time_limit
        model.Params.Method = SETTINGS["root_method"]
        model.Params.Seed = 0
        variables = model.addMVar(len(c), lb=0., ub=upper, vtype=types, obj=c)
        model.addMConstr(matrix, variables, "<", rhs)
        model.ObjCon = float(constant)
        variables.Start = np.zeros(len(c))
        setup_seconds = perf_counter() - setup
        started = perf_counter()
        model.optimize()
        solve_seconds = perf_counter() - started
        feasible = model.SolCount > 0
        solution = variables.X if feasible else None
        status = {2: "optimal", 3: "infeasible", 4: "inf_or_unbd", 5: "unbounded",
                  8: "node_limit", 9: "time_limit", 11: "interrupted"}.get(model.Status, f"status_{model.Status}")
        result = {"status": status, "status_code": int(model.Status), "solver": "Gurobi",
                  "solver_version": version("gurobipy"), "has_incumbent": feasible,
                  "objective": float(model.ObjVal) if feasible else None,
                  "lower_bound": float(model.ObjBound) if abs(model.ObjBound) < 1e29 else None,
                  "gap": float(model.MIPGap) if feasible else None,
                  "feasibility_tolerance": float(model.Params.FeasibilityTol),
                  "integer_tolerance": float(model.Params.IntFeasTol), "solver_seed": int(model.Params.Seed)}
        result.update(node_count=int(model.NodeCount), simplex_iterations=int(model.IterCount),
                      barrier_iterations=int(model.BarIterCount))
        model.dispose()
    else:
        raise ValueError(backend)
    result.update({"setup_seconds": setup_seconds, "solve_seconds": solve_seconds,
                   "n_variables": len(c), "n_constraints": len(rhs), "n_nonzeros": int(matrix.nnz),
                   "threads": SETTINGS["threads"],
                   "target_gap": SETTINGS["mip_gap"], "time_limit": time_limit,
                   "root_method": SETTINGS["root_method"],
                   "start": "all_external", "target_reached": feasible and result["gap"] <= SETTINGS["mip_gap"]})
    if feasible:
        violation = max(float(np.maximum(matrix @ solution - rhs, 0).max(initial=0)),
                        float(np.maximum(-solution, 0).max(initial=0)),
                        float(np.maximum(solution - upper, 0).max(initial=0)))
        result["primal_violation"] = violation
        result["objective_residual"] = abs(float(c @ solution + constant) - result["objective"])
        if violation > 5 * result["feasibility_tolerance"]:
            raise ArithmeticError(f"Solver solution fails matrix feasibility: {violation}")
    return solution, result


def solve_discrete(inst, method, n_train=200, repeat=0, backend="gurobi", time_limit=None):
    started = perf_counter()
    m, n = len(inst.sites), len(inst.xy)
    values = cost_values(inst, inst.xy)
    external, costs = values[:, 0], values[:, 1:].T
    q = scenarios(inst, n_train, "train", repeat).astype(float) if method.startswith("SAA") else inst.probabilities[None, :, :]
    s_count = len(q)
    # Only strictly dominated reduced-cost arcs are omitted. Avoided leftover
    # cost matters: pruning by c > a1 would be incorrect for fixed inventories.
    eligible = costs - external - inst.holding[:, None] < 0
    demand_s, demand_j, demand_p = np.nonzero(q)
    occurrence, i = np.nonzero(eligible[:, demand_j].T)
    s, j, p = demand_s[occurrence], demand_j[occurrence], demand_p[occurrence]
    count = len(i)
    stock_index = i * 2 + p
    col = np.arange(3 * m, 3 * m + count)
    c = np.r_[inst.fixed_costs, np.repeat(inst.holding, 2), (costs[i, j] - external[j] - inst.holding[i]) / s_count]
    # No useful stock can exceed maximum total demand of its own part. For
    # integer MVP stock, round that valid continuous bound upward.
    stock_bound = q.sum(axis=1).max(axis=0)
    if method.endswith("I"):
        stock_bound = np.ceil(stock_bound)
    stock_bound = np.tile(stock_bound, m)
    upper = np.r_[np.ones(m), stock_bound, q[s, j, p]]
    types = np.array(["B"] * m + (["I"] if method.endswith("I") else ["C"]) * (2 * m) + ["C"] * count)
    # demand rows, warehouse/scenario stock rows, warehouse opening links,
    # and per-arc opening links (valid also for fractional MVP allocations).
    stock_row, link_row, arc_row = s_count * n * 2, s_count * (n + m) * 2, s_count * (n + m) * 2 + 2*m
    rows = np.r_[s * n * 2 + j * 2 + p, stock_row + s * m * 2 + stock_index,
                 stock_row + np.arange(s_count * m * 2),
                 link_row + np.arange(m * 2), link_row + np.arange(m * 2),
                 arc_row + np.arange(count), arc_row + np.arange(count)]
    cols = np.r_[col, col, m + np.tile(np.arange(m * 2), s_count),
                 m + np.arange(m * 2), np.repeat(np.arange(m), 2), col, i]
    coefficients = np.r_[np.ones(count * 2), -np.ones(s_count * m * 2),
                         np.ones(m * 2), -stock_bound, np.ones(count), -q[s, j, p]]
    rhs = np.r_[q.ravel(), np.zeros(s_count * m * 2 + m * 2 + count)]
    matrix = sparse.coo_matrix((coefficients, (rows, cols)), shape=(len(rhs), len(c))).tocsr()
    constant = float(np.mean(np.einsum("snp,n->s", q, external)))
    pre = perf_counter() - started
    solution, solver = solve_matrix(c, matrix, rhs, upper, types, constant, backend, time_limit)
    result = {**solver, "method": method, "n_train": s_count,
              "training_repeat": repeat, "stock_upper_bounds": stock_bound.reshape(m, 2),
              "timing": {"preprocess_and_matrix": pre,
                  "model_setup": solver["setup_seconds"], "solve": solver["solve_seconds"],
                  "planning": perf_counter() - started}}
    if solution is None:
        result.update(inventory=None, active=None, planning_objective=None)
        return result
    raw = solution[m:3 * m].reshape(m, 2).copy()
    stock = np.maximum(raw, 0.)
    if method.endswith("I"):
        if np.any(np.abs(stock - np.rint(stock)) > solver["integer_tolerance"]):
            raise ArithmeticError("Noninteger solver inventory")
        stock = np.rint(stock)
    # Recheck all constraints after solver-tolerance corrections; do not erase
    # legal tiny positive continuous stocks, and do not add safety stock.
    normalized = solution.copy()
    normalized[m:3 * m] = stock.ravel()
    normalized[:m][stock.sum(axis=1) == 0] = 0
    residual = float(np.maximum(matrix @ normalized - rhs, 0).max(initial=0))
    if residual > 5 * solver["feasibility_tolerance"]:
        raise ArithmeticError("Normalized stock violates planning constraints")
    result.update({"inventory": stock, "active": np.flatnonzero(stock.sum(axis=1) > 0),
                   "raw_solver_inventory": raw, "solver_opened": solution[:m],
                   "normalization_residual": residual,
                   "planning_objective": float(c @ normalized + constant)})
    result["timing"]["planning"] = perf_counter() - started
    return result

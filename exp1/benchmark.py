"""Same-continuous-model MILP; feasible incumbents and actual solver gaps are retained."""
from time import perf_counter
import numpy as np
from scipy import sparse
from environment import SETTINGS
from geometry import ExactEvaluator, ordering_coefficients

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
        model.Params.Seed = 0
        model.Params.Method = SETTINGS["root_method"]
        variables = model.addMVar(len(c), lb=0., ub=upper, vtype=types, obj=c)
        model.addMConstr(matrix, variables, "<", rhs)
        model.ObjCon = constant
        variables.Start = np.zeros(len(c))
        setup_seconds = perf_counter() - setup
        started = perf_counter()
        model.optimize()
        solve_seconds = perf_counter() - started
        feasible = model.SolCount > 0
        solution = variables.X if feasible else None
        result = {"status": {2: "optimal", 3: "infeasible", 9: "time_limit", 11: "interrupted"}.get(model.Status, f"status_{model.Status}"),
                  "status_code": int(model.Status), "solver": "Gurobi",
                  "solver_version": ".".join(map(str, gp.gurobi.version())), "has_incumbent": feasible,
                  "objective": float(model.ObjVal) if feasible else None,
                  "lower_bound": float(model.ObjBound) if abs(model.ObjBound) < 1e29 else None,
                  "gap": float(model.MIPGap) if feasible else None,
                  "feasibility_tolerance": float(model.Params.FeasibilityTol),
                  "integer_tolerance": float(model.Params.IntFeasTol)}
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


def solve_continuous_milp(inst, backend="gurobi", time_limit=None):
    started = perf_counter()
    ev = ExactEvaluator(inst)
    q, h, order, pieces = ordering_coefficients(ev)
    m, k = len(inst.sites), len(q)
    # Eliminate u_0k = 1 - sum_i u_ik. Nonpositive savings arcs cannot improve
    # any open set, so external-dominated arcs can be omitted exactly.
    cell, warehouse = np.nonzero(h[:, 1:] < h[:, :1])
    count = len(cell)
    c = np.r_[inst.fixed_costs, h[cell, warehouse + 1] - h[cell, 0]]
    col = m + np.arange(count)
    rows = np.r_[cell, k + np.arange(count), k + np.arange(count)]
    cols = np.r_[col, col, warehouse]
    values = np.r_[np.ones(count * 2), -np.ones(count)]
    rhs = np.r_[np.ones(k), np.zeros(count)]
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(rhs), len(c))).tocsr()
    pre = perf_counter() - started
    solution, solver = solve_matrix(c, matrix, rhs, np.ones(len(c)), np.array(["B"] * m + ["C"] * count),
                                    float(h[:, 0].sum()), backend, time_limit)
    result = {**solver, "method": "MILP", "ordering_piece_count": pieces,
              "ordering_group_count": k, "coefficient_mass_by_part": q.sum(axis=0),
              "cost_affine_cells": ev.cell_count,
              "timing": {"subdivision_and_matrix": pre, "model_setup": solver["setup_seconds"],
                         "solve": solver["solve_seconds"], "planning": perf_counter() - started}}
    if solution is None:
        result.update(inventory=None, active=None, planning_objective=None)
    else:
        recovery_start = perf_counter()
        allowed = solution[:m] > .5
        available = np.r_[True, allowed]
        chosen = np.array([ranking[np.flatnonzero(available[ranking])[0]] for ranking in order])
        stock = np.column_stack([np.bincount(chosen, weights=q[:, p], minlength=m + 1)[1:] for p in range(2)])
        original = float(h[np.arange(k), chosen].sum() + inst.fixed_costs[stock.sum(axis=1) > 0].sum())
        recovered = ev.evaluate(np.zeros(m), allowed)
        if not np.allclose(stock, recovered.inventory, rtol=2e-8, atol=2e-9):
            raise ArithmeticError("Ordering-based recovery differs from regional integration")
        if original > solver["objective"] + 1e-7 * max(1., abs(original)):
            raise ArithmeticError("Canonical recovery exceeds the solver incumbent")
        result.update(inventory=stock, active=np.flatnonzero(stock.sum(axis=1) > 0),
                      planning_objective=original, recovery_objective_residual=original - solver["objective"])
        result["timing"]["recovery"] = perf_counter() - recovery_start
    result["timing"]["planning"] = perf_counter() - started
    return result, {"mass": q, "cost_integrals": h, "full_order": order}

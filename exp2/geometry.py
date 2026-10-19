"""Exact regional evaluation of the paper's adopted continuous demand.

Costs are convex max-affine functions, including external supply (option 0).
Each kernel is integrated separately and summed by linearity; this avoids a
global overlay of overlapping kernel supports without changing the integral.
Within a support, all relevant cost branches are commonly refined before the
minimum adjusted option is selected, with the paper's fixed tie order.

In s=(dx+dy)/sigma, t=(dx-dy)/sigma coordinates the kernel measure is
(35/32)^2 (1-s^2)^3 (1-t^2)^3 ds dt on [-1,1]^2. Green's theorem reduces its
zeroth and first moments to edge polynomials of degree at most 14. Eight-point
Gauss-Legendre evaluates those polynomials exactly in exact arithmetic (degree
15 exactness); it is not a numerical approximation of the area integral.
Floating point clipping and summation still have rounding errors.

Code provenance: the two-segment L1 cost constructor below is directly adapted
from Flexibility_Supply_Chains/Exp_1/l1_tv_uot_common.py. Convex half-plane
clipping and Green edge integration follow that file's
_clip_convex_polygon_to_halfplane, integrate_edge_gaussian, and
exact_polygon_integral. They are expressed in kernel-local coordinates and
compiled with Numba here; the separable primitive avoids expanding the same
polynomial in global coordinates. The common-refinement and subset reuse
implementation is adapted from the validated numerical-experiment geometry module. No fixed-inventory
dual iterations from the old experiments are used.
"""

from dataclasses import dataclass
import time

import numpy as np
from numba import njit


@dataclass
class Evaluation:
    inventory: np.ndarray
    local_cost: float
    external_cost: float
    outside_mass: np.ndarray
    weighted_objective: float
    option_costs: np.ndarray
    external_by_option: np.ndarray
    mass_corrections: dict = None


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(8)
_GAUSS_X = (_GAUSS_X + 1.0) / 2.0
_GAUSS_W = _GAUSS_W / 2.0
_KERNEL_CONSTANT = (35.0 / 32.0) ** 2
_GEOMETRY_TOL = 2e-12


def build_two_segment_l1_affine_family(anchor, breakpoint, intercept,
                                      slope_inner, slope_outer):
    """Original project's convex two-segment L1 cost, as affine branches.

    Directly reused from l1_tv_uot_common.py, with finite/nonnegative cost
    parameter checks and an array return. Its maximum equals the continuous
    two-segment distance cost; no fitted or approximated cost is introduced.
    """
    anchor = np.asarray(anchor, dtype=float).reshape(2)
    values = [breakpoint, intercept, slope_inner, slope_outer]
    if not np.isfinite(anchor).all() or not np.isfinite(values).all():
        raise ValueError("Cost parameters must be finite")
    if breakpoint < 0 or intercept < 0 or slope_inner < 0 or slope_outer < slope_inner:
        raise ValueError("Require breakpoint, intercept >= 0 and 0 <= inner <= outer slope")
    families = []
    outer_offset = (slope_inner - slope_outer) * breakpoint
    for slope, offset in ((slope_inner, 0.0), (slope_outer, outer_offset)):
        for sx, sy in ((1., 1.), (1., -1.), (-1., 1.), (-1., -1.)):
            families.append((slope * sx, slope * sy,
                             intercept + offset - slope * (sx * anchor[0] + sy * anchor[1])))
    return np.asarray(families)


@njit(cache=True)
def _clip(poly, a, b, c):
    """Clip a CCW convex polygon by a*s+b*t+c <= 0."""
    count = len(poly)
    if count < 3:
        return np.empty((0, 2))
    minimum = 1e300
    maximum = -1e300
    for k in range(count):
        d = a * poly[k, 0] + b * poly[k, 1] + c
        minimum = min(minimum, d)
        maximum = max(maximum, d)
    if maximum <= _GEOMETRY_TOL:
        return poly
    if minimum >= 0.0:
        return np.empty((0, 2))
    out = np.empty((count + 1, 2))
    used = 0
    last = count - 1
    previous_d = a * poly[last, 0] + b * poly[last, 1] + c
    for k in range(count):
        current_d = a * poly[k, 0] + b * poly[k, 1] + c
        previous_in = previous_d <= 0.0
        current_in = current_d <= 0.0
        if previous_in != current_in:
            fraction = previous_d / (previous_d - current_d)
            out[used] = poly[last] + fraction * (poly[k] - poly[last])
            used += 1
        if current_in:
            out[used] = poly[k]
            used += 1
        last = k
        previous_d = current_d
    return out[:used]


@njit(cache=True)
def _area(poly):
    value = 0.0
    for k in range(len(poly)):
        j = (k + 1) % len(poly)
        value += poly[k, 0] * poly[j, 1] - poly[j, 0] * poly[k, 1]
    return 0.5 * value


@njit(cache=True)
def polygon_moments(poly, order=3):
    """Return integrals of [1,s,t] against a normalized kernel on poly."""
    mass = first_s = first_t = 0.0
    if len(poly) < 3:
        return mass, first_s, first_t
    for k in range(len(poly)):
        j = (k + 1) % len(poly)
        ds = poly[j, 0] - poly[k, 0]
        dt = poly[j, 1] - poly[k, 1]
        if dt == 0.0:
            continue
        for q in range(8):
            s = poly[k, 0] + ds * _GAUSS_X[q]
            t = poly[k, 1] + dt * _GAUSS_X[q]
            s2 = s * s
            if order == 3:
                primitive0 = s * (1.0 + s2 * (-1.0 + s2 * (0.6 - s2 / 7.0)))
                primitive1 = s2 * (0.5 + s2 * (-0.75 + s2 * (0.5 - s2 / 8.0)))
                constant = _KERNEL_CONSTANT
            else:
                primitive0 = s - s * s2 / 3.
                primitive1 = s2 / 2. - s2 * s2 / 4.
                constant = (3. / 4.) ** 2
            weight = dt * _GAUSS_W[q] * (1.0 - t * t) ** order * constant
            mass += weight * primitive0
            first_s += weight * primitive1
            first_t += weight * primitive0 * t
    # Near-zero, thin polygons can have roundoff-level negative mass.
    if mass < 0.0 and mass > -1e-12:
        return 0.0, 0.0, 0.0
    return mass, first_s, first_t


@njit(cache=True)
def _cell_evaluate(poly, coefficients, penalties, allowed, order=3):
    """Demand fractions and unpenalized cost integrals on one affine cell."""
    count = len(coefficients)
    masses = np.zeros(count)
    costs = np.zeros(count)
    external = np.zeros(count)
    lows = np.empty(count)
    highs = np.empty(count)
    upper = 1e300
    for i in range(count):
        if not allowed[i]:
            lows[i] = highs[i] = 1e300
            continue
        low, high = 1e300, -1e300
        for v in range(len(poly)):
            value = (coefficients[i, 0] * poly[v, 0]
                     + coefficients[i, 1] * poly[v, 1]
                     + coefficients[i, 2] + penalties[i])
            low = min(low, value)
            high = max(high, value)
        lows[i], highs[i] = low, high
        upper = min(upper, high)
    candidates = np.empty(count, dtype=np.int64)
    n_candidates = 0
    for i in range(count):
        if allowed[i] and lows[i] <= upper + _GEOMETRY_TOL:
            candidates[n_candidates] = i
            n_candidates += 1
    for ii in range(n_candidates):
        i = candidates[ii]
        region = poly
        for jj in range(n_candidates):
            j = candidates[jj]
            if i == j:
                continue
            a = coefficients[i, 0] - coefficients[j, 0]
            b = coefficients[i, 1] - coefficients[j, 1]
            c = coefficients[i, 2] - coefficients[j, 2] + penalties[i] - penalties[j]
            if abs(a) + abs(b) + abs(c) <= _GEOMETRY_TOL:
                if j < i:
                    region = np.empty((0, 2))
                    break
                continue
            region = _clip(region, a, b, c)
            if len(region) < 3:
                break
        if len(region) >= 3:
            mass, ms, mt = polygon_moments(region, order)
            masses[i] = mass
            costs[i] = (coefficients[i, 0] * ms + coefficients[i, 1] * mt
                        + coefficients[i, 2] * mass)
            external[i] = (coefficients[0, 0] * ms + coefficients[0, 1] * mt
                           + coefficients[0, 2] * mass)
    return masses, costs, external


@njit(cache=True)
def _evaluate_cells(vertices, offsets, coefficients, kernel_ids, probabilities, penalties, allowed, order):
    option_count = coefficients.shape[1]
    part_count = probabilities.shape[1]
    masses = np.zeros((option_count, part_count))
    costs = np.zeros((option_count, part_count))
    external = np.zeros((option_count, part_count))
    for c in range(len(kernel_ids)):
        poly = vertices[offsets[c]:offsets[c + 1]]
        local_masses, local_costs, local_external = _cell_evaluate(poly, coefficients[c], penalties, allowed, order)
        n = kernel_ids[c]
        for p in range(part_count):
            for i in range(option_count):
                masses[i, p] += local_masses[i] * probabilities[n, p]
                costs[i, p] += local_costs[i] * probabilities[n, p]
                external[i, p] += local_external[i] * probabilities[n, p]
    return masses, costs, external


def _transform(branches, center, bandwidth, kernel="diamond3"):
    result = np.empty_like(branches, dtype=float)
    if kernel.startswith("diamond"):
        result[:, 0] = 0.5 * bandwidth * (branches[:, 0] + branches[:, 1])
        result[:, 1] = 0.5 * bandwidth * (branches[:, 0] - branches[:, 1])
    else:
        result[:, :2] = bandwidth * branches[:, :2]
    result[:, 2] = branches[:, :2] @ center + branches[:, 2]
    return result


@njit(cache=True)
def _branch_regions(poly, branches):
    """Active max-affine branches, retaining only full-dimensional pieces."""
    values = np.empty((len(poly), len(branches)))
    sums = np.zeros(len(branches))
    for v in range(len(poly)):
        for k in range(len(branches)):
            values[v, k] = (poly[v, 0] * branches[k, 0]
                            + poly[v, 1] * branches[k, 1] + branches[k, 2])
            sums[k] += values[v, k]
    winner = int(np.argmax(sums))
    constant = True
    for v in range(len(poly)):
        for k in range(len(branches)):
            if values[v, winner] < values[v, k] - _GEOMETRY_TOL:
                constant = False
    pieces = []
    if constant:
        pieces.append((poly.copy(), branches[winner].copy()))
        return pieces
    for i, branch in enumerate(branches):
        region = poly
        for j, other in enumerate(branches):
            if i == j:
                continue
            difference = other - branch
            if np.max(np.abs(difference)) <= _GEOMETRY_TOL:
                if j < i:
                    region = np.empty((0, 2))
                    break
                continue
            region = _clip(region, difference[0], difference[1], difference[2])
            if len(region) < 3:
                break
        if len(region) >= 3 and _area(region) > 1e-14:
            pieces.append((region.copy(), branch.copy()))
    return pieces


def clipped_kernel_support(center, bandwidth, domain_size, kernel="diamond3"):
    polygon = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    bounds = np.array([[-1., 0., 0.], [1., 0., -domain_size],
                       [0., -1., 0.], [0., 1., -domain_size]])
    for halfplane in _transform(bounds, center, bandwidth, kernel):
        polygon = _clip(polygon, *halfplane)
    return polygon


class ExactEvaluator:
    """Precomputed branch refinement; repeated weighted updates are fast.

    Penalties are nonnegative as in G(lambda,S). The continuous density is
    exactly the paper's normalized rhombic triweight mixture. The square
    domain clips crossing supports exactly; boundary mass is not normalized
    away. The original fixed opening cost is deliberately not included in
    Evaluation: callers charge the warehouses with positive final inventory.
    """

    source = "exact_green"

    def __init__(self, instance):
        started = time.perf_counter()
        self.instance = instance
        self.order = 1 if instance.kernel == "diamond1" else 3
        self.m = len(instance.sites)
        probabilities = np.asarray(instance.probabilities, dtype=float)
        self.probabilities = probabilities.reshape(len(instance.xy), -1)
        cost_branches = instance.cost_branches
        if len(cost_branches) != self.m + 1:
            raise ValueError("cost_branches must contain external option then all warehouses")
        if np.any(self.probabilities < 0) or not np.isfinite(self.probabilities).all():
            raise ValueError("Demand weights must be finite and nonnegative")
        sigma = float(instance.kernel_bandwidth)
        if sigma <= 0:
            raise ValueError("bandwidth must be positive")
        square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        polygons, costs, kernel_ids = [], [], []
        cost_branches = [np.asarray(b) for b in cost_branches]
        transformed_all = None
        # Batch the standard float64 families without changing branch order or
        # tolerances. Other family shapes/dtypes keep the scalar arithmetic.
        if all(b.shape == cost_branches[0].shape and b.dtype == np.dtype(float)
               for b in cost_branches):
            families = np.stack(cost_branches)
            transformed_all = np.empty((len(instance.xy), *families.shape))
            if instance.kernel.startswith("diamond"):
                transformed_all[..., 0] = 0.5 * sigma * (families[..., 0] + families[..., 1])
                transformed_all[..., 1] = 0.5 * sigma * (families[..., 0] - families[..., 1])
            else:
                transformed_all[..., :2] = sigma * families[..., :2]
            transformed_all[..., 2] = (
                np.matmul(families[None, ..., :2], instance.xy[:, None, :, None])[..., 0]
                + families[..., 2])
            values_all = (np.matmul(square, transformed_all[..., :2].swapaxes(-1, -2))
                          + transformed_all[..., 2][:, :, None, :])
            winners = values_all.mean(axis=2).argmax(axis=2)
            winning_values = np.take_along_axis(values_all, winners[:, :, None, None], axis=3)
            fixed_mask = np.all(winning_values >= values_all - _GEOMETRY_TOL, axis=(2, 3))
        self.kernel_mass = np.zeros(len(instance.xy))
        for n, center in enumerate(instance.xy):
            if not np.any(self.probabilities[n] > 0):
                continue
            support = clipped_kernel_support(center, sigma, instance.domain_size, instance.kernel)
            if len(support) < 3:
                continue
            self.kernel_mass[n] = polygon_moments(support, self.order)[0]
            if transformed_all is None:
                transformed = [_transform(b, center, sigma, instance.kernel) for b in cost_branches]
                fixed = np.empty((self.m + 1, 3))
                variable = []
                for option, branches in enumerate(transformed):
                    values = square @ branches[:, :2].T + branches[:, 2]
                    winner = int(np.argmax(values.mean(axis=0)))
                    if np.all(values[:, winner, None] >= values - _GEOMETRY_TOL):
                        fixed[option] = branches[winner]
                    else:
                        variable.append((option, branches))
            else:
                transformed = transformed_all[n]
                fixed = transformed[np.arange(self.m + 1), winners[n]].copy()
                variable = [(option, transformed[option])
                            for option in np.flatnonzero(~fixed_mask[n])]
            # Costs affine throughout a support do not need to be revisited
            # on every smaller cell. Only actual branch switches refine it.
            cells = [(support, [])]
            for option, branches in variable:
                new_cells = []
                for poly, active in cells:
                    for region, branch in _branch_regions(poly, branches):
                        new_cells.append((region, active + [branch]))
                cells = new_cells
            for poly, active in cells:
                polygons.append(poly)
                row = fixed.copy()
                for j, (option, _) in enumerate(variable):
                    row[option] = active[j]
                costs.append(row)
                kernel_ids.append(n)
        if not polygons:
            self.vertices = np.empty((0, 2))
            self.offsets = np.array([0], dtype=np.int64)
            self.coefficients = np.empty((0, self.m + 1, 3))
        else:
            self.vertices = np.ascontiguousarray(np.concatenate(polygons))
            self.offsets = np.r_[0, np.cumsum([len(poly) for poly in polygons])].astype(np.int64)
            self.coefficients = np.ascontiguousarray(costs, dtype=float)
        self.kernel_ids = np.asarray(kernel_ids, dtype=np.int64)
        self.cell_count = len(kernel_ids)
        self.total_mass = self.kernel_mass @ self.probabilities
        self.preprocess_seconds = time.perf_counter() - started

    def evaluate(self, penalties, allowed=None, part=None):
        penalties = np.asarray(penalties, dtype=float)
        if penalties.shape != (self.m,) or not np.isfinite(penalties).all():
            raise ValueError("penalties must be a finite vector of length M")
        if allowed is None:
            allowed = np.ones(self.m, dtype=bool)
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != (self.m,):
            raise ValueError("allowed must have length M")
        probabilities = self.probabilities if part is None else np.ascontiguousarray(self.probabilities[:, part:part + 1])
        masses, costs, external_by_option = _evaluate_cells(self.vertices, self.offsets, self.coefficients,
                                        self.kernel_ids, probabilities,
                                        np.r_[0., penalties], np.r_[True, allowed], self.order)
        if np.any(masses < -1e-10):
            raise ArithmeticError("Negative regional mass beyond floating-point tolerance")
        corrections = {"negative_mass_entries": int(np.count_nonzero(masses < 0)),
                       "negative_mass_sum": float(masses[masses < 0].sum()),
                       "rule": "only negative floating-point mass within 1e-10 is clipped; positive mass is retained"}
        inventory = np.maximum(masses[1:], 0.)
        outside = np.maximum(masses[0], 0.)
        expected = self.total_mass if part is None else self.total_mass[part:part + 1]
        if not np.allclose(inventory.sum(axis=0) + outside, expected, rtol=2e-9, atol=2e-10):
            raise ArithmeticError("Regional partition failed demand-mass conservation")
        local = float(costs[1:].sum())
        external = float(costs[0].sum())
        return Evaluation(inventory, local, external, outside,
                          local + external + float(penalties @ inventory.sum(axis=1)), costs, external_by_option, corrections)

def nominal_regions(inst, allowed, penalties=None):
    """Closed polygon representatives throughout Omega, including zero density."""
    ids = np.r_[0, np.flatnonzero(allowed) + 1]
    penalty = np.zeros(len(inst.sites) + 1) if penalties is None else np.r_[0., penalties]
    length = inst.domain_size
    cells = [(np.array([[0., 0.], [length, 0.], [length, length], [0., length]]), [])]
    branches = inst.cost_branches
    for i in ids:
        cells = [(piece, active + [coefficient])
                 for p, active in cells for piece, coefficient in _branch_regions(p, branches[i])]
    result = []
    for p, active in cells:
        for ii, i in enumerate(ids):
            region = p
            for jj, j in enumerate(ids):
                if i == j:
                    continue
                diff = active[ii] - active[jj]
                diff = diff.copy()
                diff[2] += penalty[i] - penalty[j]
                if np.max(np.abs(diff)) <= _GEOMETRY_TOL:
                    if j < i:
                        region = np.empty((0, 2))
                        break
                else:
                    region = _clip(region, *diff)
                if len(region) < 3:
                    break
            if len(region) >= 3 and _area(region) > 1e-12:
                result.append({"option": int(i), "vertices": region})
    if not np.isclose(sum(_area(r["vertices"]) for r in result), length ** 2, rtol=2e-9):
        raise ArithmeticError("Nominal polygons do not cover Omega")
    return result

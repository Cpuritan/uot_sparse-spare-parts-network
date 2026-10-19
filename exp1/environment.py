"""Frozen inputs for the 2026-10-17 plan; no outcomes enter generation."""
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "input"
OUT = ROOT / "output"
VERSION = "20261018-exp1"
KERNEL_SCALES = {"diamond3": 1., "diamond1": np.sqrt(5.) / 3., "square3": 1. / np.sqrt(2.)}
SCALES = {"S": (30., 16, (16, 14, 14, 12, 12, 12, 20)),
          "M": (45., 24, (40, 35, 35, 30, 30, 30, 50)),
          "L": (60., 32, (80, 70, 70, 60, 60, 60, 100))}
STREAMS = {"centers": 1, "equipment": 2, "warehouses": 3, "probability": 4,
           "fixed": 5, "cost": 6, "mc": 7, "train": 8, "test": 9,
           "initial": 10, "bootstrap": 11}
SETTINGS = {"version": VERSION, "seeds": list(range(42, 62)), "fixed_seed": 42,
            "parts": 2, "epsilon": 1., "probability_bounds": [[.15, .35], [.05, .25]],
            "mc_sizes": [1000, 10000, 100000], "trajectory_repeats": 5, "quality_repeats": 20,
            "kernels": ["diamond3", "diamond1", "square3"],
            "tolerance": 1e-6, "max_iterations": 200, "evaluation_gap": 1e-7,
            "mip_gap": 1e-3, "time_limit": 1800., "threads": 1, "root_method": 2}


def rng(seed, stream, repeat=0, part=None):
    entropy = [20261017, int(seed), STREAMS[stream], int(repeat)]
    if part is not None:
        entropy.append(int(part))
    return np.random.default_rng(np.random.SeedSequence(entropy))


def output_file(name):
    return OUT / name


def input_file(name):
    return DATA / name


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass
class Instance:
    scale: str
    seed: int
    domain_size: float
    xy: np.ndarray
    sites: np.ndarray
    probabilities: np.ndarray
    true_probabilities: np.ndarray
    fixed_costs: np.ndarray
    base: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    equipment_ids: np.ndarray
    warehouse_ids: np.ndarray
    kernel: str = "diamond3"
    bandwidth: float = 1.
    bias: str = "unbiased"

    @property
    def kernel_bandwidth(self):
        return self.bandwidth * KERNEL_SCALES[self.kernel]

    @property
    def Q(self):
        return float(self.probabilities.sum())

    @property
    def part_q(self):
        return self.probabilities.sum(axis=0)

    @property
    def holding(self):
        return np.full(len(self.sites), 16.)

    @property
    def cost_branches(self):
        from geometry import build_two_segment_l1_affine_family as branches
        return [branches([self.domain_size / 2] * 2, 10., 68., .9, 1.8)] + [
            branches(y, 10., b, beta, beta + gamma)
            for y, b, beta, gamma in zip(self.sites, self.base, self.beta, self.gamma)]


def _locations(generator, count, centers, center=None, sd=None):
    points = []
    while len(points) < count:
        x = generator.uniform(.05, .95, 2) if center is None else generator.normal(center, sd, 2)
        if np.all((x >= .05) & (x <= .95)) and (
                center is not None or np.all(np.linalg.norm(x - centers, axis=1) >= .15)):
            points.append(x)
    return np.array(points)


@lru_cache(maxsize=None)
def master(seed):
    path = input_file(f"master_{seed}.npz")
    centers = np.array([[.20, .22], [.50, .18], [.80, .26], [.22, .73], [.53, .78], [.80, .68]]) + rng(seed, "centers").uniform(-.03, .03, (6, 2))
    rd, rw = rng(seed, "equipment"), rng(seed, "warehouses")
    equipment = np.concatenate([_locations(rd, n, centers, centers[g] if g < 6 else None, .06)
                                for g, n in enumerate(SCALES["L"][2])])
    sites = np.concatenate([_locations(rw, 4 if g < 6 else 8, centers, centers[g] if g < 6 else None, .08)
                            for g in range(7)])
    result = dict(centers=centers, xy=equipment, sites=sites,
                  p=np.column_stack([rng(seed, "probability", part).uniform(lower, upper, 500)
                                     for part, (lower, upper) in enumerate(SETTINGS["probability_bounds"])]),
                  fixed=rng(seed, "fixed").integers(120, 181, 32),
                  coefficients=rng(seed, "cost").uniform(.9, 1.1, (32, 3)) * [40., .9, .9])
    if path.exists():
        with np.load(path) as old:
            if set(old.files) != set(result) or any(not np.array_equal(old[k], v) for k, v in result.items()):
                raise ValueError(f"Frozen input differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **result)
        write_json(path.with_suffix(".json"), {"seed": seed, "version": VERSION,
                   "probability_bounds": SETTINGS["probability_bounds"],
                   "stream_ids": STREAMS, "sha256": {k: digest(v) for k, v in result.items()}})
    return result


def instance(scale, seed, kernel="diamond3", bandwidth_sigma=1., bias="unbiased"):
    pool = master(seed)
    length, m, counts = SCALES[scale]
    ids = np.concatenate([np.arange(start, start + count)
                          for start, count in zip(np.r_[0, np.cumsum(SCALES["L"][2])[:-1]], counts)])
    wid = np.concatenate([np.arange(4*g, 4*g + m//8) for g in range(6)] + [np.arange(24, 24 + m//4)])
    xy, p = pool["xy"][ids] * length, pool["p"][ids]
    predicted = p.copy()
    if bias == "total_low":
        predicted *= .8
    elif bias == "total_high":
        predicted *= 1.2
    elif bias in ("spatial_plus", "spatial_minus"):
        delta = .25 if bias == "spatial_plus" else -.25
        predicted *= (1 + delta * np.where(xy[:, 0] <= length / 2, 1, -1))[:, None]
        predicted *= p.sum(axis=0) / predicted.sum(axis=0)
    elif bias != "unbiased":
        raise ValueError(bias)
    if kernel not in SETTINGS["kernels"] or bandwidth_sigma <= 0 or np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("Invalid density or forecast")
    support_radius = bandwidth_sigma * KERNEL_SCALES[kernel]
    if np.any(xy < support_radius) or np.any(xy > length - support_radius):
        raise ValueError("Kernel supports must remain inside Omega")
    b, beta, gamma = pool["coefficients"][wid].T
    return Instance(scale, seed, length, xy, pool["sites"][wid] * length, predicted, p,
                    pool["fixed"][wid].astype(float), b, beta, gamma, ids, wid,
                    kernel, float(bandwidth_sigma), bias)



def initial_inventory(inst, name="default", repeat=0):
    m = len(inst.sites)
    if name == "random":
        v = rng(inst.seed, "initial", repeat).uniform(.1, 1., (32, 2))[inst.warehouse_ids]
        return inst.part_q * v / v.sum(axis=0)
    factors = {"low": .5, "default": 1., "high": 2.}
    return np.tile(factors[name] * inst.part_q / m, (m, 1))


def cost_values(inst, points):
    d = np.abs(np.asarray(points)[:, None, :] - inst.sites[None, :, :]).sum(axis=2)
    local = inst.base + inst.beta * d + inst.gamma * np.maximum(d - 10., 0.)
    d0 = np.abs(np.asarray(points) - inst.domain_size / 2).sum(axis=1)
    external = 68. + .9 * d0 + .9 * np.maximum(d0 - 10., 0.)
    return np.column_stack((external, local))


def continuous_samples(inst, count, repeat, part):
    # Generate one row of independent uniforms per sample: requesting a shorter
    # prefix cannot change the beta draws or device indices of that prefix.
    from scipy.special import betaincinv
    u = rng(inst.seed, "mc", repeat, part).random((count, 3))
    ids = np.searchsorted(np.cumsum(inst.probabilities[:, part]) / inst.part_q[part], u[:, 0], side="right")
    order = 1 if inst.kernel == "diamond1" else 3
    s, t = (2 * betaincinv(order + 1, order + 1, u[:, 1:]) - 1).T
    offset = np.column_stack(((s + t) / 2, (s - t) / 2)) if inst.kernel.startswith("diamond") else np.column_stack((s, t))
    return inst.xy[ids] + inst.kernel_bandwidth * offset

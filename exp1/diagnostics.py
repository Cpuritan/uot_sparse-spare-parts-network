"""Instance-level path discrepancies; five repeats form one observation."""
from time import perf_counter
import numpy as np
from algorithm import MonteCarloEvaluator
from environment import SETTINGS, digest


def mass_history(record, part_q):
    history = np.asarray(record["inventory_history"], float)
    if history.shape[0] != record["iterations"] + 1 or not np.isfinite(history).all():
        raise ValueError("Incomplete or nonfinite pre-recovery trajectory")
    np.testing.assert_allclose(history[-1], record["pre_recovery_inventory"], atol=2e-9)
    outside = part_q - history.sum(axis=1)
    if history.min() < -2e-9 or outside.min() < -2e-9 * max(1., part_q.sum()):
        raise ValueError("Trajectory violates nonnegative demand balance")
    np.testing.assert_allclose(outside[1:], [step["external_mass"] for step in record["trace"]], atol=2e-9)
    return np.concatenate((outside[:, None, :], history), axis=1)


def align(history, horizon):
    if len(history) - 1 > horizon:
        raise ValueError("Alignment must not truncate a trajectory")
    return history[np.minimum(np.arange(horizon + 1), len(history) - 1)]


def path_metrics(reference, repetitions, total_demand):
    mean_path = repetitions.mean(axis=0)
    squared = np.square(repetitions - reference).sum(axis=(-2, -1))
    repeat_deviation = np.sqrt(squared.mean(axis=1)) / total_demand
    deviation = float(np.sqrt(squared.mean()) / total_demand)
    dispersion = float(np.sqrt(np.square(repetitions - mean_path).sum(axis=(-2, -1)).mean()) / total_demand)
    mean_deviation = float(np.sqrt(np.square(mean_path - reference).sum(axis=(-2, -1)).mean()) / total_demand)
    np.testing.assert_allclose(deviation ** 2, dispersion ** 2 + mean_deviation ** 2, atol=2e-15, rtol=2e-12)
    np.testing.assert_allclose(deviation ** 2, np.mean(repeat_deviation ** 2), atol=2e-15, rtol=2e-12)
    return deviation, dispersion, mean_deviation, repeat_deviation


def trajectory_records(records):
    return [record for record in records.values() if record["method"] != "MILP"
            and (record["budget"] == 0 or record["repeat"] < SETTINGS["trajectory_repeats"])]


def diagnose(inst, records):
    started = perf_counter()
    selected = trajectory_records(records)
    expected_count = 1 + len(SETTINGS["mc_sizes"]) * SETTINGS["trajectory_repeats"]
    if len(selected) != expected_count:
        raise ValueError("A trajectory instance requires one analytical and fifteen MC paths")
    reference = next(record for record in selected if record["budget"] == 0)
    horizon = max(record["iterations"] for record in selected)
    reference_mass = mass_history(reference, inst.part_q)
    aligned_reference = align(reference_mass, horizon)
    weights = -np.asarray(reference["threshold_history"])
    np.testing.assert_allclose(weights, inst.fixed_costs / (np.asarray(reference["inventory_history"])[:-1].sum(axis=2) + SETTINGS["epsilon"]))
    arrays = {"reference_mass": reference_mass, "aligned_reference": aligned_reference}
    instance_rows, repeat_rows, provenance = [], [], {}
    samplers = {}
    for repeat in range(SETTINGS["trajectory_repeats"]):
        previous = None
        for budget in SETTINGS["mc_sizes"]:
            sampler = MonteCarloEvaluator(inst, budget, repeat)
            if previous is not None:
                for part in range(2):
                    np.testing.assert_array_equal(previous.points[part], sampler.points[part][:previous.counts[part]])
            previous = sampler
            samplers[budget, repeat] = sampler
            provenance[f"{budget}/{repeat}"] = {"point_hashes": [digest(points) for points in sampler.points],
                "counts": sampler.counts, "streams": [[20261017, inst.seed, 7, repeat, part] for part in range(2)]}
    for budget in SETTINGS["mc_sizes"]:
        group = sorted([record for record in selected if record["budget"] == budget], key=lambda record: record["repeat"])
        assert [record["repeat"] for record in group] == list(range(SETTINGS["trajectory_repeats"]))
        aligned = np.stack([align(mass_history(record, inst.part_q), horizon) for record in group])
        np.testing.assert_allclose(aligned[:, 0], np.broadcast_to(reference_mass[0], aligned[:, 0].shape), atol=1e-12)
        replay = []
        for record in group:
            sampler = samplers[budget, record["repeat"]]
            assert sampler.counts == record["sample_sizes_by_part"]
            replay.append([np.vstack((update.outside_mass, update.inventory)) for penalties in weights
                           for update in [sampler.evaluate(penalties)]])
        replay = np.asarray(replay)
        errors = np.sqrt(np.square(replay - reference_mass[1:]).sum(axis=(-2, -1)).mean(axis=1)) / inst.Q
        deviation, dispersion, mean_deviation, per_repeat = path_metrics(aligned_reference, aligned, inst.Q)
        common = {"scale": inst.scale, "seed": inst.seed, "budget": budget, "horizon": horizon,
                  "reference_iterations": reference["iterations"], "Q_total": inst.Q, "n_repeats": len(group)}
        instance_rows.append({**common, "D": deviation, "V": dispersion, "mean_path_deviation": mean_deviation,
                              "E": float(np.sqrt(np.mean(errors ** 2)))})
        for position, record in enumerate(group):
            repeat_rows.append({**common, "repeat": record["repeat"], "run_id": record["run_id"],
                "D_repeat": float(per_repeat[position]), "E_repeat": float(errors[position]),
                "terminal_D_repeat": float(np.linalg.norm(aligned[position, -1] - aligned_reference[-1]) / inst.Q),
                "iterations": record["iterations"], "status": record["status"],
                "extension_steps": horizon - record["iterations"]})
        for part in range(2):
            assert len({provenance[f"{budget}/{repeat}"]["point_hashes"][part]
                        for repeat in range(len(group))}) == len(group)
        arrays[f"aligned_{budget}"], arrays[f"replay_{budget}"] = aligned, replay
    return {"instance_rows": instance_rows, "repeat_rows": repeat_rows, "provenance": provenance,
            "horizon": horizon, "seconds": perf_counter() - started}, arrays

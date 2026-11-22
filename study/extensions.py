"""Optional E5 finite-difference interactions and E6 measured allocations."""
from itertools import combinations
import random
import time

import numpy as np

from .allocation import cost, greedy, matched_random
from .evaluation import paired_interval
from .experiments import quality
from .io import atomic_json, digest
from .quantization import rtn, perturb, fingerprint, payload, error_terms


def select_pairs(rows, count, seed):
    rows = sorted(rows, key=lambda r: r["trial_key"])
    median = np.median([r["delta_nll"] for r in rows])
    strata = {}
    for a, b in combinations(rows, 2):
        if a["module_name"] == b["module_name"]:
            if a["col_start"] < b["col_end"] and b["col_start"] < a["col_end"]:
                continue
            category = "same_matrix"
        elif a["layer_idx"] == b["layer_idx"]:
            category = "same_layer"
        else:
            category = "cross_layer"
        levels = sorted(["high" if r["delta_nll"] >= median else "low" for r in (a, b)])
        sensitivity = "/".join(levels)
        strata.setdefault((category, sensitivity), []).append((a, b, category, sensitivity))
    rng = random.Random(seed)
    for pool in strata.values():
        rng.shuffle(pool)
    # Round-robin stratified random sampling covers categories when available.
    selected = []
    while len(selected) < count and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < count:
                selected.append(strata[key].pop())
    return selected


def e5(study):
    cfg = study.config["experiments"]["e5"]
    individual = [r for r in study.store("e3").rows() if r["bit_width"] == cfg["bits"]]
    if len(individual) < 2:
        raise ValueError("E5 needs measured E3 chunk sensitivities")
    if any(r["protocol_id"] != study.protocol_id for r in individual):
        raise ValueError("Individual and joint measurements must share the protocol")
    pairs = select_pairs(individual, cfg["pairs"], study.config["seed"])
    atomic_json(study.out / "e5_selection.json", {
        "rule": "Seeded random pairs, round-robin over location × above/below global median individual W2 sensitivity",
        "seed": study.config["seed"], "requested": cfg["pairs"], "selected": len(pairs),
        "pairs": [{"i": a["trial_key"], "j": b["trial_key"], "location": c, "sensitivity": d} for a, b, c, d in pairs]})
    store, baseline = study.store("e5"), study.baseline()
    for a, b, category, sensitivity in pairs:
        key = f'{a["trial_key"]}+{b["trial_key"]}'
        if store.get(key):
            continue
        clock = time.perf_counter()
        regions = []
        for row in (a, b):
            target = study.by_name[row["module_name"]]
            candidate = rtn(target.module.weight, cfg["bits"], study.config["group_size"])[:, row["col_start"]:row["col_end"]]
            if fingerprint(candidate) != row["candidate_sha256"]:
                raise ValueError("Candidate differs from individual perturbation")
            regions.append((target, row["col_start"], row["col_end"], candidate))
        with perturb(regions, group_size=study.config["group_size"]):
            measured = study.evaluate()
        additive = a["delta_nll"] + b["delta_nll"]
        observed = measured["nll"] - baseline["nll"]
        predicted_eval = {"sequence_loss_sums": (np.array(a["validation_evaluation"]["sequence_loss_sums"])
                          + np.array(b["validation_evaluation"]["sequence_loss_sums"])
                          - np.array(baseline["sequence_loss_sums"])).tolist(),
                          "sequence_token_counts": baseline["sequence_token_counts"]}
        store.add(key, {**study.common(), **quality(study, measured, baseline),
                       "region_i": a["trial_key"], "region_j": b["trial_key"],
                       "pair_type": category, "sensitivity_pair": sensitivity, "bit_width": cfg["bits"],
                       "delta_i": a["delta_nll"], "delta_j": b["delta_nll"],
                       "predicted_additive_delta": additive, "observed_joint_delta": observed,
                       "interaction_residual": observed-additive,
                       "interaction_ci": paired_interval(measured, predicted_eval, study.config["seed"]),
                       "runtime": time.perf_counter()-clock})
        study.verify()
        store.export()
    return store.export()


def e6(study):
    cfg = study.config["experiments"]["e6"]
    rows = study.store("e2").rows()
    sizes = {t.name: t.n_parameters for t in study.targets}
    sensitivities = {}
    for row in rows:
        if row["protocol_id"] != study.protocol_id:
            raise ValueError("E6 must use E2 measurements from the identical validation protocol")
        if row["bit_width"] in cfg["bits"]:
            sensitivities.setdefault(row["module_name"], {})[row["bit_width"]] = row["delta_nll"]
    plans = []
    for budget in cfg["budgets"]:
        plan = greedy(sensitivities, sizes, cfg["bits"], budget)
        plans.append(("sensitivity_greedy", budget, None, plan))
        for seed in cfg["random_seeds"]:
            plans.append(("random_matched", budget, seed, matched_random(plan, sizes, seed)))
    for bits in cfg["bits"]:
        plans.append(("uniform", float(bits), None, {name: bits for name in sizes}))
    store, baseline, test_baseline = study.store("e6"), study.baseline(), study.baseline("test")
    for strategy, budget, seed, plan in plans:
        key = f"{strategy}:{budget}:{seed}:{digest(plan)}"
        if store.get(key):
            continue
        clock = time.perf_counter()
        plan_path = study.out / "plans" / f"{digest(key)}.json"
        atomic_json(plan_path, {"strategy": strategy, "target_bpw": budget, "seed": seed,
                              "plan": plan, "payload_bits": cost(plan, sizes),
                              "selection_split": "validation", "protocol_id": study.protocol_id})
        # CPU candidates bound GPU memory; all use the common RTN operator.
        regions, numerator, denominator = [], 0., 0.
        for target in study.targets:
            candidate = rtn(target.module.weight, plan[target.name], study.config["group_size"])
            a, b = error_terms(target.module.weight, candidate)
            numerator += a
            denominator += b
            regions.append((target, 0, target.module.weight.shape[1], candidate.cpu()))
        with perturb(regions, group_size=study.config["group_size"]):
            measured = study.evaluate()
            test = study.evaluate("test")
        study.verify()
        realized = cost(plan, sizes)/sum(sizes.values())
        store.add(key, {**study.common(), **quality(study, measured, baseline),
                       "strategy": strategy, "target_bpw": budget, "realized_bpw": realized,
                       "budget_deviation": realized-budget, "random_seed": seed,
                       "plan_file": str(plan_path.relative_to(study.out)), "allocation": plan,
                       "normalized_weight_mse": numerator/(denominator+1e-12),
                       "test_nll": test["nll"], "test_ppl": test["ppl"], "test_evaluation": test,
                       "test_delta_nll": test["nll"]-test_baseline["nll"],
                       **payload(study.targets, plan, group_size=study.config["group_size"]),
                       **{f"n_matrices_w{b}": sum(v==b for v in plan.values()) for b in cfg["bits"]},
                       "runtime": time.perf_counter()-clock})
        store.export()
    return store.export()

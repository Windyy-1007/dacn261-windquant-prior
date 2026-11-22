"""Validation-only matrix allocation; no learned policy or training."""
import random


def cost(plan, sizes):
    return sum(sizes[name] * bits for name, bits in plan.items())


def greedy(sensitivities, sizes, actions, target_bpw):
    actions = sorted(set(actions))
    if target_bpw < actions[0]:
        raise ValueError("Budget is below the minimum precision")
    if set(sensitivities) != set(sizes):
        raise ValueError("Sensitivity data must cover every matrix")
    for name in sizes:
        if set(sensitivities[name]) != set(actions):
            raise ValueError(f"Missing candidate bit measurements: {name}")
    budget = int(target_bpw * sum(sizes.values()))
    plan = {name: actions[0] for name in sizes}
    while True:
        spent = cost(plan, sizes)
        upgrades = []
        for name, current in plan.items():
            for higher in actions:
                extra = sizes[name] * (higher-current)
                if higher <= current or spent + extra > budget:
                    continue
                benefit = sensitivities[name][current] - sensitivities[name][higher]
                if benefit > 0:
                    upgrades.append((benefit/extra, name, higher))
        if not upgrades:
            break
        _, name, higher = max(upgrades, key=lambda x: (x[0], x[1], x[2]))
        plan[name] = higher
    return plan


def matched_random(plan, sizes, seed):
    """Randomize within equal-size classes, preserving the EXACT code payload.

    This is a conditional random baseline, not a uniform draw over all feasible
    allocations. Preserve precision histograms within each parameter-size class.
    """
    rng = random.Random(seed)
    result = dict(plan)
    for size in sorted(set(sizes.values())):
        names = sorted(name for name in plan if sizes[name] == size)
        bits = [plan[name] for name in names]
        rng.shuffle(bits)
        result.update(zip(names, bits))
    if cost(result, sizes) != cost(plan, sizes):
        raise RuntimeError("Matched randomization changed the bit budget")
    return result

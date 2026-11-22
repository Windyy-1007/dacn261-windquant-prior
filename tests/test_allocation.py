import pytest

from study.allocation import greedy, matched_random, cost
from study.extensions import select_pairs


def test_greedy_budget_and_direct_upgrade_when_intermediate_is_worse():
    sizes = {"a": 10, "b": 20}
    losses = {"a": {2: 1., 3: 2., 4: 0.}, "b": {2: .4, 3: .3, 4: .2}}
    plan = greedy(losses, sizes, [2, 3, 4], 3.)
    assert plan == {"a": 4, "b": 2}
    assert cost(plan, sizes) <= 90
    with pytest.raises(ValueError):
        greedy({"a": {2: 1}}, sizes, [2, 3, 4], 3.)


def test_random_exact_matching_for_unequal_matrices():
    sizes = {"a": 5, "b": 5, "c": 10, "d": 10, "e": 20}
    plan = dict(zip(sizes, [2, 4, 2, 3, 4]))
    for seed in range(10):
        randomized = matched_random(plan, sizes, seed)
        assert cost(randomized, sizes) == cost(plan, sizes)
        assert randomized == matched_random(plan, sizes, seed)


def test_pair_sampling_distinct_reproducible_and_categories():
    rows = [{"trial_key": f"{layer}:{m}:{c}", "module_name": f"{layer}:{m}", "layer_idx": layer,
             "col_start": c*256, "col_end": (c+1)*256, "delta_nll": (layer+m+c)/100}
            for layer in range(3) for m in range(2) for c in range(4)]
    pairs = select_pairs(rows, 24, 42)
    assert pairs == select_pairs(rows, 24, 42)
    assert len({(a["trial_key"], b["trial_key"]) for a,b,_,_ in pairs}) == 24
    assert {category for _,_,category,_ in pairs} == {"same_matrix", "same_layer", "cross_layer"}
    assert {sensitivity for _,_,_,sensitivity in pairs} == {"high/high", "high/low", "low/low"}

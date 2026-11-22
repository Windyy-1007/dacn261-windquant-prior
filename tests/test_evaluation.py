from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from study.evaluation import windows, evaluate, paired_interval
from study.io import Results
from study.activations import collect, output_error
from study.quantization import discover
from test_quantization import tiny_model


class Markov(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor([[2., 0., -1.], [0., 1., 2.], [-2., 2., 0.]]))

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(logits=self.table[input_ids])


@pytest.mark.parametrize("length", [2, 7, 8, 9, 16, 19])
@pytest.mark.parametrize("stride", [1, 3, 7])
def test_each_target_once_with_correct_previous_token(length, stride):
    tokens = np.arange(length)
    recovered = []
    for ids, start in windows(tokens, 8, stride):
        recovered.extend(ids[start:])
    assert recovered == list(range(1, length))


def test_token_weighting_padding_and_causal_shift():
    model = Markov()
    tokens = np.array([0, 1, 2, 2, 1, 0, 0, 2, 1, 0, 2])
    result = evaluate(model, tokens, context=5, stride=3, batch_size=2, pad_id=2)
    expected = torch.nn.functional.cross_entropy(model.table[tokens[:-1]], torch.tensor(tokens[1:])).item()
    assert result["nll"] == pytest.approx(expected, rel=1e-6)
    assert result["ppl"] == pytest.approx(math.exp(expected), rel=1e-6)
    assert result["evaluated_tokens"] == 10
    other = evaluate(model, tokens, context=5, stride=3, batch_size=1)
    assert result["nll"] == other["nll"]
    assert paired_interval(result, result, block_windows=1) == [0., 0.]


def test_atomic_resume_and_torn_write(tmp_path):
    results = Results(tmp_path, "e2")
    results.add("trial", {"delta_nll": .1})
    results.filename("unfinished").with_suffix(".json.tmp").write_text('{"truncated":')
    assert results.get("unfinished") is None
    assert Results(tmp_path, "e2").get("trial")["delta_nll"] == .1
    assert len(results.export()) == 1


def test_activation_cap_hook_cleanup_and_chunk_dimensions():
    model = tiny_model()
    targets = discover(model)
    cache, meta = collect(model, targets, np.array([[1, 2, 3], [4, 5, 6]]), memory_mb=.01)
    assert meta["cached_bytes"] <= meta["budget_bytes"]
    assert all(not t.module._forward_pre_hooks for t in targets)
    w = torch.randn(3, 8)
    x = torch.randn(5, 8)
    q = w[:, 4:8] + .1
    y = x[:, 4:8] @ w[:, 4:8].T
    expected = ((x[:, 4:8] @ (w[:, 4:8]-q).T)**2).sum() / (y*y).sum()
    assert output_error(x, w, q, 4, 8) == pytest.approx(expected.item(), rel=1e-6)


def test_hook_cleanup_after_failed_forward():
    model = tiny_model()
    targets = discover(model)
    with pytest.raises((IndexError, RuntimeError)):
        collect(model, targets, np.array([[9999, 2, 3]]), memory_mb=.01)
    assert all(not t.module._forward_pre_hooks for t in targets)

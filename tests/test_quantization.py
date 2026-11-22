import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from study.quantization import rtn, discover, perturb, Target, payload


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_reference_scalar_equations_and_tail(bits):
    w = torch.tensor([[-3., -1., 0., 2., 9.], [1., 1., 1., 1., 1.]])
    result = rtn(w, bits, 4)
    expected = torch.empty_like(w)
    for row in range(len(w)):
        for start in range(0, w.shape[1], 4):
            x = w[row, start:start+4]
            lo, hi = min(float(x.min()), 0), max(float(x.max()), 0)
            scale = (hi-lo) / (2**bits - 1) or 1
            zero = max(0, min(2**bits - 1, round(-lo/scale)))
            expected[row, start:start+4] = (torch.round(x/scale).add(zero).clamp(0, 2**bits-1)-zero)*scale
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert torch.equal(result, rtn(w, bits, 4))
    assert torch.equal(rtn(torch.zeros(2, 5), bits, 4), torch.zeros(2, 5))


def tiny_model():
    torch.manual_seed(7)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                      head_dim=8)).eval()


def test_exception_restores_region_and_logits_and_excludes_other_parameters():
    model = tiny_model()
    targets = discover(model)
    assert len(targets) == 14
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    target = targets[0]
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits = model(ids).logits.clone()
    candidate = rtn(target.module.weight, 2, 4)[:, 4:8]
    with pytest.raises(RuntimeError, match="intentional"):
        with perturb([(target, 4, 8, candidate)], group_size=4):
            for name, weight in model.state_dict().items():
                if name == target.name + ".weight":
                    assert torch.equal(weight[:, :4], before[name][:, :4])
                    assert torch.equal(weight[:, 8:], before[name][:, 8:])
                    assert not torch.equal(weight[:, 4:8], before[name][:, 4:8])
                else:
                    assert torch.equal(weight, before[name])
            raise RuntimeError("intentional")
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, before[name])
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, logits, rtol=0, atol=0)


def test_joint_regions_and_boundary_rejection():
    target = discover(tiny_model())[0]
    w = target.module.weight.detach().clone()
    candidate = rtn(w, 2, 4)
    with perturb([(target, 0, 4, candidate[:, :4]), (target, 4, 8, candidate[:, 4:8])], group_size=4):
        assert torch.equal(target.module.weight[:, :8], candidate[:, :8])
    assert torch.equal(target.module.weight, w)
    for regions in [[(target, 1, 4, None)], [(target, 0, 8, None), (target, 4, 12, None)]]:
        with pytest.raises(ValueError):
            with perturb(regions, bits=2, group_size=4):
                pass
        assert torch.equal(target.module.weight, w)


def test_group_aligned_chunk_matches_full_operator():
    w = torch.randn(4, 17)
    assert torch.equal(rtn(w, 2, 4)[:, 8:16], rtn(w[:, 8:16], 2, 4))
    assert torch.equal(rtn(w, 2, 4)[:, 16:], rtn(w[:, 16:], 2, 4))
    t = Target("x", 0, "q_proj", torch.nn.Linear(17, 4, bias=False))
    result = payload([t], {"x": 2}, group_size=4)
    assert result["payload_bits"] == 4*17*2
    assert result["metadata_bits"] == 4*5*40

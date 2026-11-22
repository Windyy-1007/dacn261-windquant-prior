import math
import torch


@torch.inference_mode()
def collect(model, targets, samples, memory_mb=256):
    """Evenly sample input rows from every calibration example, within a hard cap."""
    budget = int(memory_mb * 1024 * 1024)
    dtype_bytes = next(model.parameters()).element_size()
    per_target = budget // len(targets)
    buffers, quotas = {}, {}
    for target in targets:
        cols = target.module.weight.shape[1]
        quota = per_target // (cols * dtype_bytes)
        if quota < len(samples):
            raise ValueError("Activation budget must allow at least one row per calibration sample per target")
        quotas[target.name] = quota
        buffers[target.name] = []
    handles = []
    sample_index = 0
    def hook_for(name):
        def hook(module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            q = quotas[name] // len(samples) + int(sample_index < quotas[name] % len(samples))
            q = min(q, len(x))
            # Midpoints cover the entire sequence without privileging its prefix.
            idx = ((torch.arange(q, device=x.device) + .5) * len(x) / q).long()
            buffers[name].append(x[idx].cpu())
        return hook
    try:
        for target in targets:
            handles.append(target.module.register_forward_pre_hook(hook_for(target.name)))
        device = next(model.parameters()).device
        for sample_index, sample in enumerate(samples):
            ids = torch.as_tensor(sample, device=device).unsqueeze(0)
            # Backbone outputs are enough; avoid allocating vocabulary logits.
            model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    cache = {name: torch.cat(xs) for name, xs in buffers.items()}
    actual = sum(x.numel() * x.element_size() for x in cache.values())
    if actual > budget:
        raise RuntimeError("Activation memory budget exceeded")
    return cache, {"cached_bytes": actual, "budget_bytes": budget,
                   "sampling": "Evenly spaced token midpoints; quota distributed over all calibration examples",
                   "rows_by_module": {name: len(x) for name, x in cache.items()}}


@torch.inference_mode()
def output_error(activation, weight, candidate, start=0, end=None):
    end = end if end is not None else weight.shape[1]
    original = weight[:, start:end].float()
    delta = original - candidate.float()
    numerator, denominator = 0.0, 0.0
    for x in activation.split(128):
        x = x[:, start:end].to(device=weight.device, dtype=torch.float32)
        y = x @ original.T
        error = x @ delta.T
        numerator += float(error.square().sum(dtype=torch.float64))
        denominator += float(y.square().sum(dtype=torch.float64))
    return numerator / (denominator + 1e-12)

"""The single deterministic quantization operator used by every experiment."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Target:
    name: str
    layer: int
    projection: str
    module: torch.nn.Module

    @property
    def n_parameters(self):
        return self.module.weight.numel()


def discover(model):
    """Discover projections and their containing ModuleList; no fixed layer count."""
    targets = []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        parts = name.split(".")
        if parts[-1] not in PROJECTIONS or not isinstance(module, torch.nn.Linear):
            continue
        ancestors = [(int(parts[i]), modules.get(".".join(parts[:i])))
                     for i in range(1, len(parts)) if parts[i].isdigit()]
        layers = [idx for idx, parent in ancestors if isinstance(parent, torch.nn.ModuleList)]
        if len(layers) != 1:
            raise ValueError(f"Cannot identify unique Transformer layer for {name}")
        targets.append(Target(name, layers[0], parts[-1], module))
    if not targets:
        raise ValueError("No supported projection matrices found")
    return targets


@torch.no_grad()
def rtn(weight, bits, group_size=128):
    """Zero-inclusive affine min/max, nearest-even rounding, FP32 scale.

    q in [0, 2**bits-1]; lo=min(min(w),0), hi=max(max(w),0).
    s=(hi-lo)/qmax (s=1 for all-zero groups); z=clip(round(-lo/s)).
    q=clip(round(w/s)+z); w_hat=s*(q-z), cast to source dtype.
    Groups run along input columns independently for each output row.
    A final short group uses only its real elements (padding is zero).
    """
    if bits not in (2, 3, 4, 8) or group_size <= 0 or weight.ndim != 2:
        raise ValueError("Expected a 2D tensor, bits in {2,3,4,8}, positive group size")
    if not torch.isfinite(weight).all():
        raise ValueError("Non-finite reference weights")
    rows, cols = weight.shape
    pad = (-cols) % group_size
    w = torch.nn.functional.pad(weight.float(), (0, pad)).reshape(rows, -1, group_size)
    lo = torch.minimum(w.amin(-1, keepdim=True), torch.zeros((), device=w.device))
    hi = torch.maximum(w.amax(-1, keepdim=True), torch.zeros((), device=w.device))
    qmax = 2 ** bits - 1
    scale = (hi - lo) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = (-lo / scale).round().clamp(0, qmax)
    codes = ((w / scale).round() + zero).clamp(0, qmax)
    return ((codes - zero) * scale).reshape(rows, -1)[:, :cols].to(weight.dtype)


def error_terms(original, candidate):
    a, b = original.float(), candidate.float()
    return float((a - b).square().sum(dtype=torch.float64)), float(a.square().sum(dtype=torch.float64))


def normalized_error(original, candidate):
    numerator, denominator = error_terms(original, candidate)
    return numerator / (denominator + 1e-12)


def fingerprint(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def payload(targets, plan, source_bits=16, group_size=128):
    """Ideal packed codes; metadata is a separate explicit FP32 scale + uint8 z."""
    n = sum(t.n_parameters for t in targets)
    code_bits = sum(t.n_parameters * plan.get(t.name, source_bits) for t in targets)
    groups = sum(t.module.weight.shape[0] * math.ceil(t.module.weight.shape[1] / group_size)
                 for t in targets if t.name in plan)
    return {"quantizable_params": n, "payload_bits": code_bits, "payload_bpw": code_bits / n,
            "metadata_bits": groups * 40, "effective_bpw": (code_bits + groups * 40) / n,
            "actual_serialized_bytes": None}


@contextmanager
def perturb(regions, bits=None, group_size=128):
    """Apply (target,start,end,candidate) regions; restore even after exceptions.

    Candidates are fit on the original complete matrix, then sliced, so chunks
    never change the group fit. Overlapping regions are rejected before mutation.
    CPU backups avoid doubling GPU memory for E1/E6.
    """
    regions = list(regions)
    seen = {}
    for target, start, end, candidate in regions:
        width = target.module.weight.shape[1]
        if not (0 <= start < end <= width):
            raise ValueError("Invalid region bounds")
        if start % group_size or (end != width and end % group_size):
            raise ValueError("Regions must align with quantization groups")
        if any(start < b and end > a for a, b in seen.get(target.name, [])):
            raise ValueError("Overlapping perturbations are not allowed")
        seen.setdefault(target.name, []).append((start, end))
        if candidate is not None and candidate.shape != (target.module.weight.shape[0], end - start):
            raise ValueError("Candidate shape does not match region")
    backups = []
    try:
        with torch.no_grad():
            for target, start, end, candidate in regions:
                weight = target.module.weight
                view = weight[:, start:end]
                backup = view.detach().cpu().clone()
                backups.append((view, backup))
                if candidate is None:
                    candidate = rtn(weight, bits, group_size)[:, start:end]
                view.copy_(candidate)
        yield
    finally:
        with torch.no_grad():
            for view, backup in reversed(backups):
                view.copy_(backup)
            for view, backup in backups:
                if not torch.equal(view.detach().cpu(), backup):
                    raise RuntimeError("Exact weight restoration failed")

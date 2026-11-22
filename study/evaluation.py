"""Token-weighted causal NLL with overlapping context and paired uncertainty."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def windows(tokens, context, stride):
    if not 1 <= stride < context:
        raise ValueError("Require 1 <= stride < context")
    if len(tokens) < 2:
        raise ValueError("At least two tokens are required")
    previous_end = 1
    end = min(context, len(tokens))
    while previous_end < len(tokens):
        start = max(0, end - context)
        yield tokens[start:end], max(1, previous_end - start)
        previous_end = end
        end = min(end + stride, len(tokens))


@torch.inference_mode()
def evaluate(model, tokens, context=512, stride=256, batch_size=2, pad_id=0):
    model.eval()
    device = next(model.parameters()).device
    parts = list(windows(tokens, context, stride))
    sums, counts = [], []
    for offset in range(0, len(parts), batch_size):
        batch = parts[offset:offset + batch_size]
        length = max(len(t) for t, _ in batch)
        ids = torch.full((len(batch), length), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        for i, (t, _) in enumerate(batch):
            ids[i, :len(t)] = torch.as_tensor(t, device=device)
            attention[i, :len(t)] = 1
        output = model(input_ids=ids, attention_mask=attention, use_cache=False)
        for i, (t, first_target) in enumerate(batch):
            total = 0.0
            for pos in range(first_target, len(t), 64):
                stop = min(pos + 64, len(t))
                loss = F.cross_entropy(output.logits[i, pos-1:stop-1].float(), ids[i, pos:stop], reduction="sum")
                total += float(loss)
            sums.append(total)
            counts.append(len(t) - first_target)
        del output
    n = sum(counts)
    nll = sum(sums) / n
    if n != len(tokens) - 1 or not math.isfinite(nll):
        raise RuntimeError("Incorrect token accounting or non-finite NLL")
    return {"nll": nll, "ppl": math.exp(nll) if nll < 709 else None,
            "evaluated_tokens": n, "sequence_loss_sums": sums, "sequence_token_counts": counts}


def paired_interval(measured, baseline, seed=42, resamples=1000, block_windows=8):
    if measured["sequence_token_counts"] != baseline["sequence_token_counts"]:
        raise ValueError("Paired evaluation requires identical target tokens and windows")
    diff = np.array(measured["sequence_loss_sums"]) - np.array(baseline["sequence_loss_sums"])
    count = np.array(measured["sequence_token_counts"])
    ds = np.array([x.sum() for x in np.array_split(diff, np.arange(block_windows, len(diff), block_windows))])
    ns = np.array([x.sum() for x in np.array_split(count, np.arange(block_windows, len(count), block_windows))])
    if len(ds) < 2:
        return None
    indices = np.random.default_rng(seed).integers(len(ds), size=(resamples, len(ds)))
    values = ds[indices].sum(1) / ns[indices].sum(1)
    return np.quantile(values, [.025, .975]).tolist()

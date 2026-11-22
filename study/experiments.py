"""E1--E3 measurements, always derived from the same unmodified reference."""
import time

import torch

from .activations import output_error
from .evaluation import paired_interval
from .io import atomic_json
from .quantization import rtn, perturb, error_terms, normalized_error, payload, fingerprint


def quality(study, measured, baseline):
    return {"baseline_nll": baseline["nll"], "quantized_nll": measured["nll"],
            "validation_nll": measured["nll"], "validation_ppl": measured["ppl"],
            "delta_nll": measured["nll"] - baseline["nll"],
            "delta_nll_ci": paired_interval(measured, baseline, seed=study.config["seed"]),
            "validation_evaluation": measured}


def e1(study):
    store = study.store("e1")
    baseline = study.baseline()
    test_baseline = study.baseline("test")
    cfg = study.config["experiments"]["e1"]
    for bits in [None, *cfg["bits"]]:
        key = "FP" if bits is None else f"W{bits}"
        if store.get(key):
            continue
        start = time.perf_counter()
        numerator, denominator = 0.0, 0.0
        plan = {} if bits is None else {t.name: bits for t in study.targets}
        # Quantize one matrix at a time inside perturb; no full GPU candidate copy.
        regions = [(t, 0, t.module.weight.shape[1], None) for t in study.targets] if bits else []
        if bits:
            for target in study.targets:
                candidate = rtn(target.module.weight, bits, study.config["group_size"])
                a, b = error_terms(target.module.weight, candidate)
                numerator += a
                denominator += b
            del candidate
        with perturb(regions, bits=bits, group_size=study.config["group_size"]):
            measured = study.evaluate() if bits else baseline
            test = study.evaluate("test") if bits in cfg["test_bits"] else None
            if bits is None:
                test = test_baseline
        study.verify()
        row = {**study.common(), **quality(study, measured, baseline),
               "precision": key, "bit_width": bits or next(study.model.parameters()).element_size()*8,
               "test_nll": test["nll"] if test else None,
               "test_ppl": test["ppl"] if test else None,
               "test_evaluation": test,
               "test_delta_nll": test["nll"] - test_baseline["nll"] if test else None,
               "normalized_weight_mse": numerator / (denominator + 1e-12),
               **payload(study.targets, plan, next(study.model.parameters()).element_size() * 8, study.config["group_size"]),
               "runtime": time.perf_counter() - start,
               "quantization_targets": list(plan)}
        store.add(key, row)
        store.export()
    return store.export()


def measure_region(study, target, bits, start, end, candidate, baseline, cache):
    clock = time.perf_counter()
    weight = target.module.weight
    original_hash = fingerprint(weight)
    local_weight = normalized_error(weight[:, start:end], candidate)
    local_output = output_error(cache[target.name], weight, candidate, start, end)
    with perturb([(target, start, end, candidate)], group_size=study.config["group_size"]):
        measured = study.evaluate()
    if fingerprint(weight) != original_hash:
        raise RuntimeError("Cross-trial contamination detected")
    # A small reference forward confirms functional recovery after every trial.
    torch.testing.assert_close(study.probe(), study.reference_logits, rtol=1e-5, atol=1e-5)
    return {**study.common(), **quality(study, measured, baseline),
            "layer_idx": target.layer, "module_name": target.name, "projection_type": target.projection,
            "bit_width": bits, "col_start": start, "col_end": end,
            "n_parameters": weight.shape[0] * (end-start), "normalized_weight_mse": local_weight,
            "output_mse": local_output, "candidate_sha256": fingerprint(candidate),
            "quantizable_params": sum(t.n_parameters for t in study.targets),
            "region_payload_bpw": bits,
            "payload_bpw": (sum(t.n_parameters for t in study.targets) * weight.element_size() * 8
                            - weight.shape[0] * (end-start) * (weight.element_size() * 8 - bits))
                           / sum(t.n_parameters for t in study.targets),
            "runtime": time.perf_counter() - clock}


def e2(study):
    store, baseline, cache = study.store("e2"), study.baseline(), study.activations()
    for bits in study.config["experiments"]["e2"]["bits"]:
        for target in study.targets:
            key = f"{target.name}:W{bits}"
            if store.get(key):
                continue
            start = time.perf_counter()
            candidate = rtn(target.module.weight, bits, study.config["group_size"])
            row = measure_region(study, target, bits, 0, target.module.weight.shape[1], candidate, baseline, cache)
            row["runtime"] = time.perf_counter() - start
            row["granularity"] = "matrix"
            store.add(key, row)
            store.export()
    study.verify()
    return store.export()


def representative_targets(targets, projections):
    layers = sorted({t.layer for t in targets})
    selected = {layers[0], layers[len(layers)//2], layers[-1]}
    return [t for t in targets if t.layer in selected and t.projection in projections]


def e3(study):
    cfg = study.config["experiments"]["e3"]
    chunk_size = cfg["chunk_size"]
    if chunk_size % study.config["group_size"]:
        raise ValueError("Analysis chunk size must be a multiple of group size")
    targets = representative_targets(study.targets, cfg["projections"])
    atomic_json(study.out / "e3_selection.json", {
        "rule": "First, floor(L/2), last discovered Transformer layer; configured projections; selected before reading sensitivities",
        "modules": [t.name for t in targets], "chunk_size": chunk_size})
    store, baseline, cache = study.store("e3"), study.baseline(), study.activations()
    for bits in cfg["bits"]:
        for target in targets:
            candidate = None
            for chunk_idx, start in enumerate(range(0, target.module.weight.shape[1], chunk_size)):
                end = min(start + chunk_size, target.module.weight.shape[1])
                key = f"{target.name}:{start}:{end}:W{bits}"
                if store.get(key):
                    continue
                if candidate is None:
                    candidate = rtn(target.module.weight, bits, study.config["group_size"])
                row = measure_region(study, target, bits, start, end, candidate[:, start:end], baseline, cache)
                row.update(chunk_idx=chunk_idx, granularity="chunk")
                store.add(key, row)
                store.export()
    study.verify()
    return store.export()

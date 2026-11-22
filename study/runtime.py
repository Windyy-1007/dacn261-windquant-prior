import contextlib
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import BACKEND_VERSION
from .activations import collect
from .data import prepare
from .evaluation import evaluate
from .io import atomic_json, digest, Results, software
from .quantization import discover, fingerprint


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def source_digest(root):
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted((root / "study").glob("*.py"))})


class Study:
    def __init__(self, config, root):
        self.config, self.root = config, Path(root)
        self.out = self.root / config["output"]
        self.out.mkdir(parents=True, exist_ok=True)
        self._lock = (self.out / "run.lock").open("w")
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        seed_all(config["seed"])
        self.started = time.perf_counter()
        m = config["model"]
        for revision in (m["revision"], m["tokenizer_revision"]):
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("Pin model and tokenizer to exact 40-character commit revisions")
        self.tokenizer = AutoTokenizer.from_pretrained(m["name"], revision=m["tokenizer_revision"], local_files_only=config["local_only"])
        self.tokens, self.data_metadata = prepare(config, self.tokenizer, self.root)
        self.model = AutoModelForCausalLM.from_pretrained(
            m["name"], revision=m["revision"], dtype=getattr(torch, m["dtype"]),
            local_files_only=config["local_only"], attn_implementation="sdpa").to(config["device"]).eval()
        self.model.requires_grad_(False)
        self.targets = discover(self.model)
        self.by_name = {t.name: t for t in self.targets}
        self.reference_hashes = {t.name: fingerprint(t.module.weight) for t in self.targets}
        self.reference_logits = self.probe()
        # Execution device is included: resuming on different hardware must be explicit.
        protocol = {k: v for k, v in config.items() if k not in ("output", "experiments")}
        self.protocol_id = digest({"config": protocol, "data": self.data_metadata, "backend": BACKEND_VERSION,
                                   "weights": self.reference_hashes})
        path = self.out / "manifest.json"
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing and existing["protocol_id"] != self.protocol_id:
            raise ValueError("Output contains a different protocol; choose a new output directory")
        self.metadata = {"protocol_id": self.protocol_id, "backend": BACKEND_VERSION,
                         "configuration": config, "model": m,
                         "resolved_model_revision": self.model.config._commit_hash,
                         "model_config": self.model.config.to_dict(),
                         "parameter_count": sum(p.numel() for p in self.model.parameters()),
                         "n_layers": len({t.layer for t in self.targets}),
                         "targets": [{"name": t.name, "layer": t.layer, "projection": t.projection,
                                      "shape": list(t.module.weight.shape), "n_parameters": t.n_parameters,
                                      "reference_sha256": self.reference_hashes[t.name]} for t in self.targets],
                         "datasets": self.data_metadata, "software": software(),
                         "source_digest": source_digest(self.root)}
        if existing:
            for package in ("torch", "transformers", "numpy"):
                if existing["software"]["packages"][package] != self.metadata["software"]["packages"][package]:
                    raise ValueError(f"Cannot resume with a different {package} version; use a new output directory")
        if not existing:
            atomic_json(path, self.metadata)
        atomic_json(self.out / "config.json", config)
        # Record every invocation, including source changes after a resumed run.
        atomic_json(self.out / "invocations" / f"{time.time_ns()}.json", self.metadata)
        self.cache = None
        self.baselines = {}

    def validate(self):
        """Gate A on the actual model, including a second full validation pass."""
        path = self.out / "gate_a.json"
        if path.exists() and json.loads(path.read_text()).get("passed"):
            self.verify()
            return
        clock = time.perf_counter()
        first = self.baseline()
        second = self.evaluate()
        self.verify()
        if abs(first["nll"] - second["nll"]) > 1e-7:
            raise RuntimeError("Full-precision baseline is not reproducible within 1e-7 NLL")
        atomic_json(path, {"passed": True, "baseline_nll": first["nll"], "repeated_nll": second["nll"],
                           "evaluated_tokens": first["evaluated_tokens"], "runtime": time.perf_counter()-clock,
                           "exact_weight_restoration": True, "reference_logits_recovered": True})

    @torch.inference_mode()
    def probe(self):
        ids = torch.as_tensor(self.tokens["validation"][:16], device=self.config["device"]).unsqueeze(0)
        return self.model(input_ids=ids, use_cache=False).logits.cpu()

    def verify(self):
        for target in self.targets:
            if fingerprint(target.module.weight) != self.reference_hashes[target.name]:
                raise RuntimeError(f"Reference weights changed: {target.name}")
        torch.testing.assert_close(self.probe(), self.reference_logits, rtol=1e-5, atol=1e-5)

    def evaluate(self, split="validation"):
        return evaluate(self.model, self.tokens[split], **self.config["evaluation"],
                        pad_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

    def baseline(self, split="validation"):
        if split in self.baselines:
            return self.baselines[split]
        path = self.out / f"baseline_{split}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            start = time.perf_counter()
            result = {**self.evaluate(split), "runtime": time.perf_counter() - start,
                      "protocol_id": self.protocol_id}
            atomic_json(path, result)
        self.baselines[split] = result
        return result

    def activations(self):
        if self.cache is not None:
            return self.cache
        path = self.out / "activations.pt"
        if path.exists():
            self.cache = torch.load(path, weights_only=True)
        else:
            self.cache, metadata = collect(self.model, self.targets, self.tokens["calibration"],
                                           self.config["activation_memory_mb"])
            torch.save(self.cache, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").replace(path)
            atomic_json(self.out / "activation_manifest.json", metadata)
        return self.cache

    def common(self):
        return {"model": self.config["model"]["name"], "model_revision": self.config["model"]["revision"],
                "backend": BACKEND_VERSION, "group_size": self.config["group_size"],
                "seed": self.config["seed"], "protocol_id": self.protocol_id,
                "source_digest": self.metadata["source_digest"],
                "dataset": self.config["data"]["eval_repo"], "split": "validation",
                "evaluated_tokens": len(self.tokens["validation"]) - 1,
                "cpu_rss_bytes": psutil.Process().memory_info().rss,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(self.config["device"]) if self.config["device"].startswith("cuda") else 0}

    def store(self, experiment):
        settings = {k: v for k, v in self.config["experiments"][experiment].items() if k != "bits"}
        path = self.out / f"{experiment}_settings.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError(f"{experiment} selection/settings differ from this run; choose another output directory")
        if not path.exists():
            atomic_json(path, settings)
        return Results(self.out, experiment)

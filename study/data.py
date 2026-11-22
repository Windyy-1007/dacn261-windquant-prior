"""Pinned data sources, token cache, deterministic training-only calibration."""
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .io import atomic_json, digest


def text_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def fetch(repo, filename, revision, cache, local_only):
    try:
        return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                     local_files_only=True))
    except (OSError, ValueError):
        local = cache / repo.split("/")[-1] / filename
        if local.exists():
            return local
        if local_only:
            raise FileNotFoundError(f"Missing {repo}@{revision}/{filename}; download it or disable local_only")
        return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                                     local_dir=cache / repo.split("/")[-1]))


def prepare(config, tokenizer, root):
    data = config["data"]
    cache = Path(root) / ".cache"
    cache.mkdir(exist_ok=True)
    key = digest({"data": data, "model": config["model"], "seed": config["seed"], "protocol": "concat-eos-v1"})
    saved = cache / f"tokens-{key}.npz"
    manifest = cache / f"tokens-{key}.json"
    if saved.exists() and manifest.exists():
        with np.load(saved) as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, json.loads(manifest.read_text())
    arrays, splits, hashes = {}, {}, {}
    for split in ("validation", "test"):
        path = fetch(data["eval_repo"], f'{data["eval_config"]}/{split}-00000-of-00001.parquet',
                     data["eval_revision"], cache, config["local_only"])
        texts = [t for t in pd.read_parquet(path)["text"].tolist() if t.strip()]
        hashes[split] = {text_hash(t) for t in texts}
        # Preserve the standard WikiText row order; no random test-driven selection.
        ids = tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]
        ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id] + ids
        budget = data.get(f"{split}_tokens")
        ids = ids[:budget + 1] if budget else ids
        arrays[split] = np.asarray(ids, dtype=np.int64)
        splits[split] = {"repository": data["eval_repo"], "revision": data["eval_revision"],
                         "configuration": data["eval_config"], "split": split,
                         "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "token_sha256": hashlib.sha256(arrays[split].tobytes()).hexdigest(),
                         "input_tokens": len(ids), "evaluated_tokens": len(ids) - 1,
                         "nonempty_rows": len(texts)}
    calibration = data["calibration"]
    if calibration["split"] != "train" or "train" not in calibration["file"]:
        raise ValueError("Calibration must use the training partition")
    path = fetch(calibration["repo"], calibration["file"], calibration["revision"], cache, config["local_only"])
    pool = []
    with gzip.open(path, "rt") as stream:
        for idx, line in enumerate(stream):
            if idx >= calibration["pool_documents"]:
                break
            text = json.loads(line)["text"]
            pool.append((idx, text))
    rng = random.Random(config["seed"])
    rng.shuffle(pool)
    samples, sample_ids, used, rejected = [], [], set(), 0
    for idx, text in pool:
        h = text_hash(text)
        if h in used or h in hashes["validation"] or h in hashes["test"]:
            rejected += 1
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        length = calibration["length"]
        if len(ids) < length:
            continue
        start = rng.randrange(len(ids) - length + 1)
        samples.append(ids[start:start + length])
        sample_ids.append({"row": idx, "token_start": start, "length": length, "text_sha256": h})
        used.add(h)
        if len(samples) == calibration["samples"]:
            break
    if len(samples) != calibration["samples"]:
        raise ValueError("Calibration pool is too small; increase pool_documents explicitly")
    arrays["calibration"] = np.array(samples, dtype=np.int64)
    metadata = {"cache_key": key, "evaluation": splits,
                "calibration": {**calibration, "sample_ids": sample_ids,
                                "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "token_sha256": hashlib.sha256(arrays["calibration"].tobytes()).hexdigest()},
                "overlap_check": {"method": "Whitespace-normalized exact text-row SHA256; not semantic deduplication",
                                  "validation_test_shared_rows": len(hashes["validation"] & hashes["test"]),
                                  "calibration_duplicates_or_eval_overlap_rejected": rejected,
                                  "calibration_vs_eval_shared_rows": len(used & (hashes["validation"] | hashes["test"]))},
                "tokenization": "Nonempty WikiText rows joined by two newlines; one leading BOS; fixed prefix; no padding targets"}
    with saved.with_suffix(".tmp").open("wb") as f:
        np.savez(f, **arrays)
    saved.with_suffix(".tmp").replace(saved)
    atomic_json(manifest, metadata)
    return arrays, metadata

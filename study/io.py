import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import platform
import importlib.metadata

import yaml


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if "extends" in data:
        base = load_config(path.parent / data.pop("extends"))
        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
            return a
        return merge(base, data)
    return data


def software():
    import torch
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        git = None
    return {"python": sys.version, "platform": platform.platform(), "git_commit": git,
            "packages": {p: importlib.metadata.version(p) for p in
                         ["torch", "transformers", "datasets", "numpy", "pandas", "matplotlib", "scipy", "pyyaml"]},
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


class Results:
    """One atomic file per trial; interrupted writes never count as completed."""
    def __init__(self, root, experiment):
        self.root = Path(root)
        self.experiment = experiment
        self.path = self.root / "trials" / experiment
        self.path.mkdir(parents=True, exist_ok=True)

    def filename(self, key):
        return self.path / (digest(key) + ".json")

    def get(self, key):
        path = self.filename(key)
        return json.loads(path.read_text()) if path.exists() else None

    def add(self, key, row):
        row = {"experiment_id": self.experiment, "trial_key": key, **row}
        atomic_json(self.filename(key), row)
        print(json.dumps({"experiment": self.experiment, "trial": key,
                          "delta_nll": row.get("delta_nll"), "runtime": row.get("runtime")}), flush=True)

    def rows(self):
        return [json.loads(p.read_text()) for p in sorted(self.path.glob("*.json"))]

    def export(self):
        rows = self.rows()
        csv_write(self.root / f"{self.experiment}.csv", rows)
        return rows


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})

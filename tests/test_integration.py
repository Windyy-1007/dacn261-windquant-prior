import numpy as np
import torch

from study.analysis import correlations, cluster_interval, rho, report
from study.experiments import e1, e2, e3
from study.extensions import e5, e6
from study.io import atomic_json, load_config
from study.quantization import discover, fingerprint
from study.runtime import Study
from test_quantization import tiny_model


def test_full_pipeline_on_tiny_random_fixture_only(tmp_path):
    """Test-only random model/results stay in pytest tmp_path, never research outputs."""
    from pathlib import Path
    from types import SimpleNamespace
    torch.set_num_threads(2)
    config = load_config(Path(__file__).parents[1]/"configs/common.yaml")
    config.update(device="cpu", group_size=4, activation_memory_mb=.05)
    config["model"]["name"] = "TEST_FIXTURE_ONLY"
    config["evaluation"] = {"context": 8, "stride": 4, "batch_size": 2}
    config["experiments"]["e3"]["chunk_size"] = 8
    config["experiments"]["e4"]["bootstrap_samples"] = 10
    study = Study.__new__(Study)
    study.config, study.out = config, tmp_path
    study.model = tiny_model()
    study.model.requires_grad_(False)
    study.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    study.targets = discover(study.model)
    study.by_name = {t.name: t for t in study.targets}
    rng = np.random.default_rng(42)
    study.tokens = {"validation": rng.integers(1, 32, 25), "test": rng.integers(1, 32, 30),
                    "calibration": rng.integers(1, 32, (3, 8))}
    study.reference_logits = study.probe()
    study.reference_hashes = {t.name: fingerprint(t.module.weight) for t in study.targets}
    study.cache, study.baselines = None, {}
    study.protocol_id = "test-only"
    study.metadata = {"source_digest": "test-only"}
    atomic_json(tmp_path / "manifest.json", {"model": config["model"], "targets": [
        {"name": t.name, "layer": t.layer, "projection": t.projection, "shape": list(t.module.weight.shape)}
        for t in study.targets]})
    study.validate()
    assert len(e1(study)) == 5
    assert len(e2(study)) == 14
    assert len(e3(study)) == 16
    result = correlations(tmp_path, config)
    assert set(result.granularity) == {"matrix", "chunk"}
    assert result.n_units.sum() == 60  # each unit appears in all + its module group
    assert len(e2(study)) == 14  # resumes without accumulating duplicate results
    study.verify()
    status = report(tmp_path, config)
    assert status["gate_b"]
    assert len(list((tmp_path/"figures").glob("*.pdf"))) == 11
    assert len(list((tmp_path/"figures").glob("*.png"))) == 11
    assert "TEST_FIXTURE_ONLY" in (tmp_path/"REPORT.md").read_text()
    config["experiments"]["e5"]["pairs"] = 3
    assert len(e5(study)) == 3
    config["experiments"]["e2"]["bits"] = [2, 3, 4]
    assert len(e2(study)) == 42
    config["experiments"]["e6"]["budgets"] = [2.5, 3.0]
    config["experiments"]["e6"]["random_seeds"] = [42, 43]
    allocations = e6(study)
    assert len(allocations) == 9
    for budget in [2.5, 3.0]:
        matched = [r for r in allocations if r["target_bpw"] == budget and r["strategy"] != "uniform"]
        assert len({r["payload_bits"] for r in matched}) == 1
        assert all(r["realized_bpw"] <= budget for r in matched)
    report(tmp_path, config)
    assert (tmp_path/"figures/E5-A_predicted_vs_observed.pdf").exists()
    assert (tmp_path/"figures/E6-C_matched_costs.png").exists()
    study.verify()


def test_cluster_correlation_handles_degenerate_and_small_samples():
    import pandas as pd
    assert rho([1, 1, 1], [1, 2, 3]) is None
    assert rho([1, 2, 3, 4], [2, 4, 6, 8]) == 1.
    df = pd.DataFrame({"module_name": ["a", "a", "b", "b"], "proxy": [1, 2, 3, 4], "delta_nll": [1, 2, 3, 4]})
    assert cluster_interval(df, "proxy", "module_name", 100, 42) == (None, 0)

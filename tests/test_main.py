"""CLI routing tests: no model download or GPU use is needed here."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prior_entrypoint", ROOT / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.fixture
def dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(entry, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    return calls


def test_default_matches_original_protocol_from_any_directory(tmp_path, monkeypatch, dispatched):
    monkeypatch.chdir(tmp_path)
    entry.main([])
    (config, root, experiments), modes = dispatched[0]
    assert config == entry.load_config(ROOT / "configs/common.yaml")
    assert root == ROOT and experiments == ["e1", "e2", "e3", "e4"]
    assert modes == {"report_only": False, "check_only": False}


def test_all_enables_required_bit_measurements_without_editing_yaml(dispatched):
    entry.main(["--all"])
    config, _, experiments = dispatched[0][0]
    assert experiments == list(entry.ORDER)
    assert config["experiments"]["e2"]["bits"] == [2, 3, 4]
    assert entry.load_config(ROOT / "configs/common.yaml")["experiments"]["e2"]["bits"] == [2]


@pytest.mark.parametrize("requested,expected", [
    (["e1"], ["e1"]), (["e4"], ["e2", "e3", "e4"]),
    (["e5"], ["e3", "e5"]), (["e6"], ["e2", "e6"]),
    (["e5", "e4", "e5"], ["e2", "e3", "e4", "e5"]),
])
def test_prerequisites_run_once_in_order(requested, expected, dispatched):
    entry.main(["--experiments", *requested])
    assert dispatched[0][0][2] == expected


def test_3b_and_cli_overrides_preserve_pinned_revision(dispatched):
    entry.main(["--model", "3b", "--gpu", "2", "--batch-size", "1", "--validation-tokens", "4096",
                "--output", "results/custom", "--allow-downloads", "--experiments", "e6"])
    config = dispatched[0][0][0]
    assert config["model"] == entry.load_config(ROOT / "configs/llama32_3b.yaml")["model"]
    assert config["device"] == "cuda:2"
    assert config["activation_memory_mb"] == 512
    assert config["evaluation"]["batch_size"] == 1
    assert config["data"]["validation_tokens"] == 4096
    assert config["local_only"] is False and config["output"] == "results/custom"
    assert config["experiments"]["e2"]["bits"] == [2, 3, 4]


def test_check_has_separate_output_and_never_runs_sweeps(dispatched):
    entry.main(["--check"])
    assert dispatched[0][0][0]["output"] == "results/startup_check"
    assert dispatched[0][0][2] == [] and dispatched[0][1]["check_only"]


def test_dry_run_subprocess_outside_project_writes_nothing(tmp_path):
    output = tmp_path / "should_not_exist"
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--all", "--dry-run", "--output", str(output)],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    preview = json.loads(result.stdout)
    assert preview["experiments"] == list(entry.ORDER) and preview["e2_bits"] == [2, 3, 4]
    assert preview["output"] == str(output) and not output.exists()


def test_report_uses_saved_run_configuration(tmp_path, dispatched):
    config = entry.load_config(ROOT / "configs/llama32_3b.yaml")
    config["data"]["validation_tokens"] = 1234
    (tmp_path / "config.json").write_text(json.dumps(config))
    entry.main(["--report-only", "--output", str(tmp_path)])
    actual = dispatched[0][0][0]
    assert actual["model"] == config["model"] and actual["data"]["validation_tokens"] == 1234
    assert actual["output"] == str(tmp_path) and dispatched[0][1]["report_only"]


@pytest.mark.parametrize("args", [
    ["--gpu", "-1"], ["--batch-size", "0"], ["--validation-tokens", "0"],
    ["--all", "--experiments", "e1"], ["--check", "--all"], ["--report-only", "--gpu", "1"],
])
def test_invalid_arguments_do_not_launch_experiments(args, dispatched):
    with pytest.raises(SystemExit) as error:
        entry.main(args)
    assert error.value.code == 2 and dispatched == []


def test_report_without_results_is_a_clear_error(tmp_path, dispatched):
    with pytest.raises(SystemExit) as error:
        entry.main(["--report-only", "--output", str(tmp_path)])
    assert error.value.code == 2 and dispatched == []


def test_full_entrypoint_dispatches_one_model_and_all_stages(tmp_path, monkeypatch):
    from study import runtime, experiments, extensions, analysis
    events = []

    class FakeStudy:
        def __init__(self, config, root):
            assert config["experiments"]["e2"]["bits"] == [2, 3, 4]
            events.append("load")

        def validate(self):
            events.append("validate")

    monkeypatch.setattr(runtime, "Study", FakeStudy)
    for module, names in [(experiments, ["e1", "e2", "e3"]), (extensions, ["e5", "e6"])]:
        for name in names:
            monkeypatch.setattr(module, name, lambda study, name=name: events.append(name))
    monkeypatch.setattr(analysis, "correlations", lambda *args: events.append("e4"))
    monkeypatch.setattr(analysis, "report", lambda *args: events.append("report"))
    entry.main(["--all", "--output", str(tmp_path)])
    assert events == ["load", "validate", "e1", "e2", "e3", "e4", "e5", "e6", "report"]


def test_original_script_cli_remains_compatible(monkeypatch):
    from study import cli
    calls = []
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["e3_chunk_sensitivity.py", "--config", str(ROOT / "configs/e3.yaml")])
    cli.main("e3")
    assert calls[0][0][2] == ["e3"]
    assert calls[0][0][0] == entry.load_config(ROOT / "configs/common.yaml")

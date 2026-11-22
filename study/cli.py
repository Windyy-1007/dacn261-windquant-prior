import argparse
from pathlib import Path
import traceback

from .io import atomic_json, load_config


def run(config, root, experiments, *, report_only=False, check_only=False):
    """Shared execution path for main.py and the original experiment scripts."""
    root = Path(root)
    if report_only:
        from .analysis import report
        return report(root / config["output"], config)
    study = None
    try:
        if check_only:
            from .runtime import Study
            study = Study(config, root)
            study.validate()
            baseline = study.baseline()
            result = {"passed": True, "scope": "model/data loading, repeated validation baseline and reference restoration; no E1-E6 sweep",
                      "model": config["model"]["name"], "evaluated_tokens": baseline["evaluated_tokens"],
                      "nll": baseline["nll"], "ppl": baseline["ppl"]}
            atomic_json(study.out / "startup_check.json", result)
            print(f"Startup check passed: NLL={baseline['nll']:.9f}, PPL={baseline['ppl']:.9f}", flush=True)
            return result
        for experiment in experiments:
            if experiment == "e4":
                from .analysis import correlations
                correlations(root / config["output"], config)
            else:
                if study is None:
                    from .runtime import Study
                    study = Study(config, root)
                    study.validate()
                if experiment in ("e1", "e2", "e3"):
                    from . import experiments
                    getattr(experiments, experiment)(study)
                else:
                    from . import extensions
                    getattr(extensions, experiment)(study)
        from .analysis import report
        return report(root / config["output"], config)
    except Exception:
        atomic_json(root / config["output"] / "last_error.json", {"traceback": traceback.format_exc()})
        raise


def main(default=None):
    parser = argparse.ArgumentParser(description="Reproducible DACN RTN quantization experiments")
    parser.add_argument("--config", default="configs/common.yaml")
    parser.add_argument("--experiments", nargs="+", choices=["e1", "e2", "e3", "e4", "e5", "e6"],
                        default=[default] if default else ["e1", "e2", "e3", "e4"])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else root / config_path)
    return run(config, root, args.experiments, report_only=args.report_only)


if __name__ == "__main__":
    main()

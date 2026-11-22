"""Run the quantization study with command-line options; no YAML editing needed."""
import argparse
import json
import os
from pathlib import Path

from study.cli import run
from study.io import load_config

ROOT = Path(__file__).resolve().parent
ORDER = ("e1", "e2", "e3", "e4", "e5", "e6")
DEPENDENCIES = {"e4": ("e2", "e3"), "e5": ("e3",), "e6": ("e2",)}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Giá trị phải là số nguyên lớn hơn 0")
    return number


def gpu_index(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("GPU index phải >= 0")
    return number


def expand_experiments(requested):
    """Run prerequisites first; existing completed trials are resumed/skipped."""
    selected = set(requested)
    for experiment in list(selected):
        selected.update(DEPENDENCIES.get(experiment, ()))
    return [experiment for experiment in ORDER if experiment in selected]


def parser():
    result = argparse.ArgumentParser(
        description="Chạy nghiên cứu lượng tử hóa; mặc định E1–E4 trên LLaMA-3.2-1B.",
        epilog="Ví dụ: python main.py | python main.py --all | python main.py --experiments e6 --gpu 1")
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Chạy đủ E1–E6, tự chuẩn bị độ nhạy W2/W3/W4 cho E6")
    selection.add_argument("--experiments", nargs="+", choices=ORDER, help="Chọn thí nghiệm; tự thêm thí nghiệm phụ thuộc")
    result.add_argument("--model", choices=("1b", "3b"), default="1b", help="Kích thước LLaMA-3.2 base (mặc định: 1b)")
    result.add_argument("--gpu", type=gpu_index, help="GPU CUDA index, ví dụ --gpu 1 (mặc định: 0)")
    result.add_argument("--batch-size", type=positive_int, help="Số cửa sổ đánh giá mỗi batch (mặc định: 2)")
    result.add_argument("--validation-tokens", type=positive_int, help="Ngân sách token đích validation (mặc định: 32768)")
    result.add_argument("--output", help="Thư mục kết quả; đường dẫn tương đối tính từ prior/")
    result.add_argument("--allow-downloads", action="store_true", help="Cho phép tải model/dataset thiếu theo revision đã ghim")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Chỉ kiểm tra model/dữ liệu và baseline; dùng output kiểm tra riêng")
    mode.add_argument("--dry-run", action="store_true", help="Xem kế hoạch; không tải model, không dùng GPU hoặc ghi kết quả")
    mode.add_argument("--report-only", action="store_true", help="Tạo lại báo cáo bằng config.json đã lưu; không đo lại")
    return result


def resolve(args):
    preset = "llama32_3b.yaml" if args.model == "3b" else "common.yaml"
    config = load_config(ROOT / "configs" / preset)
    requested = list(ORDER) if args.all else (args.experiments or list(ORDER[:4]))
    experiments = expand_experiments(requested)
    if "e6" in experiments:
        config["experiments"]["e2"]["bits"] = list(config["experiments"]["e6"]["bits"])
    if args.gpu is not None:
        config["device"] = f"cuda:{args.gpu}"
    if args.batch_size is not None:
        config["evaluation"]["batch_size"] = args.batch_size
    if args.validation_tokens is not None:
        config["data"]["validation_tokens"] = args.validation_tokens
    if args.allow_downloads:
        config["local_only"] = False
    if args.check:
        experiments = []
        config["output"] = "results/startup_check" if args.model == "1b" else "results/startup_check_3b"
    if args.output:
        config["output"] = str(Path(args.output).expanduser())
    if args.report_only:
        output = (ROOT / config["output"]).resolve()
        path = output / "config.json"
        if not path.is_file():
            raise ValueError(f"Chưa có cấu hình của lượt đo: {path}. Chạy thí nghiệm trước hoặc chọn --output đã có kết quả.")
        config = json.loads(path.read_text())
        config["output"] = str(output)
        experiments = []
    return config, experiments


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if (args.check or args.report_only) and (args.all or args.experiments):
        argument_parser.error("--check/--report-only không đi cùng --all hoặc --experiments")
    if args.report_only and any((args.gpu is not None, args.batch_size is not None,
                                 args.validation_tokens is not None, args.allow_downloads)):
        argument_parser.error("--report-only dùng cấu hình đã lưu; không nhận tham số thay đổi phép đo")
    try:
        config, experiments = resolve(args)
    except ValueError as exc:
        argument_parser.error(str(exc))
    preview = {"model": config["model"]["name"], "revision": config["model"]["revision"],
               "device": config["device"], "experiments": experiments,
               "e2_bits": config["experiments"]["e2"]["bits"],
               "validation_tokens": config["data"]["validation_tokens"],
               "batch_size": config["evaluation"]["batch_size"],
               "allow_downloads": not config["local_only"],
               "output": str((ROOT / config["output"]).resolve())}
    print(json.dumps(preview, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return preview
    # Preserve explicit user thread settings and keep the simple command bounded.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    return run(config, ROOT, experiments, report_only=args.report_only, check_only=args.check)


if __name__ == "__main__":
    main()

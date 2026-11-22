# DACN: Low-bit Quantization

Experiments on 2/3/4/8-bit weight quantization for LLaMA-3.2, evaluating sensitivity and error on WikiText-2 with C4 calibration data. Runs E1–E4 by default; E5–E6 are optional extensions.

## Requirements & Installation

- Linux or WSL2, Python 3.10+, and an NVIDIA GPU supporting CUDA and BF16.
- A Hugging Face account with access to the LLaMA-3.2 model.
- Dependencies listed in [requirements.txt](requirements.txt), with a CUDA-enabled PyTorch build compatible with your system.

Run from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf auth login
```

## Quick Start

```bash
# Check the model, data, and baseline; download missing files
python main.py --check --allow-downloads

# Run E1–E4 with LLaMA-3.2-1B on GPU 0
python main.py --allow-downloads

# Run all experiments (E1–E6)
python main.py --all --allow-downloads

# Show available options
python main.py --help
```

Results are saved to `results/llama32_1b_reference/` by default. Omit `--allow-downloads` once all required files are cached. Rerun the same command and configuration to resume unfinished trials.

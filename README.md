# DACN: Low-bit Quantization

Thực nghiệm lượng tử hóa trọng số LLaMA-3.2 ở 2/3/4/8 bit, đánh giá độ nhạy và sai số trên WikiText-2, dùng C4 để hiệu chuẩn. Mặc định chạy E1–E4; E5–E6 là phần mở rộng.

## Requirements & Install

- Linux hoặc WSL2, Python 3.10+, GPU NVIDIA hỗ trợ CUDA và BF16.
- Tài khoản Hugging Face đã được cấp quyền truy cập model LLaMA-3.2.
- Các thư viện trong [requirements.txt](requirements.txt); PyTorch cần hỗ trợ CUDA phù hợp với máy.

Chạy từ thư mục dự án:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf auth login
```

## Quick Start

```bash
# Kiểm tra model, dữ liệu và baseline; tải dữ liệu còn thiếu
python main.py --check --allow-downloads

# Chạy E1–E4 với LLaMA-3.2-1B trên GPU 0
python main.py --allow-downloads

# Chạy toàn bộ E1–E6
python main.py --all --allow-downloads

# Xem các tùy chọn
python main.py --help
```

Kết quả mặc định lưu tại `results/llama32_1b_reference/`. Khi đã có đủ cache, có thể bỏ `--allow-downloads`. Chạy lại cùng lệnh và cấu hình để tiếp tục các trial chưa hoàn tất.

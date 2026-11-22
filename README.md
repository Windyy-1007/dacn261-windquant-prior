# DACN: Low-bit quantization experiments

Bản sao mã thực nghiệm trong `prior`, theo [đặc tả đã cung cấp](SPECIFICATION.md).
Nguồn: `extra/dacn_quantization_study`.
Mã này không nhập các thành phần RL/PPO, AWQ hay bộ tối ưu của WindQuant.
E1–E4 là phạm vi bắt buộc; E5/E6 là phần mở rộng.

**Entry point chính: [`main.py`](main.py). Không cần đọc hoặc sửa YAML để chạy.**
Chương trình tự nạp các giá trị mặc định; bạn chọn mô hình, thí nghiệm, GPU và
thư mục kết quả bằng tham số dòng lệnh.

## Có tự chạy được không?

**Có. Một lệnh sẽ chạy lần lượt E1 → E2 → E3 → E4**, tự lưu từng phép đo,
tổng hợp CSV/JSON, tính tương quan và xuất hình PDF/PNG cùng báo cáo Markdown.
Không cần gọi thủ công từng experiment. Mặc định dùng LLaMA-3.2-1B trên `cuda:0`.

Chương trình tự bỏ qua trial đã hoàn tất **khi bạn chạy lại cùng lệnh**. Nó không
tự khởi động lại sau lỗi hoặc reboot. Chạy trực tiếp cần giữ terminal; dùng
`nohup` bên dưới nếu muốn tiếp tục sau khi ngắt SSH. E5, E6 và mô hình 3B chỉ
chạy khi được chọn bằng các lệnh tương ứng, không nằm trong lượt mặc định.

### Trạng thái đã kiểm tra trên máy này — 14/11/2026

- Môi trường Python trong `.venv` có đủ dependency và CUDA hoạt động.
- Snapshot mô hình/tokenizer 1B và WikiText-2 đã có trong cache Hugging Face.
- C4 train đã được chuẩn bị ở `prior/.cache/c4/en/`, đối chiếu SHA256 với
  shard đã dùng cho thực nghiệm gốc. Cấu hình `local_only: true` hiện dùng được.
- **54 tests đạt**, bao gồm pipeline E1–E6, resume, các tham số `main.py`,
  tự thêm phụ thuộc và tương thích với CLI cũ:
  [kết quả kiểm thử của bản prior](results/validation_tests.xml).
- Đã tải mô hình **1B thật**, tính baseline hai lần và kiểm tra trạng thái tham
  chiếu: **NLL 2.203214329984803; PPL 9.054069543778843; 32.768 token đích**.
  Xem [startup_check.json](results/startup_check/startup_check.json) và
  [gate_a.json](results/startup_check/gate_a.json).
- Đã kiểm tra entry point mới bằng `main.py --check --output results/main_entrypoint_check`:
  [kết quả kiểm tra main.py](results/main_entrypoint_check/startup_check.json).

Lượt kiểm tra khởi động lưu riêng trong `results/startup_check/`. Nó không phải
lượt quét E1–E6 mới. Các kết quả E1–E5 trước đây vẫn nằm ở thư mục nguồn bên dưới.

## Kết quả đo và trạng thái

Các liên kết dưới đây trỏ đến kết quả đã đo trong thư mục nguồn. Khi chạy mã
ở `prior`, kết quả mới được lưu trong `prior/results/` và dữ liệu cache trong
`prior/.cache/`, theo cấu hình tương ứng.

- [Báo cáo nghiên cứu](../extra/dacn_quantization_study/results/llama32_1b_reference/REPORT.md)
- [Kiểm tra điều kiện hoàn thành](../extra/dacn_quantization_study/results/llama32_1b_reference/acceptance.json)
- [Cấu hình, model/data revision, calibration IDs, phần cứng/phần mềm](../extra/dacn_quantization_study/results/llama32_1b_reference/manifest.json)
- [Kết quả kiểm thử tự động](../extra/dacn_quantization_study/results/validation_tests.xml)
- Các bảng `e1.csv` … `e6.csv`, nếu đã đo, nằm trong thư mục kết quả.
- Mỗi hình có cả PDF, PNG và CSV nguồn trong `results/<run>/figures/`.
- `measurement_log.jsonl` tổng hợp trial, runtime và thời điểm file được lưu;
  `artifact_index.json` ghi SHA256 các artifact để kiểm tra toàn vẹn.

`acceptance.json` là trạng thái của phép đo thực tế, không phải tuyên bố rằng
mọi cấu hình có trong mã đều đã chạy. E6 cần E2 ở cả W2/W3/W4; không thể dùng
chỉ độ nhạy W2 để suy ra độ nhạy W3/W4. Kết quả kiểm thử mô hình ngẫu nhiên nhỏ
chỉ tồn tại trong thư mục tạm pytest, không được ghi vào bảng nghiên cứu.

## Chạy ngay trên máy hiện tại

Từ thư mục gốc `WindQuant_v1`, chạy:

```bash
.venv/bin/python -u prior/main.py
```

Nếu đã ở trong `prior/`, lệnh tương đương là:

```bash
../.venv/bin/python -u main.py
```

Các ví dụ còn lại dùng thư mục làm việc `prior/`; từ gốc repository, chạy
`cd prior` một lần trước khi dùng chúng. Khi đã activate môi trường `.venv`,
có thể viết ngắn thành `python main.py`. Môi trường và cache trên máy hiện tại
đã được kiểm tra. `main.py` mặc định đặt OMP/MKL ở 4 thread nếu bạn chưa đặt
các biến môi trường này.

### Các lệnh thường dùng, không cần YAML

```bash
# Mặc định: E1–E4, mô hình 1B, GPU 0:
../.venv/bin/python -u main.py

# Toàn bộ E1–E6:
../.venv/bin/python -u main.py --all

# Chỉ E1:
../.venv/bin/python -u main.py --experiments e1

# E6: tự thêm E2 W2/W3/W4, bỏ qua trial E2 đã hoàn tất:
../.venv/bin/python -u main.py --experiments e6

# Mô hình 3B, GPU 1:
../.venv/bin/python -u main.py --model 3b --gpu 1

# Xem kế hoạch trước, không dùng GPU hoặc ghi kết quả:
../.venv/bin/python main.py --all --dry-run

# Kiểm tra model/dữ liệu và baseline trước khi chạy toàn bộ:
../.venv/bin/python -u main.py --check

# Xem toàn bộ tùy chọn:
../.venv/bin/python main.py --help
```

| Tham số | Công dụng |
| --- | --- |
| Không thêm tham số | Chạy E1–E4, LLaMA-3.2-1B, GPU 0 |
| `--all` | Chạy E1–E6, tự bật đủ W2/W3/W4 cho độ nhạy E2 |
| `--experiments e1 e5` | Chọn thí nghiệm; tự thêm các bước phụ thuộc cần thiết |
| `--model 1b` / `--model 3b` | Chọn model base và revision đã ghim, output mặc định riêng |
| `--gpu 1` | Chọn `cuda:1`; index tính theo các GPU đang được CUDA nhìn thấy |
| `--batch-size 1` | Đổi batch đánh giá; mặc định 2 |
| `--validation-tokens 4096` | Đổi ngân sách validation; mặc định 32768 |
| `--output results/my_run` | Output mới, đường dẫn tương đối tính từ `prior/`, dù gọi lệnh ở đâu |
| `--allow-downloads` | Cho phép tải model/dataset còn thiếu theo revision đã ghim |
| `--check` | Chỉ kiểm tra baseline; mặc định lưu riêng `results/startup_check/` cho 1B hoặc `results/startup_check_3b/` cho 3B |
| `--dry-run` | In kế hoạch đã giải quyết phụ thuộc; không nạp mô hình hay tạo output |
| `--report-only` | Tạo lại báo cáo bằng `config.json` trong output đã có |

Ví dụ chạy thử protocol khác mà giữ riêng kết quả:

```bash
../.venv/bin/python -u main.py --experiments e1 --validation-tokens 4096 --batch-size 1 --output results/quick_e1
```

Ngân sách validation nhỏ là phép chạy thử; muốn báo cáo core theo mặc định,
dùng 32768 token. `--check` không chạy sweep E1–E6. `--check`, `--dry-run` và
`--report-only` là ba chế độ riêng, không ghép với nhau.

Thứ tự xử lý tự động:

1. Nạp giá trị mặc định và tham số dòng lệnh, rồi chuẩn bị tokenizer/dataset/cache token.
2. Nạp mô hình gốc, khám phá projection và kiểm tra baseline (Gate A).
3. E1: đánh giá BF16, W8, W4, W3, W2 trên validation và test.
4. E2: đo từng projection W2; E3: đo từng chunk W2 ở các ma trận đại diện.
5. E4: tính tương quan, bootstrap CI; xuất bảng, hình và `REPORT.md`.

Khi đo, terminal in các dòng JSON có `experiment`, `trial`, `delta_nll` và
`runtime`. Chuẩn bị dữ liệu, thu thập activation hoặc bootstrap có thể mất
một lúc giữa các dòng log. Lệnh kết thúc khi pipeline hoàn tất hoặc gặp lỗi.

### Chạy nền, tiếp tục khi ngắt SSH

Từ `prior/`:

```bash
mkdir -p logs
nohup ../.venv/bin/python -u main.py >> logs/e1-e4.log 2>&1 &
echo $! > logs/e1-e4.pid
```

Theo dõi log hoặc kiểm tra tiến trình:

```bash
tail -f logs/e1-e4.log
ps -p "$(cat logs/e1-e4.pid)" -o pid,etime,stat,args
```

`Ctrl+C` khi đang xem `tail -f` chỉ thoát trình xem log. `nohup` giúp tiến trình
tiếp tục khi terminal/SSH đóng; không giữ tiến trình qua reboot và không tự
retry khi gặp lỗi. Dùng tên log/PID khác nếu chủ động chạy một cấu hình khác.

### Tiếp tục sau khi bị ngắt

Chạy lại đúng lệnh foreground hoặc background với **cùng cấu hình và output**.
Trial đã có JSON hoàn tất được bỏ qua; trial đang dở sẽ được tính lại. Mỗi trial
được lưu atomically, file `.tmp` không được tính là hoàn thành. Không cần cờ
`--resume`; runner đã tích hợp cơ chế này.

Giữ nguyên các tham số đã dùng như `--model`, `--gpu`, `--batch-size`,
`--validation-tokens`, `--allow-downloads`, `--output` và biến
`CUDA_VISIBLE_DEVICES` khi resume. Các mặc định nghiên cứu cũng phải giữ nguyên.
Đổi protocol thì dùng `--output results/ten_luot_moi`. Phiên bản Torch,
Transformers và NumPy cũng phải khớp lượt trước.

Chỉ chạy một tiến trình cho một output. Khóa `flock` báo lỗi nếu output đang
được sử dụng; file `run.lock` còn trên đĩa sau khi tiến trình kết thúc là bình
thường, khóa được hệ điều hành giải phóng. Không xóa file khóa để ép chạy song song.

## Chọn phạm vi chạy

`main.py` tự thêm thí nghiệm phụ thuộc rồi sắp xếp theo E1 → E6. Các trial đã
hoàn tất được bỏ qua, nên có thể chọn thêm phần mở rộng sau lượt core.

| Phạm vi | Điều kiện/cấu hình |
| --- | --- |
| Mặc định E1–E4 | `main.py`; tự chạy đủ bốn thí nghiệm |
| E1–E5 | `main.py --experiments e1 e2 e3 e4 e5` |
| E1–E6 | `main.py --all`; E2 tự đo đủ W2/W3/W4 |
| Chọn E4 | Tự chạy E2 → E3 → E4; dùng `--report-only` nếu chỉ muốn phân tích dữ liệu sẵn có |
| Chọn E5 | Tự chạy E3 → E5 |
| Chọn E6 | Tự chạy E2 W2/W3/W4 → E6 |
| 3B | `main.py --model 3b`, output riêng; cần cache/quyền tải model 3B |

```bash
# Core + tương tác E5 trong một lệnh:
../.venv/bin/python -u main.py --experiments e1 e2 e3 e4 e5

# Toàn bộ E1–E6, có đủ độ nhạy W2/W3/W4 cho mixed precision:
../.venv/bin/python -u main.py --all

# Bổ sung E6 sau khi lượt E1–E4 đã xong:
../.venv/bin/python -u main.py --experiments e6

# Mở rộng 3B sau khi hoàn thành 1B:
../.venv/bin/python -u main.py --model 3b
```

Để chạy nền, thêm các tham số mong muốn sau `main.py` trong lệnh `nohup` và
dùng log/PID tương ứng. Ví dụ chạy cả E1–E6:

```bash
mkdir -p logs
nohup ../.venv/bin/python -u main.py --all >> logs/e1-e6.log 2>&1 &
echo $! > logs/e1-e6.pid
```

## Xem tiến độ và kết quả

Với cấu hình 1B mặc định, kết quả mới nằm ở `prior/results/llama32_1b_reference/`:

| File/thư mục | Nội dung |
| --- | --- |
| `trials/e1/`, `trials/e2/`, `trials/e3/` | JSON của từng trial đã hoàn tất, dùng để resume |
| `e1.csv` … `e6.csv` | Bảng phép đo; chỉ có dữ liệu của experiment đã chạy |
| `manifest.json`, `config.json`, `invocations/` | Protocol, cấu hình, phiên bản và metadata |
| `REPORT.md`, `figures/` | Báo cáo và hình PDF/PNG/CSV, xuất ở bước tổng hợp cuối |
| `gate_a.json`, `acceptance.json` | Kết quả kiểm tra baseline và mức độ hoàn thành |
| `last_error.json` | Traceback của lỗi được CLI bắt; đọc cùng log terminal |

Đếm các trial đã lưu, kể cả khi runner vẫn đang chạy:

```bash
../.venv/bin/python - <<'PY'
from pathlib import Path
root = Path('results/llama32_1b_reference/trials')
for experiment in ['e1', 'e2', 'e3', 'e5', 'e6']:
    print(experiment, len(list((root / experiment).glob('*.json'))))
PY
```

Lượt 1B mặc định cần **5 trial E1, 112 trial E2 W2, 144 trial E3 W2**. E4
tổng hợp dữ liệu, không có thư mục trial riêng. Nếu bật E5 mặc định cần 24 cặp;
nếu dùng cấu hình E6, E2 cần 336 trial cho ba bit và E6 có 15 kế hoạch đánh giá.

Sau khi pipeline kết thúc, xem `acceptance.json`: `gate_a=true` và
`gate_b=true` xác nhận các điều kiện core mà runner kiểm tra đã đạt. Đọc kèm
`REPORT.md` để biết phạm vi, CI và giới hạn. Exit code 0 hoặc một bảng đã xuất
chưa đủ để kết luận hoàn thành khi bạn chỉ chọn chạy một phần experiment.

Để tạo lại báo cáo từ trial có sẵn sau khi dừng/chạy xong, không nạp mô hình GPU:

```bash
../.venv/bin/python main.py --report-only
# Với một output riêng:
../.venv/bin/python main.py --report-only --output results/quick_e1
```

Nếu chỉ có một phần trial, báo cáo sẽ phản ánh mức độ hoàn thành tương ứng;
lệnh này không thực hiện các phép đo còn thiếu. `last_error.json` có thể còn
từ lần chạy lỗi cũ; đối chiếu thời gian với log và trạng thái của lần chạy mới.
`--report-only` lấy đúng cấu hình đã lưu của lượt đo; không cần nhập lại GPU,
batch size hoặc ngân sách token. Nó báo lỗi nếu output chưa có `config.json`.

## Cài đặt và chuẩn bị trên máy khác

Từ gốc repository, khi chưa có môi trường:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r prior/requirements.txt
.venv/bin/python -m pytest prior/tests -q
cd prior
```

`requirements-measured.txt` ghi phiên bản thư viện trong môi trường đã đo.
CUDA/GPU và commit Git được ghi trong manifest; hash nguồn được ghi trong
manifest của từng invocation. Không cam kết tái lập bit-for-bit qua kiến trúc
GPU, phiên bản CUDA hoặc thư viện khác. Khi resume, Torch/Transformers/NumPy
phải trùng phiên bản. Giữ nguyên cấu hình số thread để so sánh thời gian.

## Chuẩn bị dữ liệu và mô hình

Mặc định `local_only: true` để dùng snapshot đã có, không tải ngầm phiên bản
mới. Trên máy hiện tại, cache cần cho 1B đã được chuẩn bị và kiểm tra ở trên.
Trên máy mới, khi chưa có cache, cần quyền truy cập model LLaMA và đăng nhập
Hugging Face nếu repository yêu cầu xác thực:

```bash
../.venv/bin/python -m huggingface_hub.cli.hf auth login
```

Sau đó chạy với `--allow-downloads` **ngay từ lượt đầu tiên**:

```bash
../.venv/bin/python -u main.py --allow-downloads
```

Runner sẽ tải các file cần thiết theo revision đã ghim, tạo cache token và
tiếp tục thí nghiệm. Giữ cờ `--allow-downloads` khi resume cùng output;
đổi nó giữa chừng sẽ thay protocol ID. Chỉ tải một shard C4 train, không tải
toàn bộ C4. Dùng `python -m ...` cho pip/pytest/CLI để tránh launcher trỏ đến
đường dẫn interpreter cũ khi virtual environment đã được di chuyển.

Nếu đang ở repository này và muốn khôi phục riêng cache C4 từ bản gốc để dùng
`local_only: true`, chạy từ `prior/`:

```bash
mkdir -p .cache/c4/en
cp ../extra/dacn_quantization_study/.cache/c4/en/c4-train.00000-of-01024.json.gz .cache/c4/en/
```

Model và tokenizer bắt buộc dùng commit SHA đầy đủ. Mặc định:

| Thành phần | Snapshot |
| --- | --- |
| LLaMA-3.2-1B base và tokenizer | `4e20de362430cd3b72f300e6b0f18e50e7166e08` |
| Salesforce/wikitext | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| allenai/c4 | `1588ec454efa1a09f29cd18ddd04fe05fc8653a2` |

Calibration lấy 128 đoạn dài 512 token từ C4 **train**: xáo trộn 10.000 tài liệu
đầu của shard `en/c4-train.00000-of-01024.json.gz` với seed 42, chọn tài liệu
đủ dài rồi chọn offset theo cùng RNG. Lưu row ID, token offset, text hash và
hash toàn file. Loại bản sao và các tài liệu có hash trùng hàng WikiText.

Validation là 32.768 token đích đầu của WikiText-2 validation. Test dùng toàn bộ
split. Các hàng không rỗng nối bằng hai newline; thêm một BOS ở đầu. Token IDs
được cache dưới `.cache/`; manifest lưu hash và số token chính xác. Không dùng
test để chọn bit, ma trận đại diện, cặp tương tác hoặc hyperparameter. E1 chọn
đo test cả W8/W4/W3/W2 **trong cấu hình trước khi đo**.

Kiểm tra overlap là exact text-row hash sau chuẩn hóa khoảng trắng. Các heading
hay đoạn lặp giữa validation/test có thể trùng; số lượng được công bố trong
manifest. Đây không phải bảo đảm tách biệt ngữ nghĩa hoặc kiểm tra trùng với
dữ liệu pretraining của LLaMA. Không thay đổi benchmark test sau khi xem kết quả.

`--model 3b` tự chọn preset với model/tokenizer commit SHA chính xác và output
riêng; preset này chưa đồng nghĩa đã có kết quả đo 3B.
Layer và kích thước ma trận được khám phá từ
`ModuleList`/`Linear`, không hardcode số layer. Chạy 1B thành công trước.

## Định nghĩa RTN (`affine-rtn-v1`)

Với từng nhóm tối đa 128 phần tử liên tiếp theo chiều input của **một hàng**:

```text
q_min = 0; q_max = 2**b - 1
lo = min(min(w), 0); hi = max(max(w), 0)
scale = (hi - lo) / q_max           # scale = 1 nếu cả nhóm bằng 0
zero = clip(round(-lo / scale), 0, q_max)
q = clip(round(w / scale) + zero, 0, q_max)
w_hat = scale * (q - zero)
```

`round` là round-to-nearest, ties-to-even của PyTorch. Tính min/max, scale,
zero-point và giải lượng tử bằng FP32, rồi cast `w_hat` về dtype nguồn. Không
stochastic rounding, clipping percentile, outlier protection hoặc fallback.
Đưa 0 vào khoảng min/max xử lý được cả nhóm hằng dương/hằng âm. Nhóm cuối
ngắn dùng các phần tử thực; zero-padding nội bộ không đổi khoảng zero-inclusive.

Toàn bộ E1–E6 dùng đúng hàm `study.quantization.rtn`. Chunk phân tích 256 cột
khác với nhóm lượng tử 128 cột. Candidate của chunk là lát cắt của kết quả RTN
toàn ma trận; phép đo joint dùng lại cùng giá trị và kiểm tra SHA256 với phép
đo individual. Chunk phải nằm trên biên nhóm; không fit lại theo cặp tương tác.

Chỉ lượng tử `q/k/v/o_proj`, `gate/up/down_proj`. Giữ nguyên embedding,
LM head, normalization, activation và KV cache. Không training.

## Loss, reconstruction và payload

- Context 512, stride 256. Mỗi token đích được tính đúng một lần với phần ngữ
  cảnh chồng lấp; mỗi lần reset position theo cửa sổ. Cửa sổ cuối ngắn được
  right-pad để batch nhưng padding không nằm trong loss. Token đầu của corpus
  chỉ làm ngữ cảnh. NLL là tổng CE chia **tổng token đích**, không trung bình
  không trọng số của các batch. `PPL = exp(NLL)`.
- Weight error: `sum((W-W_hat)^2) / (sum(W^2)+1e-12)`.
- Output error: `sum((X @ (W-W_hat).T)^2) / (sum((X @ W.T)^2)+1e-12)`.
  Đối với chunk, cắt cả `X[:, start:end]` và `W[:, start:end]`.
- Activation lấy trên model gốc, chỉ hook projection cần thiết; lấy các token
  phân bố đều trên **mọi** mẫu calibration. Cache cuối giới hạn 256 MiB, chia
  quota cho các module và số mẫu; không giữ toàn bộ activation. Hook luôn được
  gỡ kể cả khi forward lỗi. Quota quá nhỏ để lấy mỗi mẫu sẽ báo lỗi.
- `payload_bpw = sum(n_i*b_i)/sum(n_i)` cho **tập projection lượng tử hóa**.
  E2/E3 cũng ghi `region_payload_bpw` và chi phí của toàn scope khi các vùng
  còn lại vẫn ở precision nguồn.
- `effective_bpw` thêm metadata giả định **32-bit scale + 8-bit zero-point**
  mỗi nhóm. Payload không gồm embedding/head/norm hoặc alignment/container
  overhead. `actual_serialized_bytes=null`: không có packed checkpoint.
- Đây là fake quantization để đo chất lượng, tensor vẫn BF16. Không suy ra
  giảm file size, giảm VRAM, hay tăng tốc từ kích thước tensor fake-quantized.
- `runtime` đo wall-clock từng trial; thời gian baseline và khởi tạo/cache được
  lưu riêng. `peak_gpu_bytes` là CUDA peak allocated của tiến trình đến trial
  đó, không phải tổng VRAM GPU dùng chung; `cpu_rss_bytes` là RSS tại thời điểm ghi.

## Nội dung từng thí nghiệm

| Thí nghiệm | Thao tác và sản phẩm |
| --- | --- |
| E1 | BF16, W8, W4, W3, W2; validation/test, MSE, payload; 2 đường precision–quality/error |
| E2 | Mỗi lần đúng 1 projection; W2 mặc định; heatmap, phân bố theo loại, đường theo độ sâu, bảng cực trị |
| E3 | Layer đầu, giữa `floor(L/2)`, cuối × q/gate/down; mọi chunk 256 cột; phân bố, median/IQR/std/range |
| E4 | Tái sử dụng E2/E3; Spearman weight/output error với ΔNLL; matrix/chunk × all/attention/MLP; CI và rank disagreement |
| E5 | 24 cặp chunk; phân tầng same-matrix/same-layer/cross-layer × high-high/high-low/low-low; residual và CI |
| E6 | Greedy theo lợi ích NLL/bit từ E2; random nhiều seed; uniform endpoints; đo toàn model và lưu plan |

E3 lưu quy tắc lựa chọn trước khi đọc kết quả. E4 không gộp các bit-width hoặc
granularity. Bootstrap tương quan lấy lại **layer** cho E2 và **ma trận** cho E3,
2.000 lần với seed 42. Báo số cluster và số bootstrap hợp lệ; ít hơn 3 cluster
thì CI không được ước lượng. Rank ties dùng average rank; proxy hằng thì
correlation không xác định, không tự gán 0.

ΔNLL/interaction CI dùng paired block bootstrap (1.000 lần), mỗi khối 8 cửa
sổ liên tiếp, cùng token và loss mask cho các điều kiện. Phụ thuộc dài hạn
vẫn có thể tồn tại; CI không phải kiểm định ý nghĩa thống kê đã hiệu chỉnh.

E5 giữ cả residual dương/âm, không chọn lọc theo kết quả joint. Cặp high/low
được xác định từ median độ nhạy individual trên validation. Diễn giải chỉ
giới hạn ở các cặp đã lấy mẫu, không quy cho một cơ chế causal trong Transformer.

E6 khởi tạo mức thấp nhất rồi xét mọi upgrade khả thi có lợi; dừng khi hết
budget hoặc hết upgrade có lợi. Budget có thể còn dư và được báo đúng giá trị.
Random baseline hoán vị bit **trong nhóm ma trận có cùng số tham số**, bảo toàn
chính xác realized cost và histogram từng nhóm kích thước của greedy. Đây là
random baseline có điều kiện, không phải lấy đều mọi plan khả thi. Ở fractional
budget, uniform W2/W3/W4 là điểm tham chiếu; chỉ random/greedy cùng realized
cost mới được gọi là so sánh matched-budget.

## An toàn trạng thái, resume và kiểm thử

`perturb` dùng `try/finally`, CPU backup và kiểm tra exact equality sau restore.
Mỗi E2/E3 kiểm tra hash ma trận và logits mẫu trở về baseline. Cuối mỗi sweep
kiểm tra hash toàn bộ projection. Không reload toàn model cho từng trial.
Đổi protocol/dataset/revision cần output riêng; file manifest ngăn trộn dữ liệu.
Không sửa các file trial thủ công để ép resume. Chỉ một tiến trình GPU cho
một run; các thư mục output độc lập có thể chạy riêng nếu đủ tài nguyên.

Các kiểm thử bao phủ phương trình RTN, nhóm đuôi/hằng/zero, tính quyết định,
biên chunk, restore khi exception, trọng số ngoài vùng không đổi, logits phục
hồi, shift/mask/token weighting, hook cleanup, memory quota, persistence,
pair sampling, greedy budget, random đúng cost, và pipeline E1–E6 trên mô hình
LLaMA ngẫu nhiên nhỏ trong thư mục tạm.

## Checklist tái lập

- [ ] Giữ model/tokenizer/dataset commit SHA và sample IDs.
- [ ] Giữ token hash, context, stride, số token và scope projection.
- [ ] Dùng cùng RTN version, group size, dtype, precision set và selection rule.
- [ ] Kiểm thử đạt và `gate_a.json` xác nhận baseline lặp lại được.
- [ ] Mỗi trial hoàn tất có JSON, CSV tổng hợp, runtime và protocol ID.
- [ ] Đủ 5 cấu hình E1, toàn bộ W2 E2, mọi chunk của ít nhất 6 ma trận E3.
- [ ] E4 có cả 2 lỗi cục bộ, số mẫu/cluster và CI, hình PDF/PNG/CSV.
- [ ] Đọc `acceptance.json` và giới hạn trong báo cáo trước khi tuyên bố hoàn thành.

## Bố cục

`main.py` là entry point dùng hằng ngày. `study/` chứa quantization, evaluation, dữ liệu, cache activation, runtime,
experiments, analysis và allocation. `experiments/` là entrypoint riêng.
`configs/` chứa preset nội bộ; không cần sửa khi dùng các tùy chọn của `main.py`.
`tests/` độc lập với test/research code có sẵn
ở repository gốc. Cache tải xuống và activation lớn không đưa vào Git.

### Giao diện cấu hình nâng cao

Các script cũ và `python -m study.cli --config ...` vẫn hoạt động để tương
thích các lệnh trước đây. Giao diện cũ chạy đúng danh sách `--experiments`,
không tự thêm phụ thuộc; `main.py` cung cấp cách chạy đơn giản hơn ở trên.
Các thông số nghiên cứu ít thay đổi như group size, chunk size và revision
vẫn nằm trong preset. Tham số dòng lệnh chỉ ghi đè cấu hình trong bộ nhớ,
không sửa file YAML. Cấu hình thực sự dùng được lưu thành `config.json` ở output.
#   d a c n 2 6 1 - w i n d q u a n t - p r i o r  
 
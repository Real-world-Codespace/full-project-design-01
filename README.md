# Multi-Building Cooling Load Forecasting

Dự án production-oriented dựa trên **Global AI Challenge for Building E&M Facilities 2025**: dự báo tải làm mát liên tục cho nhiều tòa nhà, với forecast horizon mặc định 3 giờ.

Notebook là nơi điều phối và trình bày thí nghiệm. Toàn bộ logic có thể tái sử dụng, kiểm thử và phục vụ API nằm trong `src/cooling_load`.

## Kiến trúc

```text
S3/local raw data
      │
      ▼
ingestion + manifest + checksum
      │
      ▼
schema validation → time-aware preprocessing
      │
      ▼
lag/rolling/HVAC features
      │
      ├── K-Means operating regimes (fit on train only)
      ▼
temporal CV → chọn họ model tốt nhất → tuning đúng họ model đã chọn
      │
      ▼
versioned model bundle
      │
      ├── FastAPI / Docker
      └── drift and performance monitoring
```

## Cấu trúc thư mục

```text
configs/                 Cấu hình nguồn dữ liệu và pipeline
data/                    raw, interim, processed và ingestion manifests
notebooks/               Entry point chạy pipeline theo thứ tự 00–06
src/cooling_load/        Logic production
  ingestion/             S3/local adapter, checksum và manifest
  api/                   HTTP schemas, inference service và FastAPI app
analysis_functions/      Module EDA có sẵn được tái sử dụng
processing_functions/    Module preprocessing có sẵn được bảo toàn
models/                  Model bundle triển khai
reports/                 Metrics và hình ảnh
tests/                   Unit/integration tests
```

## Chạy từ một bản clone mới, không cần AWS

Yêu cầu: Git, GNU Make và Python 3.11+.

```bash
git clone <repository-url>
cd <repository-directory>
make setup
make verify
```

`make setup` tự tạo `.venv` và cài toàn bộ dependency. `make verify` chạy test, sau đó thực thi tuần tự bảy notebook. Vì cấu hình mặc định là `ingestion.mode: local`, notebook đầu tiên tự tạo dữ liệu synthetic khi `data/raw/cooling_load.csv` chưa tồn tại. Không cần AWS account để xác minh repository.

Các lệnh thường dùng:

```bash
make test       # unit và integration tests
make pipeline   # chạy 00–06, lưu notebook đã chạy vào reports/
make notebooks  # mở JupyterLab
make api        # chạy FastAPI sau khi model đã được train
```

Pipeline chạy notebook theo thứ tự:

1. `00_ingestion.ipynb`
2. `01_eda.ipynb`
3. `02_preprocessing.ipynb`
4. `03_clustering_features.ipynb`
5. `04_training_tuning.ipynb`
6. `05_evaluation_inference.ipynb`
7. `06_monitoring.ipynb`

Notebook `01_eda.ipynb` luôn lưu bộ chart chuẩn vào `reports/figures/eda/` và summary dạng JSON vào `reports/eda_summary.json`. Notebook hiển thị lại chính các file đã lưu, vì vậy chart xem trong notebook và chart dùng cho báo cáo là cùng một artifact.

## Ingest từ S3

Đổi `ingestion.mode` trong `configs/base.yaml` thành `s3`, sau đó cấu hình:

```yaml
ingestion:
  mode: s3
  s3:
    bucket: your-ml-data-bucket
    prefix: cooling-load/raw/
    profile: default
```

Credentials không được ghi vào repository. Local dùng AWS profile; production dùng IAM role với quyền tối thiểu `s3:ListBucket` và `s3:GetObject` cho đúng prefix.

Ingestion giữ nguyên raw data, tải qua file `.partial`, xác minh kích thước, so sánh ETag, tính SHA-256 và ghi `data/manifests/ingestion_manifest.json`. Chạy lại sẽ bỏ qua object local có cùng ETag và kích thước.

## Data contract

Dataset canonical gồm:

```text
timestamp, building_id, cooling_load_kwh,
outdoor_temperature_c, relative_humidity_pct,
chilled_water_supply_c, chilled_water_return_c,
chilled_water_flow_m3h, active_chillers, occupancy_proxy
```

Nếu dữ liệu challenge có tên cột khác, thực hiện mapping ở adapter ingestion; không sửa logic model theo tên cột nguồn.

## Chống leakage

- Split theo timestamp, không random split.
- Feature target và sensor đều bị shift ít nhất bằng forecast horizon.
- Rolling feature được tính sau khi shift.
- K-Means chỉ fit trên training set rồi mới transform validation/test.
- Test set chỉ được đánh giá sau khi chọn và tune model.
- API gọi lại đúng `CoolingLoadFeatureBuilder` đã dùng khi train.

## API

Sau khi notebook training tạo `models/champion.joblib`:

```bash
make api
```

Endpoints:

- `GET /health/live`
- `GET /health/ready`
- `GET /model-info`
- `POST /forecast`
- `GET /docs`

`POST /forecast` nhận lịch sử của đúng một tòa nhà. Nên gửi ít nhất 168 giờ để đầy đủ lag/rolling feature; model vẫn có imputer để xử lý những history ngắn hơn.

## Lưu model bundle lên S3

Với scikit-learn, artifact cần triển khai không chỉ là trọng số. File `champion.joblib` chứa estimator, imputer, K-Means clusterer, feature schema, config snapshot và metrics. Bật model registry trong `configs/base.yaml`:

```yaml
model_registry:
  enabled: true
  bucket: your-model-bucket
  prefix: cooling-load/models
  profile: cooling-load-dev
  server_side_encryption: AES256
```

Notebook training sẽ luôn ghi model local bằng atomic replace trước, rồi publish:

```text
s3://your-model-bucket/cooling-load/models/
├── <created-at>-<model-name>/
│   ├── champion.joblib
│   └── metadata.json
└── latest.json
```

Mỗi version key là bất biến; `latest.json` là con trỏ có thể thay đổi, chứa URI, SHA-256, kích thước, metrics và thời điểm tạo. Production nên deploy bằng URI version cụ thể thay vì đọc `latest.json` trực tiếp.

FastAPI có thể tải model về local cache khi startup:

```bash
export MODEL_S3_URI=s3://your-model-bucket/cooling-load/models/<version>/champion.joblib
export MODEL_SHA256=<sha256-from-metadata>
export MODEL_PATH=models/champion.joblib
make api
```

Local dùng AWS profile. Trên ECS, EKS hoặc EC2, để `AWS_PROFILE` trống và cấp IAM role có `s3:GetObject` cho model prefix. Role dùng để training cần thêm `s3:PutObject`; bucket nên bật versioning và chặn public access.

## Docker

```bash
docker compose up --build
curl http://localhost:8000/health/ready
```

Container chạy bằng non-root user. Model được mount read-only để có thể thay artifact mà không rebuild source image.

## Bảo trì model

- Theo dõi PSI của feature và phân phối `operating_regime`.
- Khi có ground truth, tính MAE/RMSE/WAPE theo tòa nhà và regime.
- Cảnh báo khi PSI ≥ 0.1; cân nhắc retrain khi PSI ≥ 0.25.
- Lưu `model_name`, `created_at`, metrics và config snapshot trong model bundle.
- Chỉ promote model mới khi vượt seasonal baseline và champion hiện tại trên temporal holdout.

## Lưu ý dữ liệu

File do `make sample` tạo là dữ liệu synthetic để toàn bộ repository chạy được ngay. Nó không phải dữ liệu chính thức của cuộc thi. Khi dùng dữ liệu thật, giữ nguyên schema canonical và thay nguồn trong cấu hình.

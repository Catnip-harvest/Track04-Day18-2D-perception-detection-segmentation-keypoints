# Báo cáo Lab 18 — 2D Perception (Track 4)

**Học viên:** Hoàng Quốc Việt — 2A202602563

Link notebook đã chạy: https://github.com/Catnip-harvest/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

(Bản xem dự phòng nếu GitHub không hiển thị được notebook vì dung lượng ảnh: https://nbviewer.org/github/Catnip-harvest/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb)

## Môi trường chạy

Notebook được chạy từ đầu đến cuối một lần, không ô nào lỗi (tương đương *Restart session and run all*), trên GPU **Tesla T4** của Kaggle:
`torch 2.11.0+cu128 · torchvision 0.26.0 · ultralytics 8.4.171 · device = cuda`. Phần 4 train đủ **40 epoch, imgsz 640**.
`ket_qua.json` là file do ô `final_report()` sinh ra.

| Hạng mục | Kết quả |
|---|---|
| 6 hàm tự cài đặt, `polygon_to_mask` + `mask_to_yolo_seg`, `FLIP_IDX` | tất cả ✅ (không dùng phao) |
| 1C — latency | đủ 4 cấu hình; NMS của tôi và Ultralytics cùng giữ 5 box (4 người, 1 xe buýt) |
| 2C — `autolabel/bus.txt` | 5 object, round-trip IoU 0.967–0.983 |
| 4B — tiger-pose val | Box mAP50-95 0.914 · Pose mAP50 0.995 · Pose mAP50-95 0.436 · train 3.9 phút |
| Câu hỏi | 12/12 |
| ⭐ 1D `average_precision` | ✅, AP = 0.535 trên ví dụ slide, có hình đường PR |
| ⭐ 4C | bảng 2 model × 2 tập val, giải thích bên dưới |
| ⭐ Bài tập về nhà | lựa chọn 3: export ONNX và đo latency CPU, báo cáo bên dưới |

## ⭐ 4C — Tập val lật gương: metric nào đã che lỗi `flip_idx`?

| Model | Pose mAP50-95 — val gốc | Pose mAP50-95 — val lật gương |
|---|---:|---:|
| `flip_idx` giải phẫu | 0.436 | 0.425 |
| `flip_idx` đồng nhất | 0.415 | 0.275 |

Trên **val gốc**, hai model gần như ngang nhau (0.436 so với 0.415), và Pose mAP50 của cả hai đều là 0.995. Nhìn các con số này thì
không thấy có bug. Trên **val lật gương** (ảnh lật ngang, nhãn đổi theo quy ước giải phẫu, giả lập hổ quay trái), model giải phẫu gần
như giữ nguyên (−0.011), còn model đồng nhất tụt từ 0.415 xuống 0.275, mất khoảng một phần ba. Pose mAP50 của model này cũng rơi
từ 0.995 xuống 0.877.

Thứ che lỗi không phải công thức mAP mà là **tập val có cùng phân phối với tập train**. Mọi con hổ trong train (210/0) và val (53/0) đều
quay phải, nên quy ước sai "chân phía camera luôn là `right_*`" vẫn được chấm là đúng. Pose mAP50 còn che kỹ hơn vì đã bão hoà ở 0.995
cho cả hai model; chỉ mAP50-95 trên một tập val đổi hướng mới lộ ra khoảng cách. Nếu thiết kế lại tập val, tôi sẽ:

1. thêm ảnh hổ quay trái (hoặc bản lật gương đã đổi nhãn đúng), phân tầng theo hướng quay;
2. chia train/val theo đoạn video thay vì theo khung hình (Frame_17 và Frame_18 trong val gần như trùng nhau);
3. báo cáo kèm tỉ lệ đảo trái/phải và sai số theo từng keypoint, không chỉ một con số mAP.

## ⭐ Bài tập về nhà (lựa chọn 3) — Export ONNX và đo latency trên CPU

**Cách làm** (ô cuối notebook, trước phần Tổng kết): export `yolo26n.pt` sang ONNX hai lần bằng `model.export(format="onnx", imgsz=640, device="cpu", nms=...)`.
`nms=None` (mặc định) giữ head **one-to-many** và để NMS chạy ngoài model. `nms=False` export head **one-to-one** (end2end, output `(1, 300, 6)`,
không cần NMS). Đo bằng hàm `bench` của 1C (warm-up + trung bình 30 lần) trên `bus.jpg`, ONNX Runtime 1.30.0 với `CPUExecutionProvider`,
CPU Intel Xeon @ 2.00 GHz, 4 luồng. Thêm 4 dòng PyTorch-CPU để so sánh. Số liệu thô lưu ở `submission/onnx_latency.json`.

| Runtime | Cấu hình | preprocess (ms) | inference (ms) | postprocess (ms) | số box | tổng (ms) |
|---|---|---:|---:|---:|---:|---:|
| ONNX Runtime CPU | one-to-many + NMS, conf 0.25 | 3.94 | 75.01 ⚠️ | 1.59 | 5 | 80.54 |
| ONNX Runtime CPU | one-to-one NMS-free, conf 0.25 | 3.85 | 49.67 | 0.44 | 5 | 53.96 |
| ONNX Runtime CPU | one-to-many + NMS, conf 0.001 | 3.90 | 45.88 | 2.18 | 186 | 51.96 |
| ONNX Runtime CPU | one-to-one NMS-free, conf 0.001 | 4.30 | 46.37 | 0.44 | 177 | 51.11 |
| PyTorch CPU | one-to-many + NMS, conf 0.25 | 3.09 | 61.57 | 1.07 | 5 | 65.73 |
| PyTorch CPU | one-to-one NMS-free, conf 0.25 | 2.80 | 58.16 | 0.29 | 5 | 61.25 |
| PyTorch CPU | one-to-many + NMS, conf 0.001 | 3.15 | 63.17 | 1.97 | 203 | 68.29 |
| PyTorch CPU | one-to-one NMS-free, conf 0.001 | 3.13 | 63.10 | 0.34 | 204 | 66.57 |

**Nhận xét.**

- **Dòng ⚠️ là một đợt nhiễu đo, không phải đặc tính của head.** Inference không phụ thuộc conf (conf chỉ lọc sau khi mạng chạy xong), mà cùng file ONNX one-to-many
  ở conf 0.001 chỉ mất 45.88 ms. Máy ảo Kaggle dùng chung CPU nên một đợt bị chiếm tài nguyên đẩy trung bình 30 lần lên 75 ms. Tôi giữ nguyên số đo trong
  notebook thay vì chạy lại cho đẹp, và không dùng dòng này để kết luận.
- **ONNX Runtime nhanh hơn PyTorch khoảng 20–25% ở phần inference** (46–50 ms so với 58–63 ms) trên cùng CPU. Đây là phần chiếm hơn 90% tổng thời gian.
- **Postprocess đúng như lý thuyết:** NMS của head one-to-many tăng từ 1.59 ms lên 2.18 ms khi hạ conf từ 0.25 xuống 0.001, vì có ~186 box ứng viên.
  Head one-to-one giữ 0.44 ms ở cả hai mức, nên ở conf 0.001 nhanh hơn khoảng 5 lần cho riêng bước này. PyTorch cho cùng xu hướng (1.07 → 1.97 ms so với 0.29–0.34 ms).
- **Trên tổng thời gian, lợi ích NMS-free có thật nhưng nhỏ với cảnh thưa:** ở conf 0.001 one-to-one nhanh hơn 0.85 ms (ONNX) và 1.7 ms (PyTorch), khoảng 2–3%
  tổng thời gian, nhỏ hơn cả độ dao động của inference giữa các lần đo. Trên CPU này, với một ảnh 5 người, giá trị chính của NMS-free là latency ổn định, không phụ thuộc
  số box, và export end-to-end gọn (không phải viết lại NMS ở runtime đích). Phần ms tiết kiệm chỉ lớn khi cảnh đông hoặc conf thấp đẩy số ứng viên lên hàng nghìn,
  hoặc trên NPU, nơi NMS phải quay về CPU host.
- **Số box ở conf 0.001 khác PyTorch** (186/177 so với 203/204). File ONNX export với input cố định 640×640 nên ảnh được pad thành hình vuông (8400 vị trí),
  còn PyTorch letterbox chữ nhật 640×480 (6300 vị trí, như Q1). Đầu vào khác nhau thì tập box ở ngưỡng rất thấp cũng khác. Ở conf 0.25 cả hai đều ra đúng 5 box.

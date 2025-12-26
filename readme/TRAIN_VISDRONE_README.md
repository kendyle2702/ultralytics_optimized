# YOLOv12n Training on VisDrone Dataset

## 📝 Mô tả

Scripts để training model YOLOv12n trên dataset VisDrone và đánh giá metrics trên tập test.

## 🚀 Cách sử dụng

### Option 1: Script đầy đủ (Khuyến nghị)

```bash
python train_yolov12n_visdrone.py
```

Script này sẽ:

- ✅ Training YOLOv12n từ đầu hoặc từ pretrained weights
- ✅ Tự động validate trong quá trình training
- ✅ Đánh giá trên validation set
- ✅ Đánh giá trên test set với đầy đủ metrics
- ✅ Lưu tất cả plots và kết quả

### Option 2: Script đơn giản

```bash
python train_yolov12n_simple.py
```

### Option 3: Command line trực tiếp

**Training:**

```bash
yolo train \
  model=ultralytics/cfg/models/12/yolo12.yaml \
  data=VisDrone.yaml \
  epochs=300 \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=yolo12n_visdrone
```

**Validate trên validation set:**

```bash
yolo val \
  model=runs/detect/yolo12n_visdrone/weights/best.pt \
  data=VisDrone.yaml \
  split=val \
  batch=16 \
  plots=True \
  save_json=True
```

**Test trên test set:**

```bash
yolo val \
  model=runs/detect/yolo12n_visdrone/weights/best.pt \
  data=VisDrone.yaml \
  split=test \
  batch=16 \
  plots=True \
  save_json=True
```

## 📊 Dataset VisDrone

Dataset sẽ tự động được tải về lần đầu tiên chạy.

**Cấu trúc:**

- Training: 6,471 images
- Validation: 548 images
- Test: 1,610 images

**Classes (10 classes):**

1. pedestrian
2. people
3. bicycle
4. car
5. van
6. truck
7. tricycle
8. awning-tricycle
9. bus
10. motor

## 📈 Metrics được đo

Sau khi training xong, bạn sẽ nhận được:

**Training metrics:**

- Loss curves (box, cls, dfl)
- Learning rate schedule
- mAP progression

**Validation metrics:**

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1-score

**Test metrics:**

- mAP@0.5
- mAP@0.5:0.95
- mAP@0.75
- Precision
- Recall
- Per-class AP

## 📁 Kết quả

Kết quả được lưu tại:

```
runs/detect/yolo12n_visdrone/
├── weights/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 curve
├── PR_curve.png         # Precision-Recall curve
├── P_curve.png          # Precision curve
├── R_curve.png          # Recall curve
└── args.yaml            # Training arguments
```

## ⚙️ Tùy chỉnh Hyperparameters

Bạn có thể điều chỉnh các hyperparameters trong script:

```python
model.train(
    epochs=300,  # Số epochs
    batch=16,  # Batch size (tăng nếu GPU đủ RAM)
    imgsz=640,  # Image size
    lr0=0.01,  # Learning rate
    optimizer="AdamW",  # Optimizer: SGD, Adam, AdamW, NAdam
    device=0,  # GPU device: 0, [0,1], 'cpu'
    workers=8,  # Số workers cho dataloader
    patience=100,  # Early stopping patience
    # ... và nhiều tham số khác
)
```

## 🔧 Yêu cầu

- Python 3.8+
- PyTorch 1.8+
- CUDA (khuyến nghị cho training)
- ultralytics package

## 💡 Tips

1. **GPU Memory:** Nếu bị out of memory, giảm `batch` size (ví dụ: 8, 4)
2. **Training time:** Với 300 epochs, training có thể mất vài giờ đến vài ngày tùy GPU
3. **Resume training:** Nếu bị gián đoạn, thêm `resume=True` để tiếp tục
4. **Multi-GPU:** Sử dụng `device=[0,1,2,3]` cho multi-GPU training

## 📚 Tài liệu tham khảo

- [Ultralytics Docs](https://docs.ultralytics.com)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [YOLOv12 Model](https://docs.ultralytics.com/models/yolo12)

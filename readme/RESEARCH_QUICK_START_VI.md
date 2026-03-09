# 🚀 Ultralytics YOLO - Hướng Dẫn Nhanh Cho Nhà Nghiên Cứu

## 📊 Tóm Tắt Khả Năng Tùy Chỉnh

| Thành phần         | Hỗ Trợ    | Cách Thực Hiện   | Độ Khó          |
| ------------------ | --------- | ---------------- | --------------- |
| **Backbone**       | ✅ Đầy đủ | Sửa YAML file    | ⭐ Dễ           |
| **Neck**           | ✅ Đầy đủ | Sửa YAML file    | ⭐ Dễ           |
| **Head/Detect**    | ✅ Đầy đủ | Sửa YAML file    | ⭐ Dễ           |
| **Loss Functions** | ✅ Đầy đủ | Tạo Python class | ⭐⭐ Trung bình |
| **Activation**     | ✅ Đầy đủ | YAML hoặc code   | ⭐ Dễ           |
| **Custom Modules** | ✅ Đầy đủ | Viết class mới   | ⭐⭐⭐ Khó      |

---

## ⚡ 5 Phút Bắt Đầu

### 1️⃣ Thay Đổi Backbone (Dễ Nhất)

```bash
# Bước 1: Copy YAML hiện có
cp ultralytics/cfg/models/11/yolo11.yaml my_custom_yolo.yaml

# Bước 2: Sửa file YAML
nano my_custom_yolo.yaml
```

Thay đổi lớp backbone:

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 2, C2f, [128, True]] # ← Thay C3k2 thành C2f
  - [-1, 1, Conv, [256, 3, 2]]
  # ... rest ...
```

```python
# Bước 3: Train
from ultralytics import YOLO

model = YOLO("my_custom_yolo.yaml")
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
)
```

### 2️⃣ Thêm Attention vào Neck

```yaml
# Sửa head section
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [512, 256, 8]] # ← Thêm Attention

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [256, 128, 8]] # ← Thêm Attention
  # ... rest ...
```

### 3️⃣ Thay Đổi Loss Function (Trung Bình)

```python
# Step 1: Tạo file custom_loss.py
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class CustomLoss(v8DetectionLoss):
    def __call__(self, preds, batch):
        # Thay đổi logic tính loss ở đây
        loss = super().__call__(preds, batch)
        return loss


class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        return CustomLoss(self)


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CustomDetectionModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model


# Step 2: Train
trainer = CustomTrainer(cfg=dict(model="yolo11n.yaml", data="coco8.yaml", epochs=100))
trainer.train()
```

---

## 🎯 Các Mô-đun Backbone Phổ Biến

```python
# Các module có sẵn để sử dụng:

# Convolution blocks
-Conv  # Standard convolution
-DWConv  # Depthwise convolution
-GhostConv  # Ghost convolution (lightweight)

# Bottleneck/Block
-C2f  # CSP block (YOLO8 standard)
-C3k2  # YOLO11 variant
-C3  # Older variant
-RepConv  # Reparameterized convolution
-C2fAttn  # C2f with attention

# Pooling
-SPPF  # Spatial Pyramid Pooling Fast
-SPP  # Spatial Pyramid Pooling

# Attention
-ImagePoolingAttn

# ResNet
-ResNetLayer
```

---

## 📋 Loss Functions Có Sẵn

```python
# Detection losses trong ultralytics/utils/loss.py:

v8DetectionLoss  # Standard YOLO detection loss
VarifocalLoss  # For class imbalance
FocalLoss  # Down-weight easy examples
BboxLoss  # Bounding box loss
DFLoss  # Distribution focal loss
KeypointLoss  # For pose estimation
v8SegmentationLoss  # For segmentation
v8PoseLoss  # For pose tasks
v8OBBLoss  # For rotated boxes
v8ClassificationLoss  # For classification
```

---

## 🔍 Cấu Trúc File YAML Chi Tiết

```yaml
# model.yaml format

nc: 80 # Số lớp (classes)
scales: # Scaling factors cho model sizes
  n: [0.33, 0.25, 1024] # [depth_mult, width_mult, max_channels]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

backbone: # Feature extraction part
  # Format: [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # Layer 0
  - [-1, 1, Conv, [128, 3, 2]] # Layer 1
  - [-1, 3, C2f, [128, True]] # Layer 2: repeat 3 times
  # ...

head: # Detection/task part
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # Multi-input layer
  - [-1, 3, C2f, [256]]
  # ...
  - [[15, 18, 21], 1, Detect, [nc]] # Output layer
```

### Chỉ Số Layer

- `-1`: Layer trước đó
- `N`: Layer thứ N
- `[N, M]`: Concatenate layers N và M

---

## 📊 Workflow Nghiên Cứu Tiêu Chuẩn

```
┌─────────────────────────┐
│ 1. Design Architecture  │  Tạo custom YAML
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│ 2. Create Config        │  Thiết lập hyperparameters
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│ 3. Train Baseline       │  yolo train model=yolo11n.yaml
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│ 4. Train Proposal       │  yolo train model=custom.yaml
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│ 5. Compare Results      │  So sánh mAP, FPS, params
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│ 6. Analyze & Report     │  Vẽ biểu đồ, viết báo cáo
└─────────────────────────┘
```

---

## 🔧 Các Lệnh Hữu Ích

```bash
# Training
yolo detect train model=yolo11n.yaml data=coco8.yaml epochs=100 batch=16

# Với custom YAML
yolo detect train model=cfg/models/11/custom.yaml data=coco8.yaml epochs=100

# Resume training
yolo detect train model=runs/detect/train/weights/last.pt resume

# Validation
yolo detect val model=runs/detect/train/weights/best.pt data=coco8.yaml

# Prediction
yolo detect predict model=runs/detect/train/weights/best.pt source=image.jpg

# Export model
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

---

## 🎨 Modules & Activation Functions

### Activation Functions

```yaml
# Trong YAML hoặc code:
activation: torch.nn.ReLU()          # ReLU
activation: torch.nn.SiLU()          # SiLU/Swish (default)
activation: torch.nn.GELU()          # GELU
activation: torch.nn.LeakyReLU()     # LeakyReLU
```

### Các Layers Hỗ Trợ

```
nn.Conv2d, nn.ConvTranspose2d
nn.BatchNorm2d
nn.ReLU, nn.SiLU, nn.GELU, nn.LeakyReLU
nn.MaxPool2d, nn.AdaptiveAvgPool2d
nn.Upsample
nn.Identity
```

---

## 📈 Metrics Để Track

```python
# Accuracy metrics
- mAP50-95 (chính)
- mAP50
- mAP75
- Precision
- Recall

# Speed metrics
- inference (ms)
- pre-process (ms)
- post-process (ms)
- Total FPS

# Model size
- Parameters (M)
- Model size (MB)
- FLOPs (G)

# Efficiency
- mAP per 1M params
- mAP per 1G FLOPs
```

---

## 📝 Template: Training Script Cho Paper

```python
#!/usr/bin/env python3
"""
Research: Custom YOLO Architecture
Objective: Achieve X% improvement over baseline.
"""

import json

from ultralytics import YOLO


def train_experiment(name, config):
    """Train single experiment."""
    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"{'=' * 60}")

    model = YOLO(config)
    results = model.train(
        data="coco8.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
        project="runs/research",
        name=name,
    )

    # Collect metrics
    metrics = {
        "mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
        "mAP50": results.results_dict["metrics/mAP50(B)"],
        "parameters": sum(p.numel() for p in model.parameters()) / 1e6,
        "inference_ms": results.results_dict["speed/inference(ms)"],
    }

    return metrics


# Main
if __name__ == "__main__":
    configs = {
        "baseline": "yolo11n.yaml",
        "custom_v1": "cfg/models/11/custom_backbone.yaml",
        "custom_v2": "cfg/models/11/custom_attention.yaml",
    }

    all_results = {}

    for name, config in configs.items():
        all_results[name] = train_experiment(name, config)

    # Save results
    with open("experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for name, metrics in all_results.items():
        print(f"\n{name}:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")
```

---

## 🎓 Tài Liệu Than Khảo

### Các File Chính

```
ultralytics/
├── cfg/models/          # Model configurations
├── nn/modules/          # Neural network modules
├── nn/tasks.py          # DetectionModel, parse_model()
├── engine/trainer.py    # BaseTrainer class
├── utils/loss.py        # Loss functions
└── utils/metrics.py     # Evaluation metrics
```

### Useful Classes/Functions

```python
# Import chính
from ultralytics import YOLO

# Common operations
model = YOLO("config.yaml")
results = model.train(...)
predictions = model.predict(...)
metrics = model.val()
```

---

## 🚨 Debugging Tips

```python
# 1. Kiểm tra model structure
model = YOLO("custom.yaml")
model.model.info()

# 2. Kiểm tra FLOPs và parameters
from fvcore.nn import FlopCounterMode

flops = FlopCounterMode(model.model).total()

# 3. Kiểm tra layer shapes
import torch

x = torch.randn(1, 3, 640, 640)
for i, layer in enumerate(model.model):
    x = layer(x)
    print(f"Layer {i}: {x.shape}")

# 4. Kiểm tra training
model.train(
    data="coco8.yaml",
    epochs=1,  # Just 1 epoch to check
    batch=4,  # Small batch
)
```

---

## 📞 Tài Liệu Chính Thức

- **Docs**: https://docs.ultralytics.com
- **GitHub**: https://github.com/ultralytics/ultralytics
- **Issues**: https://github.com/ultralytics/ultralytics/issues

---

## ✨ Lưu Ý Quan Trọng

1. **Luôn So Sánh với Baseline** - Kiểm tra cải tiến có ý nghĩa hay không
2. **Ghi Lại Tất Cả Siêu Than Số** - Để có thể reproduce
3. **Train Multiple Times** - Để có kết quả ổn định
4. **Kiểm Tra Kỹ Lưỡng** - Channel dimensions, layer connections
5. **Lưu Weights** - Để có thể sử dụng lại
6. **Công Bố Kết Quả** - Chia sẻ code và model weights

---

**Ultralytics Version**: 8.3.228 | **Last Updated**: Nov 2025

Tất cả file hướng dẫn chi tiết trong:

- `CUSTOMIZATION_GUIDE_VI.md` - Hướng dẫn toàn diện (trên 600 dòng)
- `PRACTICAL_EXAMPLES.md` - 6 ví dụ thực tế đầy đủ
- `RESEARCH_QUICK_START_VI.md` - File này (quick start)

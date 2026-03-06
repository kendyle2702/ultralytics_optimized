# Hướng Dẫn Tùy Chỉnh Ultralytics YOLO - Chi Tiết Toàn Diện

## 📋 Tổng Quan

Bạn đang sử dụng **Ultralytics v8.3.228** - phiên bản cuối cùng hỗ trợ YOLO11 (không phải YOLO26).

### ✅ Khả Năng Tùy Chỉnh Được Hỗ Trợ

Ultralytics YOLO **có đầy đủ hỗ trợ** để tùy chỉnh:

- ✅ **Backbone** - Thay đổi kiến trúc chiết xuất đặc trưng
- ✅ **Neck** - Sửa đổi kết nối giữa backbone và head
- ✅ **Loss Functions** - Thay đổi các hàm mất mát
- ✅ **Activation Functions** - Tùy chỉnh hàm kích hoạt
- ✅ **Modules** - Tạo hoặc thay thế các mô-đun tùy chỉnh

---

## 🏗️ Phần 1: Hiểu Cấu Trúc Kiến Trúc YOLO

### 1.1 Cấu Trúc Tệp YAML

Mỗi mô hình YOLO được định nghĩa bằng file YAML có ba phần chính:

```yaml
# Ví dụ: ultralytics/cfg/models/11/yolo11.yaml

nc: 80 # Số lớp
scales: # Hệ số tỷ lệ mô hình
  n: [0.50, 0.25, 1024] # [depth_multiple, width_multiple, max_channels]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# PHẦN BACKBONE - Chiết xuất đặc trưng
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # Layer 0: Convolution
  - [-1, 1, Conv, [128, 3, 2]] # Layer 1: Downsample
  - [-1, 2, C3k2, [256, False, 0.25]] # Layer 2: Repeated block
  - [-1, 1, Conv, [256, 3, 2]] # Layer 3
  - [-1, 2, C3k2, [512, False, 0.25]] # Layer 4
  - [-1, 1, SPPF, [1024, 5]] # Layer 9: Spatial Pyramid Pooling

# PHẦN NECK - Kết nối và tổng hợp đặc trưng
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # Upsample
  - [[-1, 6], 1, Concat, [1]] # Concatenate with skip connection
  - [-1, 2, C3k2, [512, False]] # Process merged features

  # Lặp lại cho các tỷ lệ khác nhau
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  # PHẦN HEAD - Phát hiện đối tượng
  - [[16, 19, 22], 1, Detect, [nc]] # Detect layer cho cả 3 tỷ lệ
```

### 1.2 Định Dạng Lớp: [from, repeats, module, args]

| Thành phần  | Ý Nghĩa               | Ví Dụ                                                          |
| ----------- | --------------------- | -------------------------------------------------------------- |
| **from**    | Kết nối từ layer nào  | `-1` (layer trước), `6` (layer 6), `[4,6,8]` (multiple inputs) |
| **repeats** | Lặp lại bao nhiêu lần | `1` (một lần), `3` (ba lần), `2` (hai lần)                     |
| **module**  | Loại mô-đun           | `Conv`, `C2f`, `C3k2`, `SPPF`, `Detect`                        |
| **args**    | Các than số mô-đun    | `[64, 3, 2]` (channels, kernel_size, stride)                   |

---

## 🔧 Phần 2: Tùy Chỉnh Backbone

### 2.1 Cách Thay Đổi Backbone

**Phương pháp A: Tạo file YAML tùy chỉnh**

```yaml
# custom_backbone.yaml
nc: 80

backbone:
  # Thay thế backbone YOLO11 bằng các layer tùy chỉnh
  - [-1, 1, Conv, [32, 3, 1]]
  - [-1, 1, Conv, [64, 3, 2]] # P1/2
  - [-1, 1, Conv, [128, 3, 2]] # P2/4
  - [-1, 3, C2f, [128, True]] # Sử dụng C2f block
  - [-1, 1, Conv, [256, 3, 2]] # P3/8
  - [-1, 6, C2f, [256, True]] # Nhiều block hơn
  - [-1, 1, Conv, [512, 3, 2]] # P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # Spatial Pyramid Pooling - FAST

head:
  # Sử dụng head tiêu chuẩn
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, C2f, [256]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]

  - [[15, 18, 21], 1, Detect, [nc]]
```

### 2.2 Các Mô-đun Backbone Có Sẵn

Các mô-đun bạn có thể sử dụng trong backbone:

**Convolution blocks:**

- `Conv` - Conv2d + BatchNorm + Activation
- `Conv2` - Depthwise separable convolution
- `DWConv` - Depthwise convolution
- `Focus` - Focus layer (input preprocessing)
- `GhostConv` - Ghost convolution (lightweight)

**Bottleneck blocks:**

- `Bottleneck` - Inverted bottleneck
- `BottleneckCSP` - CSP bottleneck
- `C2f` - CSPDarknet block (YOLO11 tiêu chuẩn)
- `C2fAttn` - C2f với attention
- `C3k2` - YOLO11 variant
- `RepConv` - Reparameterized convolution

**Pooling layers:**

- `SPPF` - Spatial Pyramid Pooling - Fast
- `SPP` - Spatial Pyramid Pooling
- `ImagePoolingAttn` - Image pooling attention

**Khác:**

- `ResNetLayer` - ResNet layer
- `HGStem` - HG Stem layer
- `HGBlock` - HG Block layer

### 2.3 Sử Dụng Backbone từ TorchVision

```yaml
# torchvision_backbone.yaml
nc: 80

backbone:
  # Sử dụng ConvNeXt từ TorchVision
  - [-1, 1, TorchVision, [768, convnext_tiny, DEFAULT, True, 2, True]]
  #                       [channels, model_name, pretrained, train_layers, layer_indices]

head:
  - [0, 1, Index, [192, 4]] # P3 features
  - [0, 1, Index, [384, 6]] # P4 features
  - [0, 1, Index, [768, 8]] # P5 features
  - [[1, 2, 3], 1, Detect, [nc]]
```

---

## 🔌 Phần 3: Tùy Chỉnh Neck

### 3.1 Neck là gì?

**Neck** (cổ) kết nối backbone với head, thường bao gồm:

- Upsampling layers để mở rộng lại kích thước
- Concatenation để kết hợp đặc trưng từ nhiều tỷ lệ
- Processing blocks để xử lý các đặc trưng hợp nhất

### 3.2 Ví Dụ: Tùy Chỉnh Neck

```yaml
# custom_neck.yaml
backbone:
  # Lấy từ YOLO11 standard
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # Layer 3
  - [-1, 2, C3k2, [512, False, 0.25]] # Layer 4
  - [-1, 1, Conv, [512, 3, 2]] # Layer 5
  - [-1, 2, C3k2, [512, True]] # Layer 6
  - [-1, 1, Conv, [1024, 3, 2]] # Layer 7
  - [-1, 2, C3k2, [1024, True]] # Layer 8
  - [-1, 1, SPPF, [1024, 5]] # Layer 9

head:
  # NECK - Custom upsampling path với nhiều processing
  - [-1, 1, Conv, [512, 1, 1]] # Layer 10: Reduce channels
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # Skip connection từ backbone layer 6
  - [-1, 3, C2f, [512]] # Layer 13: Intensive processing

  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # Skip connection từ backbone layer 4
  - [-1, 3, C2f, [256]] # Layer 16

  # Downsampling path - tương tự nhưng ngược lại
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 3, C2f, [512]] # Layer 19

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]] # Layer 22

  # HEAD - Detection
  - [[16, 19, 22], 1, Detect, [nc]]
```

### 3.3 Cấp Cao: Sử Dụng Attention trong Neck

```yaml
# attention_neck.yaml
backbone:
  # ... backbone layers ...
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [512, 256, 8]] # C2f với attention (channels, embed_dim, heads)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [256, 128, 8]]

  # Rest of head...
  - [[final_layer_indices], 1, Detect, [nc]]
```

---

## 💔 Phần 4: Thay Đổi Loss Functions

### 4.1 Loss Functions Có Sẵn

Ultralytics cung cấp các loss functions có sẵn:

**Cho Detection (trong `ultralytics/utils/loss.py`):**

```python
class v8DetectionLoss:
    """YOLOv8 Detection Loss."""

    # Thành phần:
    # - BCE Loss cho classification
    # - IoU Loss (CIoU) cho bounding boxes
    # - DFL Loss cho distribution focal loss


class VarifocalLoss(nn.Module):
    """Varifocal Loss - xử lý class imbalance."""

    # Than số: gamma (focusing), alpha (balancing)


class FocalLoss(nn.Module):
    """Focal Loss - down-weight easy examples."""


class BboxLoss(nn.Module):
    """Bounding Box Loss với DFL."""


class DFLoss(nn.Module):
    """Distribution Focal Loss."""
```

### 4.2 Cách Thay Đổi Loss Function

**Bước 1: Tạo Custom Loss Class**

```python
# custom_loss.py
import torch
import torch.nn as nn


class CustomDetectionLoss:
    """Custom Detection Loss cho research."""

    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]  # Detect module

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.reg_max = m.reg_max
        self.device = device

        # Import cần thiết từ ultralytics
        from ultralytics.utils.loss import BboxLoss
        from ultralytics.utils.tal import TaskAlignedAssigner

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)

    def __call__(self, preds, batch):
        """Compute custom loss."""
        # preds: model predictions
        # batch: ground truth batch

        # Implement your custom loss computation
        # ...

        loss = torch.zeros(3, device=self.device)  # [box, cls, dfl]

        # Your implementation
        # loss[0] = box_loss
        # loss[1] = cls_loss
        # loss[2] = dfl_loss

        return loss * batch_size, loss.detach()
```

**Bước 2: Override init_criterion trong Model**

```python
# models/yolo/detect/train.py hoặc custom trainer

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


class CustomDetectionModel(DetectionModel):
    def init_criterion(self):
        """Initialize custom loss criterion."""
        from custom_loss import CustomDetectionLoss

        return CustomDetectionLoss(self)


class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return custom detection model."""
        model = CustomDetectionModel(cfg or self.args.model, ch=3, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model
```

**Bước 3: Sử Dụng trong Training**

```python
# train_custom.py
from custom_trainer import CustomDetectionTrainer

# Cách 1: Sử dụng trực tiếp với command line
# yolo detect train model=yolo11n.yaml data=coco8.yaml epochs=100 trainer=CustomDetectionTrainer

# Cách 2: Sử dụng trong Python
trainer = CustomDetectionTrainer(cfg=dict(model="yolo11n.yaml", data="coco8.yaml", epochs=100), overrides={})
trainer.train()
```

### 4.3 Hiểu Loss Components

```python
# v8DetectionLoss breakdown
def __call__(self, preds, batch):
    """preds: [pred_scores, pred_dist, pred_bboxes] từ model batch: {img, bboxes, cls, ...} ground truth.
    """
    # 1. Task Aligned Assignment - xác định positive/negative samples
    target_scores, target_bboxes, _fg_mask = self.assigner(...)

    # 2. Classification Loss (BCE)
    loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores.sum()

    # 3. Bbox Regression Loss
    loss_iou = self.bbox_loss.compute_iou(pred_bboxes, target_bboxes)

    # 4. Distribution Focal Loss (DFL)
    loss_dfl = self.bbox_loss.dfl_loss(pred_dist, target_dist)

    # 5. Weighted combination
    total_loss = self.hyp.box * loss_iou + self.hyp.cls * loss_cls + self.hyp.dfl * loss_dfl

    return total_loss
```

### 4.4 Loss Weight Hyperparameters

```yaml
# Trong data YAML hoặc training config
box: 7.5 # Bounding box loss weight
cls: 0.5 # Classification loss weight
dfl: 1.5 # DFL loss weight (cho distribution)
```

---

## ⚡ Phần 5: Thay Đổi Activation Functions

### 5.1 Activation Functions Mặc Định

YOLO11 sử dụng **SiLU (Swish)** làm default activation.

### 5.2 Thay Đổi Activation

**Phương pháp A: Trong YAML**

```yaml
# custom_model.yaml
activation: torch.nn.ReLU() # Hoặc bất kỳ activation nào

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  # ... rest of architecture ...
```

**Phương pháp B: Thay Đổi Toàn Cục**

```python
# Trong training script
from ultralytics.nn.modules import Conv

# Thay đổi default activation
Conv.default_act = torch.nn.ReLU()

# Sau đó load model
model = YOLO("yolo11n.yaml")
```

**Phương pháp C: Custom Activation Classes**

```python
# Kết hợp nhiều activation
class CustomActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.silu(x) * 0.5 + self.relu(x) * 0.5


# Sử dụng trong model
Conv.default_act = CustomActivation()
```

---

## 📝 Phần 6: Tạo Custom Modules

### 6.1 Cơ Bản về Module Registration

```python
# ultralytics/nn/modules/__init__.py
# Các modules được tự động import và đăng ký

# Để thêm custom module:
# 1. Tạo module trong modules/
# 2. Import trong __init__.py
# 3. Sử dụng trong YAML
```

### 6.2 Tạo Custom Module

```python
# ultralytics/nn/modules/custom_module.py

import torch.nn as nn

from ..modules import Conv


class CustomBlock(nn.Module):
    """Custom feature extraction block."""

    def __init__(self, c1, c2, n=1, shortcut=False):
        """
        Args:
            c1: input channels
            c2: output channels
            n: number of blocks
            shortcut: use skip connection.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3)
        self.cv2 = Conv(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# Đăng ký trong __init__.py
```

### 6.3 Sử Dụng Custom Module trong YAML

```yaml
# custom_architecture.yaml
nc: 80

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 3, CustomBlock, [256, True]] # Sử dụng custom block
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  # ... head layers ...
```

---

## 🎯 Phần 7: Hướng Dẫn Cho Nghiên Cứu & Paper

### 7.1 Qui Trình Tối Ưu Kiến Trúc

```
1. BỎ ĐỀ NGHIÊN CỨU (Research Proposal)
   ├─ Xác định vấn đề cần giải quyết
   ├─ Đề xuất cải tiến (backbone/neck/loss)
   └─ Thiết lập baseline

2. TẠO CẤU HÌNH YAML
   ├─ Sao chép từ yolo11.yaml hoặc template
   ├─ Thay đổi backbone/neck
   ├─ Ghi chú các thay đổi
   └─ Đặt tên có nghĩa: custom_resnet_backbone.yaml

3. THỰC HIỆN THAY ĐỔI CODE
   ├─ Custom Loss function (nếu cần)
   ├─ Custom Module (nếu cần)
   ├─ Custom Trainer (nếu cần)
   └─ Kiểm tra import đúng

4. TRAINING & EVALUATION
   ├─ Train với baseline
   ├─ Train với proposal
   ├─ So sánh metrics (mAP, FPS, parameters)
   └─ Ghi lại kết quả

5. LẬP BÁOCÁO
   ├─ Architecture diagrams
   ├─ Bảng so sánh kết quả
   ├─ Phân tích thời gian & memory
   └─ Kết luận & hướng phát triển
```

### 7.2 Template: Sáng Tạo Kiến Trúc Tối Ưu

```yaml
# research_optimized_yolo.yaml
# Đề xuất: Backbone lẹ hơn + Neck bằng attention + Loss tối ưu

nc: 80
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]

# BACKBONE: Lightweight with Ghost convolutions
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, GhostBottleneck, [128, 128, 3]] # Ghost block tiết kiệm
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 3, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 3, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

# NECK: Enhanced with Attention
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [512, 256, 8]] # Attention mechanism

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C2fAttn, [256, 128, 8]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, C2f, [512]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 2, C2f, [1024]]

  - [[15, 18, 21], 1, Detect, [nc]]
```

### 7.3 Script Training Cho Nghiên Cứu

```python
# train_research.py
from ultralytics import YOLO

# Cấu hình thí nghiệm
EXPERIMENTS = {
    "baseline": "yolo11n.yaml",
    "custom_backbone": "research_optimized_yolo.yaml",
    "custom_loss": "yolo11n.yaml",  # với custom loss trainer
}

RESULTS = {}

for exp_name, config in EXPERIMENTS.items():
    print(f"\n{'=' * 60}")
    print(f"Experiment: {exp_name}")
    print(f"{'=' * 60}")

    # Load model
    model = YOLO(config)

    # Train
    results = model.train(
        data="coco8.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
        save=True,
        project="runs/research",
        name=exp_name,
    )

    # Validate
    metrics = model.val()

    # Store results
    RESULTS[exp_name] = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "params": sum(p.numel() for p in model.parameters()),
        "speed": metrics.speed["inference"],  # ms
    }

    # Predict
    predictions = model.predict("test_image.jpg", conf=0.25)

# So sánh kết quả
print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)

for name, metrics in RESULTS.items():
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
```

### 7.4 Metrics Để Ghi Lại

```python
# Metrics cần collect cho paper

metrics_to_track = {
    "Accuracy": ["mAP@0.5", "mAP@0.75", "mAP@0.5-0.95"],
    "Performance": ["FPS", "inference_time_ms", "throughput"],
    "Model Size": ["parameters", "model_size_MB", "FLOPs"],
    "Training": ["total_time", "convergence_epoch", "memory_usage_GB"],
    "Efficiency": ["params_per_mAP", "FLOPs_per_inference"],
}


# Tính toán efficiency metrics
def calculate_efficiency(mAP, params, FLOPs):
    return {
        "mAP_per_M_params": mAP / (params / 1e6),
        "mAP_per_G_FLOPs": mAP / (FLOPs / 1e9),
    }
```

### 7.5 Visualization cho Paper

```python
# visualization.py
import matplotlib.pyplot as plt


def plot_architecture_comparison(results):
    """Vẽ biểu đồ so sánh kiến trúc."""
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    names = list(results.keys())
    mAP = [results[n]["mAP50-95"] for n in names]
    params = [results[n]["params"] / 1e6 for n in names]  # M params
    fps = [1000 / results[n]["speed"] for n in names]  # FPS

    # Plot 1: mAP vs Parameters
    axes[0, 0].scatter(params, mAP, s=100)
    for i, name in enumerate(names):
        axes[0, 0].annotate(name, (params[i], mAP[i]))
    axes[0, 0].set_xlabel("Parameters (M)")
    axes[0, 0].set_ylabel("mAP@0.5-0.95")
    axes[0, 0].set_title("Accuracy vs Model Size")

    # Plot 2: mAP vs FPS
    axes[0, 1].scatter(fps, mAP, s=100, c="red")
    for i, name in enumerate(names):
        axes[0, 1].annotate(name, (fps[i], mAP[i]))
    axes[0, 1].set_xlabel("FPS")
    axes[0, 1].set_ylabel("mAP@0.5-0.95")
    axes[0, 1].set_title("Accuracy vs Speed")

    # Plot 3: Bar chart - mAP
    axes[1, 0].bar(names, mAP, color="skyblue")
    axes[1, 0].set_ylabel("mAP@0.5-0.95")
    axes[1, 0].set_title("Mean Average Precision")

    # Plot 4: Bar chart - Parameters
    axes[1, 1].bar(names, params, color="lightcoral")
    axes[1, 1].set_ylabel("Parameters (M)")
    axes[1, 1].set_title("Model Parameters")

    plt.tight_layout()
    plt.savefig("architecture_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_architecture_comparison(RESULTS)
```

---

## 🚀 Phần 8: Ví Dụ Thực Tế Toàn Bộ

### 8.1 Ví Dụ Đầy Đủ: Backbone Tối Ưu

```python
# backbone_optimization_research.py
"""
Research: Lightweight YOLO backbone using MobileNet-inspired blocks
Objective: Giảm parameters 30% trong khi duy trì accuracy 95% so với baseline.
"""

import torch.nn as nn

from ultralytics import YOLO


# 1. Define custom lightweight backbone module
class MobileNetBlock(nn.Module):
    def __init__(self, c1, c2, kernel=3, stride=1, groups=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, kernel, stride, (kernel - 1) // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu(self.bn1(self.dw(x)))
        x = self.silu(self.bn2(self.pw(x)))
        return x


# 2. Register custom module
import ultralytics.nn.modules as modules

modules.MobileNetBlock = MobileNetBlock

# 3. Create YAML config
yaml_content = """
# Lightweight YOLO with MobileNet backbone
nc: 80
scales:
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [32, 3, 2]]
  - [-1, 1, MobileNetBlock, [64]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 2, MobileNetBlock, [128]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, MobileNetBlock, [256]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, MobileNetBlock, [512]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 1, MobileNetBlock, [1024]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 7], 1, Concat, [1]]
  - [-1, 2, C2f, [512]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C2f, [256]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 2, C2f, [512]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C2f, [1024]]

  - [[15, 18, 21], 1, Detect, [nc]]
"""

with open("mobile_yolo.yaml", "w") as f:
    f.write(yaml_content)

# 4. Train
if __name__ == "__main__":
    # Baseline
    print("Training BASELINE (YOLOv11n)")
    model_baseline = YOLO("yolo11n.yaml")
    results_baseline = model_baseline.train(
        data="coco8.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        project="runs/research",
        name="baseline",
    )

    # Proposed
    print("\nTraining PROPOSED (Mobile YOLO)")
    model_mobile = YOLO("mobile_yolo.yaml")
    results_mobile = model_mobile.train(
        data="coco8.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        project="runs/research",
        name="mobile",
    )

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    params_baseline = sum(p.numel() for p in model_baseline.parameters())
    params_mobile = sum(p.numel() for p in model_mobile.parameters())
    reduction = (1 - params_mobile / params_baseline) * 100

    print(f"Baseline parameters: {params_baseline / 1e6:.1f}M")
    print(f"Mobile parameters: {params_mobile / 1e6:.1f}M")
    print(f"Reduction: {reduction:.1f}%")
```

---

## 🔍 Phần 9: Debugging & Best Practices

### 9.1 Kiểm Tra Kiến Trúc

```python
# Visualize model architecture
model = YOLO("custom_model.yaml")
model.model.info()

# Print detailed layer information
from ultralytics.utils.torch_utils import model_info

model_info(model.model)
```

### 9.2 Common Issues & Solutions

| Issue             | Solution                                                          |
| ----------------- | ----------------------------------------------------------------- |
| Channel mismatch  | Kiểm tra output channels của layer trước phải khớp input channels |
| Model quá lớn     | Giảm width_multiple hoặc channel sizes                            |
| Out of memory     | Giảm batch size hoặc imgsz                                        |
| Loss không giảm   | Kiểm tra learning rate, data loading, loss computation            |
| Shapes không khớp | Xác nhận layer output shapes qua `model.model.info()`             |

### 9.3 Performance Profiling

```python
# Profile model performance
import time

import torch
from fvcore.nn import FlopCounterMode

model = YOLO("custom_model.yaml")

# Measure inference time
x = torch.randn(1, 3, 640, 640)
start = time.time()
for _ in range(100):
    _ = model(x)
print(f"Inference time: {(time.time() - start) / 100:.4f}s")

# Calculate FLOPs
flops = FlopCounterMode(model.model).total() / 1e9
print(f"FLOPs: {flops:.2f}B")

# Model size
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params / 1e6:.2f}M")
```

---

## 📚 Phần 10: Tài Liệu Than Khảo

### Cấu Trúc Thư Mục Ultralytics

```
ultralytics/
├── cfg/
│   ├── default.yaml              # Default config
│   ├── models/                   # Model YAML definitions
│   │   ├── 11/                   # YOLO11 configs
│   │   ├── v8/                   # YOLOv8 configs
│   │   └── ...
│   └── datasets/                 # Dataset configs
├── data/
│   ├── augment.py                # Data augmentation
│   └── dataset.py                # Dataset classes
├── engine/
│   ├── trainer.py                # Base trainer
│   ├── predictor.py              # Predictor
│   └── validator.py              # Validator
├── models/
│   ├── yolo/
│   │   ├── detect/               # Detection task
│   │   ├── segment/              # Segmentation task
│   │   ├── classify/             # Classification task
│   │   └── pose/                 # Pose estimation
│   └── ...
├── nn/
│   ├── modules/                  # Network modules
│   ├── tasks.py                  # Task definitions
│   └── autobackend.py            # Auto backend selection
└── utils/
    ├── loss.py                   # Loss functions
    └── metrics.py                # Evaluation metrics
```

### Các File Cần Chỉnh Sửa

- **Custom Architecture**: `cfg/models/11/custom_model.yaml`
- **Custom Loss**: `ultralytics/utils/loss.py` hoặc tạo file mới
- **Custom Module**: `ultralytics/nn/modules/custom_module.py`
- **Custom Trainer**: `ultralytics/models/yolo/detect/train.py` hoặc override

---

## 💡 Tóm Tắt Nhanh

### ✅ Những Gì Có Thể Tùy Chỉnh

- ✅ **Backbone Architecture** - Thay đổi các layer chiết xuất
- ✅ **Neck Connections** - Thay đổi kết nối đặc trưng
- ✅ **Head/Detection Layer** - Tùy chỉnh phát hiện
- ✅ **Loss Functions** - Tạo custom loss
- ✅ **Activation Functions** - Thay đổi activation
- ✅ **Data Augmentation** - Tùy chỉnh augmentation
- ✅ **Training Hyperparameters** - Điều chỉnh learning rate, batch size, etc.

### 📝 Quy Trình 4 Bước

1. **Tạo YAML** - Copy và sửa file YAML từ templates
2. **Implement Code** - Tạo custom modules/losses nếu cần
3. **Train** - Chạy training với `model.train()`
4. **Evaluate** - So sánh metrics với baseline

### 🎯 Cho Paper/Conference

1. Đặt tên rõ ràng cho thí nghiệm
2. Ghi lại tất cả siêu than số
3. So sánh với baseline được công bố
4. Ghi lại parameters/FLOPs/FPS
5. Vẽ biểu đồ so sánh

---

## 📞 Liên Hệ & Hỗ Trợ

- **Official Docs**: https://docs.ultralytics.com
- **GitHub**: https://github.com/ultralytics/ultralytics
- **Issues**: https://github.com/ultralytics/ultralytics/issues

---

**Generated for Ultralytics YOLO v8.3.228**
